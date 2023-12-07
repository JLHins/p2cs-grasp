""" Loss functions for training.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_importance_reweight(d_s, d_t, valid_mask, w_type='point'):
    """
    input:
        d_s: absolute error of student and GT, shape:(B, M, num_angle, num_depth)
        d_t: aligned with student's points, shape(B, M, num_angle, num_depth)
        w_type: 'point' for point reweight 
                'grasp' for grasp reweight 
    """
    assert w_type in ['point', 'grasp'], 'type error.'
    if w_type=='point':
        B, N = valid_mask.shape[0], valid_mask.shape[1]
        d_s = d_s.reshape(B, N, -1).mean(-1)[valid_mask]
        d_t = d_t.reshape(B, N, -1).mean(-1)[valid_mask]
        zero_tensors = torch.zeros(d_s.shape).to(d_s.device)
        dif = torch.max(d_s-d_t, zero_tensors)
        dif_max = torch.max(dif) + 1e-8
        W_imp = dif / dif_max
    else:
        valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).repeat(1,1,12,4)
        d_s = d_s[valid_mask]
        d_t = d_t[valid_mask]
        zero_tens = torch.zeros(d_s.shape).to(d_s.device)
        dif = torch.max(d_s-d_t, zero_tens)
        dif_max = torch.max(dif) + 1e-8
        W_imp = dif / dif_max
    return W_imp


def get_loss(end_points, end_points_T=None):
    if end_points['is_distill']:  # training P-Model
        # Supervision from labels
        end_points['objectness_label'] = end_points['objectness_label_ori']  # B, 15000
        objectness_loss_stu, end_points = compute_objectness_loss(end_points)
        B, _,  num_pts = end_points['objectness_score'].shape
        end_points['batch_graspness_labels'] = end_points['batch_graspness_labels'][:, :num_pts]
        graspness_loss_stu, end_points = compute_graspness_loss(end_points)
        view_loss_stu, end_points = compute_view_graspness_loss(end_points)
        score_loss_stu, end_points = compute_score_loss(end_points)
        width_loss_stu, end_points = compute_width_loss(end_points)
        # Distillation from teacher (i.e. C-Model) to student (P-Model)
        seed_id_teacher_batch = end_points['seed_id_teacher']  # B, M
        valid_mask = (seed_id_teacher_batch>-1)
        seed_id_teacher_batch[~valid_mask] = 0  # filled with 0 temporarily，will be masked later
        # importance reweighting
        d_s = torch.abs(end_points['grasp_score_pred'] - end_points['batch_grasp_score'])  # B, M, 12, 4  
        pred_T_aligned = end_points_T['grasp_score_pred'].gather(1, seed_id_teacher_batch.unsqueeze(-1).unsqueeze(-1).repeat(1,1, 12, 4))
        d_t = torch.abs(pred_T_aligned - end_points['batch_grasp_score'])  # B, M, 12, 4  
        end_points['d_t'] = d_t
        end_points['d_s'] = d_s
        temperature = 5.0
        alpha = 0.5
        kl_loss_func = nn.KLDivLoss(reduction='batchmean') 
        objectness_input = F.log_softmax(end_points['objectness_score'].reshape(-1, 2)/temperature, dim=-1)
        objectness_target = F.softmax(end_points_T['objectness_score'][:, :, :num_pts].reshape(-1, 2)/temperature, dim=-1)
        objectness_loss_distill = kl_loss_func(objectness_input, objectness_target)
        end_points['loss/stage1_objectness_loss'] = alpha*objectness_loss_stu + (1-alpha)*objectness_loss_distill
        # graspness distill
        end_points['batch_graspness_labels'] = end_points_T['graspness_score'][:, :num_pts]
        graspness_loss_distill, end_points = compute_graspness_loss(end_points)
        end_points['loss/stage1_graspness_loss'] = alpha*graspness_loss_stu + alpha*graspness_loss_distill
        # structural relation distill
        sr_T = end_points_T['sr_map'].gather(1, seed_id_teacher_batch.unsqueeze(-1).repeat(1,1,2048))
        sr_loss, end_points = compute_structural_relation_loss(end_points, sr_T, valid_mask)
        end_points['loss/stage2_sr_loss'] = sr_loss
        # approaching direction distill
        num_view = end_points_T['view_score'].shape[-1]
        num_angle, num_depth = end_points['batch_grasp_width'].shape[-2:]
        end_points['batch_grasp_view_graspness'] = end_points_T['view_score'].gather(1, seed_id_teacher_batch.unsqueeze(-1).repeat(1,1,num_view))
        view_loss_distill, end_points = compute_view_graspness_loss(end_points, valid_mask)
        end_points['loss/stage2_view_loss'] = alpha*view_loss_stu + alpha*view_loss_distill
        # grasp score distill
        end_points['batch_grasp_score'] = end_points_T['grasp_score_pred']
        score_loss_KD, end_points = compute_score_loss(end_points, valid_mask)
        # grasp width distall
        end_points['batch_grasp_width'] = end_points_T['grasp_width_pred']/10.
        width_loss_KD, end_points = compute_width_loss(end_points, valid_mask)
        # feature distill
        group_feat_loss = nn.SmoothL1Loss(reduction='none')(end_points['FEATS_ORI'], end_points_T['FEATS'][:,:,:num_pts].detach())
        thres = torch.tensor([0.015]).to(group_feat_loss.device)
        thres = thres.expand_as(group_feat_loss)
        zero_tensor = torch.zeros(thres.shape,device=thres.device)
        group_feat_loss, _ = torch.max(torch.stack([zero_tensor, group_feat_loss-thres],-1),dim=-1)
        end_points['loss/stage1_feat_distill_loss'] = group_feat_loss.sum()/(group_feat_loss.size(0)*group_feat_loss.size(1)*group_feat_loss.size(2))
    loss = end_points['loss/stage1_objectness_loss'] + 10 * end_points['loss/stage1_graspness_loss'] \
         + 100 * end_points['loss/stage2_view_loss'] + 15 * end_points['loss/stage3_score_loss'] + 10 * end_points['loss/stage3_width_loss'] + 0.2*end_points['loss/stage1_feat_distill_loss'] + sr_loss + 0.5*width_loss_KD + 0.5*score_loss_KD
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['batch_graspness_labels'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points, valid_mask=None):
    criterion = nn.SmoothL1Loss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)  # B, M, 300
    if valid_mask is not None:
        num_view = view_score.shape[-1]
        mask_tmp = valid_mask.unsqueeze(-1).repeat(1, 1, num_view)
        mask_tmp2 = valid_mask.unsqueeze(-1).repeat(1, 1, 5)
        w_imp = compute_importance_reweight(end_points['d_s'], end_points['d_t'], valid_mask,'point').unsqueeze(-1).repeat(1, num_view).view(-1)
        loss = loss[mask_tmp] * w_imp
    loss = loss.mean() 
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points

def compute_structural_relation_loss(end_points, sr_T, valid_mask=None):
    criterion = nn.SmoothL1Loss(reduction='none')
    sr_map_stu = end_points['sr_map']
    loss = criterion(sr_map_stu, sr_T)  # B, M, M
    MASK = torch.bmm(valid_mask.float().unsqueeze(-1), valid_mask.float().unsqueeze(1)).bool()
    loss = loss[MASK].mean()
    end_points['loss/stage2_sr_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points, valid_mask=None):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    if valid_mask is not None:
        num_angle, num_depth = grasp_score_pred.shape[-2:]
        mask_tmp = valid_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_angle, num_depth)
        w_imp = compute_importance_reweight(end_points['d_s'], end_points['d_t'], valid_mask,'grasp')
        grasp_score_pred = grasp_score_pred[mask_tmp]
        grasp_score_label = grasp_score_label[mask_tmp]
        loss = (criterion(grasp_score_pred, grasp_score_label)*w_imp).mean()
    else:
        loss = criterion(grasp_score_pred, grasp_score_label).mean()
    if valid_mask is not None:
        end_points['loss/stage3_score_loss_KD'] = loss
    else:
        end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points, valid_mask=None):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    # norm by cylinder radius(Crop module) and norm to 0~1, original with is 0~0.1, /0.05->0~2, /2->0~1
    grasp_width_label = end_points['batch_grasp_width'] * 10   # norm by cylinder radius(Crop module)
    grasp_score_label = end_points['batch_grasp_score']
    if valid_mask is not None:
        num_angle, num_depth = grasp_width_pred.shape[-2:]
        mask_tmp = valid_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_angle, num_depth)
        grasp_width_pred = grasp_width_pred[mask_tmp]
        grasp_width_label = grasp_width_label[mask_tmp]
        grasp_score_label = grasp_score_label[mask_tmp]
        w_imp = compute_importance_reweight(end_points['d_s'], end_points['d_t'], valid_mask,'grasp')
        loss = criterion(grasp_width_pred, grasp_width_label)*w_imp
    else:
        loss = criterion(grasp_width_pred, grasp_width_label)
    loss_mask = grasp_score_label > 0
    loss = loss[loss_mask].mean()
    if valid_mask is not None:
        end_points['loss/stage3_width_loss_KD'] = loss
    else:
        end_points['loss/stage3_width_loss'] = loss
    return loss, end_points


def compute_relation_distill(end_points, end_points_T):
    criterion = nn.SmoothL1Loss(reduction='mean')
    attn_map_T = end_points_T['attn_map']
    attn_map_S = end_points['attn_map']
    loss = criterion(attn_map_S, attn_map_T)
    end_points['loss/stage2_relation_loss'] = loss
    return loss, end_points