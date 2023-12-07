import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from models.backbone_resunet14SE import MinkUNet14, MinkUNet14D, MinkUNet14C
from models.modules import ApproachNet, GraspableNet, CloudCrop, SWADNet, WarpNet, RelationNet
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation, ball_query

M_POINT=2048 
class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True, is_teacher=False, num_points=15000):
        super().__init__()
        self.is_training = is_training
        self.is_teacher = is_teacher
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
        self.num_points = num_points
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        self.approaching = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)
        self.warping = WarpNet(num_angle=self.num_angle, num_depth=self.num_depth)
        self.relation_net = RelationNet(input_dim=512, hidden_dim=512, num_layers=1, num_heads=4)


    def forward(self, end_points, end_points_T=None, net_teacher=None):
        if end_points_T is not None:  # P-Model
            seed_xyz = end_points['point_clouds_ori']  # use all sampled point cloud, B*Ns*3
            B, point_num, _ = seed_xyz.shape  # batch _size
            coordinates_batch = end_points['coors_ori']
            features_batch = end_points['feats_ori']
            mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
            seed_features = self.backbone(mink_input)
            seed_features = seed_features.F
            seed_features = seed_features[end_points['quantize2original_ori']].view(B, point_num, -1).transpose(1, 2)
            seed_features = self.warping(seed_features)  # additional transformation for P-Model
            end_points['FEATS_ORI'] = seed_features      # for feature distillation
        else:  # C-Model
            seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
            B, point_num, _ = seed_xyz.shape  # batch_size
            coordinates_batch = end_points['coors']
            features_batch = end_points['feats']
            mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
            seed_features = self.backbone(mink_input)
            seed_features = seed_features.F
            seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
            # align to partial view
            seed_features = seed_features[:, :, :15000]
            seed_xyz = seed_xyz[:, :15000, :]
            end_points['SEEDXYZ'] = seed_xyz
            end_points['FEATS'] = seed_features  # for feature distillation

        end_points = self.graspable(seed_features, end_points)  
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD
        graspable_mask = objectness_mask & graspness_mask
        end_points['graspable_mask'] = graspable_mask

        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_xyz_fps = []

        graspable_num_batch = 0.
        for i in range(B):
            graspness_score_i = graspness_score[i]  # N 
            cur_mask = graspable_mask[i]
            graspable_num_batch += cur_mask.sum()
            if cur_mask.sum() == 0:
                continue
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim
            cur_seed_xyz_ = seed_xyz[i][cur_mask]  # Ns*3
            cur_seed_xyz = cur_seed_xyz_.unsqueeze(0) # 1*Ns*3
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points)
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*Ns
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # Ns*3
            seed_xyz_fps.append(cur_seed_xyz)
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous()
            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)

        if len(seed_xyz_graspable) == 0:
            end_points['xyz_graspable'] = None
            return end_points
        seed_xyz_fps = torch.stack(seed_xyz_fps, 0)  # B, Ns, 3
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*Ns*3
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*Ns
        seed_features_graspable, sr_map = self.relation_net(seed_features_graspable.permute(2,0,1))
        seed_features_graspable = seed_features_graspable.permute(1,2,0)
        end_points['sr_map'] = sr_map

        if end_points_T is not None:
            dist, seed_id_teacher = torch.cdist(seed_xyz_graspable, end_points_T['xyz_graspable']).min(-1)
            seed_id_teacher[dist>0.002] = -1
            end_points['seed_id_teacher'] = seed_id_teacher  # idx mapping，-1 means 'no matching'
        end_points['xyz_graspable'] = seed_xyz_graspable
        
        end_points['graspable_count_stage1'] = graspable_num_batch / B
        end_points['seed_features_graspable'] = seed_features_graspable.transpose(1,2) # B, Ns, feat_dim
        
        end_points, res_feat = self.approaching(seed_features_graspable, end_points)
        seed_features_graspable = seed_features_graspable + res_feat  # residual feat from view selection

        if self.is_training:  
            end_points = process_grasp_labels(end_points)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
        
        group_features = self.crop(seed_xyz_fps.contiguous(), seed_xyz_graspable.contiguous(), seed_features_graspable.contiguous(), grasp_top_views_rot)
        end_points = self.swad(group_features, end_points)
        return end_points


def pred_decode(end_points):
    M_POINT = 2048 
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()[:M_POINT,:]
        grasp_score = end_points['grasp_score_pred'][i].float()[:M_POINT,:,:]
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i][:M_POINT,:,:] / 10.  # grasp width gt has been multiplied by 10
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)
        approaching = -end_points['grasp_top_view_xyz'][i][:M_POINT,:].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)
        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)  # finger height, default: 2cm
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds
