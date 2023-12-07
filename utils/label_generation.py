""" Dynamically generate grasp labels during training.
    Author: chenxi-wang
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'knn'))

from knn.knn_modules import knn
from loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix, \
    transform_point_cloud, generate_grasp_views


def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # (B, M_point, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        poses = end_points['object_poses_list'][i]  # [(3, 4, n_obj),]

        # get merged grasp points for label computation
        grasp_points_merged = []
        # grasp_views_merged = []
        grasp_views_rot_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        for obj_idx, pose in enumerate(poses):
            pose = pose.cuda()
            grasp_points = end_points['grasp_points_list'][i][obj_idx].cuda()  # (Np, 3)
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx].cuda()  # (Np, V, A, D)
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx].cuda()  # (Np, V, A, D)
            Npts, V, A, D = grasp_scores.size()
            num_grasp_points = grasp_points.size(0)
            # generate and transform template grasp views
            grasp_views = generate_grasp_views(V).to(pose.device)  # (V, 3)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            
            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

            # assign views
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1

            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1,
                                                                              -1)  # (Np, V, 3, 3)
            grasp_scores = torch.index_select(grasp_scores, 1, view_inds)  # (Np, V, A, D)
            grasp_widths = torch.index_select(grasp_widths, 1, view_inds)  # (Np, V, A, D)
            
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # (Np', V, A, D)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', V, A, D)

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Ns)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # (Ns)

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ns, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)  # (Ns, V, A, D)

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(grasp_views_rot_merged)
        batch_grasp_scores.append(grasp_scores_merged)
        batch_grasp_widths.append(grasp_widths_merged)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)

    # compute graspness
    pts_graspness = end_points['grasp_points_graspness_list'].cuda()
    graspness = end_points['graspness_label_list'].cuda()
    
    graspness_label_batch = []
    for i in range(batch_size):
      pts_graspness_align = pts_graspness[i].T.contiguous().unsqueeze(0) #torch.matmul(pose_cam_wrt_cam0[i].float(), torch.cat([pts_graspness[i], torch.ones(len(pts_graspness[i]), 1).to(pts_graspness.device)], -1).T)[:3, :].contiguous().unsqueeze(0)
      nn_inds = knn(pts_graspness_align, end_points['point_clouds'][i].T.contiguous().unsqueeze(0), k=1).squeeze() - 1

      graspness_i = torch.index_select(graspness[i], 0, nn_inds)
      graspness_sampled_pts = torch.index_select(pts_graspness_align[0].T, 0, nn_inds)
      ''' 
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(end_points['point_clouds'][i][:,:].cpu().numpy())
      pcd2 = o3d.geometry.PointCloud()
      pcd2.points = o3d.utility.Vector3dVector(pts_graspness_align[0].T.cpu().numpy())
      o3d.visualization.draw_geometries([pcd, pcd2])
      '''
      graspness_label_invalid_mask = ((graspness_sampled_pts-end_points['point_clouds'][i]).max(1).values > 0.02)
      graspness_i[graspness_label_invalid_mask] = 0
      graspness_label_batch.append(graspness_i)
    graspness_label_batch = torch.stack(graspness_label_batch, 0)  # B, 1w, 1
    
    # compute view graspness
    view_u_threshold = 0.6
    view_grasp_num = 48
    batch_grasp_view_valid_mask = (batch_grasp_scores <= view_u_threshold) & (batch_grasp_scores > 0) # (B, Ns, V, A, D)
    batch_grasp_view_valid = batch_grasp_view_valid_mask.float()
    batch_grasp_view_graspness = torch.sum(torch.sum(batch_grasp_view_valid, dim=-1), dim=-1) / view_grasp_num  # (B, Ns, V)
    view_graspness_min, _ = torch.min(batch_grasp_view_graspness, dim=-1)  # (B, Ns)
    view_graspness_max, _ = torch.max(batch_grasp_view_graspness, dim=-1)
    view_graspness_max = view_graspness_max.unsqueeze(-1)  # (B, Ns, 1)
    view_graspness_min = view_graspness_min.unsqueeze(-1) 
    batch_grasp_view_graspness = (batch_grasp_view_graspness - view_graspness_min) / (view_graspness_max - view_graspness_min + 1e-8)

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores[~label_mask] = 0

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_view_rot'] = batch_grasp_views_rot
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness
    end_points['batch_graspness_labels'] = graspness_label_batch
    return end_points


def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    template_views_rot = end_points['batch_grasp_view_rot']  # (B, Ns, V, 3, 3)
    grasp_scores = end_points['batch_grasp_score']  # (B, Ns, V, A, D)
    grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, D, 3)

    B, Ns, V, A, D = grasp_scores.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_scores = torch.gather(grasp_scores, 2, top_view_inds_).squeeze(2)
    top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)

    u_max = top_view_grasp_scores.max()
    po_mask = top_view_grasp_scores > 0  # positive mask, for this part grasp score, apply 0~1 norm
    if po_mask.sum() > 0:
        u_min = top_view_grasp_scores[po_mask].min()
        top_view_grasp_scores[po_mask] = torch.log(u_max / top_view_grasp_scores[po_mask]) / torch.log(u_max / u_min)
    end_points['batch_grasp_score'] = top_view_grasp_scores  # (B, Ns, A, D)
    
    end_points['batch_grasp_width'] = top_view_grasp_widths  # (B, Ns, A, D)
    return top_template_views_rot, end_points


def process_grasp_labels_rot(end_points):
    """ Process labels according to scene points and object poses. """
    seed_xyzs = end_points['xyz_graspable']  # (B, M_point, 3)
    batch_size, num_samples, _ = seed_xyzs.size()

    batch_grasp_points = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    batch_grasp_rots = []
    batch_grasp_depths = []
    batch_grasp_points_nn_dist = []
    graspness_label_batch = []
    for i in range(batch_size):
        seed_xyz = seed_xyzs[i]  # (Ns, 3)
        poses = end_points['object_poses_list'][i]  # [(3, 4, n_obj),]
        grasp_points_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        grasp_rots_merged = []
        grasp_depths_merged = []
        for obj_idx, pose in enumerate(poses):
            pose = pose.cuda()
            grasp_points = end_points['grasp_points_list'][i][obj_idx].cuda()  # (Np, 3)
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx].cuda()  # (Np, 1)
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx].cuda()  # (Np, 1)
            grasp_depths = end_points['grasp_depths_list'][i][obj_idx].cuda()  # (Np, 1)
            grasp_rots = end_points['grasp_rots_list'][i][obj_idx].cuda()  # (Np, 3, 3)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_rots_trans = torch.matmul(pose[:3, :3], grasp_rots.view(-1, 3, 3))  # Np, 3, 3
            grasp_points_merged.append(grasp_points_trans)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)
            grasp_depths_merged.append(grasp_depths)
            grasp_rots_merged.append(grasp_rots_trans)
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', 1)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # (Np', 1)
        grasp_depths_merged = torch.cat(grasp_depths_merged, dim=0)  # (Np', 1)
        grasp_rots_merged = torch.cat(grasp_rots_merged, dim=0)      # (Np', 3, 3)
        grasp_points_merged_knn = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Ns)

        nn_dis, nn_inds = torch.topk(torch.cdist(seed_xyz_.permute(0, 2, 1), grasp_points_merged_knn.permute(0, 2, 1)), k=1, largest=False)
        nn_inds = nn_inds.squeeze().detach()  # Ns,
        nn_dis = nn_dis.squeeze().detach()  # Ns,
        grasp_points_merged_ = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ns, 3)
        grasp_scores_merged_ = torch.index_select(grasp_scores_merged, 0, nn_inds)  # (Ns, 1)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)  # (Ns, 1)
        grasp_rots_merged = torch.index_select(grasp_rots_merged, 0, nn_inds)  # (Ns, 3, 3)
        grasp_depths_merged = torch.index_select(grasp_depths_merged, 0, nn_inds)  # (Ns, 1)
        batch_grasp_points.append(grasp_points_merged_)
        batch_grasp_scores.append(grasp_scores_merged_)
        batch_grasp_widths.append(grasp_widths_merged)
        batch_grasp_rots.append(grasp_rots_merged)
        batch_grasp_depths.append(grasp_depths_merged)
        batch_grasp_points_nn_dist.append(nn_dis)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_rots = torch.stack(batch_grasp_rots, 0)  # (B, Ns, 3, 3)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, 1)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, 1)
    batch_grasp_depths = torch.stack(batch_grasp_depths, 0)  # (B, Ns, 1)
    batch_grasp_points_nn_dist = torch.stack(batch_grasp_points_nn_dist, 0)  # (B, Ns,)

    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)  # (B, Ns, V, A, D)
    batch_grasp_scores[~label_mask] = 0
    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_rot'] = batch_grasp_rots
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_depth'] = batch_grasp_depths
    end_points['batch_grasp_points_nn_dist'] = batch_grasp_points_nn_dist

    # assign point score
    pts_graspness = end_points['grasp_points_graspness_list'].cuda()
    graspness = end_points['graspness_label_list'].cuda()
    graspness_label_batch = []
    for i in range(batch_size):
        pts_graspness_align = pts_graspness[i].T.contiguous().unsqueeze(0) 
        nn_inds = knn(pts_graspness_align, end_points['point_clouds'][i].T.contiguous().unsqueeze(0), k=1).squeeze() - 1
        graspness_i = torch.index_select(graspness[i], 0, nn_inds)
        graspness_sampled_pts = torch.index_select(pts_graspness_align[0].T, 0, nn_inds)
        graspness_label_invalid_mask = ((graspness_sampled_pts-end_points['point_clouds'][i]).max(1).values > 0.007)  
        graspness_i[graspness_label_invalid_mask] = 0
        graspness_label_batch.append(graspness_i)
    graspness_label_batch = torch.stack(graspness_label_batch, 0)  
    end_points['batch_graspness_labels'] = graspness_label_batch
    return end_points
