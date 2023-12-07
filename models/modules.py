import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup, QueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix
from transform3D import normalize_vector, ortho9d2matrix

from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils

class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim
        self.is_training = is_training
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)
        
    def forward(self, seed_features, end_points):
        """ Forward pass.

            Input:
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score
        if self.is_training:
            _, top_view_inds = torch.max(view_score, dim=2) #B, num_seed
        else:
            _, top_view_inds = torch.max(view_score, dim=2) #(B, num_seed)
            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()  # (B*num_seed*num_view*3)
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.
        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3
        # returned group features without xyz, normalize xyz as graspness paper
        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_fps, seed_xyz_graspable, seed_features_graspable, vp_rot):
        """ Forward pass.
            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors
            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        grouped_feature = self.grouper(seed_xyz_fps, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class BallCrop(nn.Module):
    """ Ball query group for grasp configure estimation. Return grouped points with a radiu.
        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            ball_radius: [float]
                radius of the ball space
    """
    def __init__(self, nsample, seed_feature_dim, ball_radius=0.08):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.ball_radius = ball_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3
        self.grouper = QueryAndGroup(radius=ball_radius, nsample=nsample,
                                             use_xyz=True, normalize_xyz=False)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_fps, seed_xyz_graspable, seed_features_graspable):
        """ Forward pass.
            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                seed_xyz_new: [torch.FloatTensor, (batch_size,num_seed_new,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size, C, num_seed)]
                
            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed)]
                    features of grouped points in different depths
        """
        grouped_feature = self.grouper(seed_xyz_fps, seed_xyz_graspable, 
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features


class SWADNet(nn.Module):
    """ Grasp configure estimation.
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module, 128
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size, 3, num_seed)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed = vp_features.size()
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        vp_features = vp_features.permute(0, 1, 4, 2, 3)
        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B, num_seed, num angle, num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        return end_points


class WarpNet(nn.Module):
    """ Additional transformation layer for P-Model.
        Input:
            num_angle: [int]
                number of in-plane rotation angle bins
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth bins
    """
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.LeakyReLU(), nn.Conv1d(256, 256, 1),nn.LeakyReLU(),nn.Conv1d(256, 512, 1))  # input feat dim need to be consistent with CloudCrop module, 128

    def forward(self, vp_features):
        """ Forward pass.
            Input:
                vp_features: [torch.FloatTensor, (batch_size, 3, num_seed)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed = vp_features.size()
        vp_features = vp_features + self.conv1(vp_features)
        return vp_features



class RelationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(RelationNet, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.Q = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.K = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.V = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, xin):
        x = self.embedding(xin)
        attention_maps = []
        for i in range(self.num_layers):
            x_att, attn_map = self.attention_layers[i](self.Q(x), self.K(x), self.V(x))
            attention_maps.append(attn_map)
            x = x + x_att 
            x_ff = self.feedforward_layers[i](x)
        return xin+x_ff, attention_maps[-1]

