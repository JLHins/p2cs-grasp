import os
import numpy as np
import random
import scipy.io as scio
from PIL import Image
import time

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points
from EMS.utilities import sampleSuperquadricTemplate as sample_sq
from EMS.utilities import samplingRingPoints as sample_ring
import open3d as o3d
from pointnet2_utils import furthest_point_sample as fps

class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, remove_invisible=False, augment=False, load_label=True, num_pts_stu=15000):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.num_points_graspness = 10000
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.num_pts_stu = num_pts_stu
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(0,100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.primitive_path = []
        self.img_per_scene = 256
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(self.img_per_scene):
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join('/data/datasets/graspnet', 'graspness_per_scene', x, camera, str(0).zfill(4) + '.npy'))  # dict: "grasp_points", "graspness"
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                self.primitive_path.append(os.path.join(root, 'scenes', x, camera, 'sq_labels', str(img_num).zfill(4) + '.npy'))
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list, pts_graspness_align):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            pts_graspness_align = transform_point_cloud(pts_graspness_align, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        pts_graspness_align = transform_point_cloud(pts_graspness_align, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        return point_clouds, object_poses_list, pts_graspness_align

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index):
        # color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # get valid points
        depth_mask = (depth > 0)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]

        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'coors_ori': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats_ori': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict

    def get_data_label(self, index):
        # color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        sceneid = int(index/self.img_per_scene+1e-3)
        imgid = index%self.img_per_scene
        extrinsics = np.load(self.root + '/scenes/scene_{}/{}/camera_poses.npy'.format(str(sceneid).zfill(4),self.camera))
        extrinsic_i = extrinsics[imgid]
        num_graspness = 0
        graspness_dict = np.load(self.graspnesspath[index], allow_pickle=True).item()  # for each point in frame0
        graspness, grasp_points_graspness = graspness_dict['graspness'], graspness_dict['grasp_points'] 
        if len(graspness) >= self.num_points_graspness:
            idxs = np.random.choice(len(graspness), self.num_points_graspness, replace=False)
        else:
            idxs1 = np.arange(len(graspness))
            idxs2 = np.random.choice(len(graspness), self.num_points_graspness - len(graspness), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        graspness = graspness[idxs]
        grasp_points_graspness = grasp_points_graspness[idxs]  # sample num_points_graspness graspness points
        pts_graspness_align = np.matmul(np.linalg.inv(extrinsic_i), np.concatenate([grasp_points_graspness, np.ones([len(grasp_points_graspness), 1])], -1).T)[:3, :].T  # in frame i
        
        num_graspness += len(graspness)
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # get valid points
        depth_mask = (depthori > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            cam_wrt_cam0 = camera_poses[self.frameid[index]]
            trans = np.dot(align_mat, cam_wrt_cam0)
            workspace_mask = get_workspace_mask(cloudori, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloudori[mask]
        seg_masked = seg[mask]
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        pts_append = []
        for i, obj_idx in enumerate(obj_idxs):
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)
            # get extra complete points in C-Model
            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 500), len(points)), replace=False)
            grasp_points_trans = transform_point_cloud(points[idxs], poses[:, :, i], '3x4')
            pts_append.append(grasp_points_trans)
            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[:, :, i], th=0.01)
                points = points[visible_mask]
                widths = widths[visible_mask]
                scores = scores[visible_mask]
                collision = collision[visible_mask]
            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
        pts_append = np.concatenate(pts_append, 0)
        # sample extra points with FPS
        num_pts_primitive = 4000
        pts_append_tensor =  torch.tensor(pts_append).cuda().unsqueeze(0).contiguous().float()
        pts_append_sampled_idx = fps(pts_append_tensor, num_pts_primitive).cpu().squeeze(0).numpy()
        pts_append_sampled = pts_append[pts_append_sampled_idx]
        
        cloud_sampled = np.concatenate([cloud_sampled, pts_append_sampled], 0)
        objectness_label_ori = objectness_label.copy()
        objectness_label = np.concatenate([objectness_label, np.ones(num_pts_primitive)])
        
        if self.augment:
            cloud_sampled, object_poses_list, pts_graspness_align = self.augment_data(cloud_sampled, object_poses_list, pts_graspness_align)

        cloud_sampled_ori = cloud_sampled[:self.num_pts_stu, :]
        objectness_label_ori = objectness_label_ori[:self.num_pts_stu]
        cloud_sampled_append = cloud_sampled[self.num_pts_stu:, :]
        
        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'point_clouds_ori': cloud_sampled_ori.astype(np.float32),
                    'point_clouds_append': cloud_sampled_append.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'coors_ori': cloud_sampled_ori.astype(np.float32) / self.voxel_size,
                    'feats_ori': np.ones_like(cloud_sampled_ori).astype(np.float32),
                    'graspness_label_list': graspness.astype(np.float32),
                    'grasp_points_graspness_list': pts_graspness_align.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'objectness_label_ori': objectness_label_ori.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list,}
        return ret_dict


def load_grasp_labels(root):
    obj_names = list(range(1, 89))  # 88 objects in Graspnet-1B
    grasp_labels = {}
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))
    return grasp_labels


def minkowski_collate_fn(list_data):
    # C-Model
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    # P-Model
    coordinates_batch_ori, features_batch_ori = ME.utils.sparse_collate([d["coors_ori"] for d in list_data],
                                                                [d["feats_ori"] for d in list_data])
    coordinates_batch_ori, features_batch_ori, _, quantize2original_ori = ME.utils.sparse_quantize(
        coordinates_batch_ori, features_batch_ori, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original,
        "coors_ori": coordinates_batch_ori,
        "feats_ori": features_batch_ori,
        "quantize2original_ori": quantize2original_ori,
        # "labels": labels_batch,
    }
    coordinates_batch_ = coordinates_batch.cpu().numpy()

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):   # this is for list, return list, so train.py to device need list
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats' or key == 'coors_ori' or key=="feats_ori":
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
