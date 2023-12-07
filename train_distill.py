import os
import sys
import numpy as np
from datetime import datetime
import argparse
import random
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet_distill import GraspNet
from models.loss_distill import get_loss
from dataset.graspnet_dataset_complete_distill import GraspNetDataset, minkowski_collate_fn, load_grasp_labels


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/datasets/graspnet')#'/media/bot/980A6F5E0A6F38801/datasets/graspnet')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path',
                    default='')
parser.add_argument('--model_name', type=str,
                    default='')
parser.add_argument('--log_dir', default='/data/logs/')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--num_pts_stu', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 10]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.0007, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', default=0, type=int, help='Whether to resume from checkpoint')
parser.add_argument('--teaching_teacher', default=0, type=int, help='Whether to train teacher')
parser.add_argument('--distillation', default=1, type=int, help='Whether to perform distillation')
parser.add_argument('--teacher_ckpt_path', help='Teacher Model checkpoint path',
                    default='')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, remove_invisible=False, load_label=True,num_pts_stu=cfgs.num_pts_stu)
print('train dataset length: ', len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length: ', len(TRAIN_DATALOADER))



net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

if cfgs.distillation:
    net_teacher = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False, is_teacher=True)
    net_teacher.to(device)
    ckpt_teacher = torch.load(cfgs.teacher_ckpt_path)
    net_teacher.load_state_dict(ckpt_teacher['model_state_dict'], strict=False)
    for param in net_teacher.parameters():
        param.requires_grad=False
    print('teacher model loaded')


# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print('cannot load optimizer') 
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    with tqdm.tqdm(TRAIN_DATALOADER) as tepoch:
        for batch_idx, batch_data_label in enumerate(tepoch):
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)
            
            if cfgs.distillation:
                end_points_T = net_teacher(batch_data_label)
                end_points = net(batch_data_label, end_points_T)

            else:
                end_points = net(batch_data_label)
                end_points_T = None
            end_points['is_distill'] = cfgs.distillation

            loss, end_points = get_loss(end_points, end_points_T)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()
            if (batch_idx + 1) % batch_interval == 0:
                log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
                for key in sorted(stat_dict.keys()):
                    TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                            (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                    log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                    stat_dict[key] = 0


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        np.random.seed()
        train_one_epoch()
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
