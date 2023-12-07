from tqdm import tqdm
import numpy as np
import os
import scipy.io as scio


def simplify_grasp_labels(root, save_path):
    """
    original dataset grasp_label files have redundant data,  We can significantly save the memory cost
    """
    obj_names = list(range(18, 19))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in obj_names:
        print('\nsimplifying object {}:'.format(i))
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        point_num = len(label['points'])
        print('original shape:               ', label['points'].shape, label['offsets'].shape, label['scores'].shape)
        points = label['points']
        scores = label['scores']
        offsets = label['offsets']
        width = offsets[:, :, :, :, 2]
        print('after simplify, offset shape: ', points.shape, scores.shape, width.shape)
        np.savez(os.path.join(save_path, '{}_labels.npz'.format(str(i).zfill(3))),
                 points=points, scores=scores, width=width)


if __name__ == '__main__':
    root = '/data/datasets/graspnet/'
    save_path = os.path.join(root, 'grasp_label_simplified')
    simplify_grasp_labels(root, save_path)

