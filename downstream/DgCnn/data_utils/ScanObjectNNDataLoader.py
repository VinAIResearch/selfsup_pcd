import glob
import os
import os.path
import sys

import h5py
import numpy as np
import torch
import torch.utils.data as data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
sys.path.append(os.path.join(BASE_DIR, "data_utils"))
from data_utils import (
    center_point_cloud,
    jitter_point_cloud,
    normalize_point_cloud,
    rotate_point_cloud,
    translate_pointcloud,
)


class ScanObjectNNDataset(data.Dataset):
    def __init__(self, root, npoints=1024, split="train", small_data=False, ratio=10, data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.small_data = small_data
        self.ratio = ratio
        idx = 0
        if self.split == "train":
            if small_data:
                f = h5py.File("%s/%s" % (self.root, "training_objectdataset_%d.h5" % (self.ratio)), "r")
            else:
                f = h5py.File("%s/%s" % (self.root, "training_objectdataset.h5"), "r")
            # print('Key: ',list(f.keys()))
            # print('Shape:',f['data'].shape)
            # idx_pts = np.arange(f['data'].shape[1])
            # np.random.shuffle(idx_pts)
            # self.data = f['data'][:][:,idx_pts[:self.npoints],:]
            self.data = f["data"][:]
            self.label = f["label"][:]
        else:
            f = h5py.File("%s/%s" % (self.root, "test_objectdataset.h5"), "r")
            # idx_pts = np.arange(f['data'].shape[1])
            # np.random.shuffle(idx_pts)
            # self.data = f['data'][:][:,idx_pts[:self.npoints],:]
            self.data = f["data"][:]
            self.label = f["label"][:]
        labels = list(set(self.label))
        self.classes = dict(zip(sorted(labels), range(len(labels))))
        self.num_classes = len(labels)

    def __getitem__(self, index):
        point_set = np.copy(self.data[index])
        cls = self.label[index]
        # print(np.max(np.sqrt(np.sum(point_set**2, axis=1))))
        point_set = center_point_cloud(point_set)
        point_set = normalize_point_cloud(point_set)
        point_set = point_set[0 : self.npoints, :]
        if self.data_augmentation:
            # print('test')
            # point_set = rotate_point_cloud(point_set)
            # point_set = jitter_point_cloud(point_set)
            point_set = translate_pointcloud(point_set)
        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data = ScanObjectNNDataset(root="../data/h5_files/main_split", npoints=1024, split="test")
    print(data[0][0].size())
