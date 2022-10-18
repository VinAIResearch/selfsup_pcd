import h5py
import numpy as np
import torch
import torch.utils.data as data
from data_utils import center_point_cloud, normalize_point_cloud, translate_pointcloud


class ScanObjectNNDataset(data.Dataset):
    def __init__(self, root, npoints=1024, split="train", small_data=False, ratio=10, data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.small_data = small_data
        self.ratio = ratio
        if self.split == "train":
            if small_data:
                f = h5py.File("%s/%s" % (self.root, "training_objectdataset_%d.h5" % (self.ratio)), "r")
            else:
                f = h5py.File("%s/%s" % (self.root, "training_objectdataset.h5"), "r")
            self.data = f["data"][:]
            self.label = f["label"][:]
        else:
            f = h5py.File("%s/%s" % (self.root, "test_objectdataset.h5"), "r")
            self.data = f["data"][:]
            self.label = f["label"][:]
        labels = list(set(self.label))
        self.classes = dict(zip(sorted(labels), range(len(labels))))
        self.num_classes = len(labels)

    def __getitem__(self, index):
        point_set = np.copy(self.data[index])
        cls = self.label[index]
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
