# Ref https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
import os
import os.path

import h5py
import indoor3d_util
import numpy as np
import torch
import torch.utils.data as data


class S3DISDataset(data.Dataset):
    def __init__(self, root, num_points=4096, split="train", test_area=6):
        self.root = root
        self.num_points = num_points
        self.split = split
        self.test_area = test_area
        self.all_file_list = [line.strip() for line in open(os.path.join(self.root, "all_files.txt"))]
        self.room_file_list = [line.strip() for line in open(os.path.join(self.root, "room_filelist.txt"))]
        self.data_batches_list = []
        self.label_batches_list = []

        for file_name in self.all_file_list:
            print(file_name)
            f = h5py.File(file_name, "r")
            self.data_batches_list.append(f["data"][:])
            self.label_batches_list.append(f["label"][:])
        self.data_batches = np.concatenate(self.data_batches_list, 0)
        self.label_batches = np.concatenate(self.label_batches_list, 0)

        test_area = "Area_%d" % test_area
        train_inds = []
        test_inds = []
        print("Filter train test")
        for i, room in enumerate(self.room_file_list):
            if test_area in room:
                test_inds.append(i)
            else:
                train_inds.append(i)
        if self.split == "train":
            self.data_batches = self.data_batches[train_inds]
            self.label_batches = self.label_batches[train_inds]
        else:
            self.data_batches = self.data_batches[test_inds]
            self.label_batches = self.label_batches[test_inds]

    def __getitem__(self, index):
        points = self.data_batches[index]
        seg = self.label_batches[index]
        return torch.from_numpy(points), torch.from_numpy(seg)

    def __len__(self):
        return len(self.label_batches)


class S3DISDatasetWholeScene:
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, test_area=6, stride=1.0, block_size=1.0):
        self.block_points = block_points
        self.block_size = block_size
        self.root = root
        self.stride = stride
        self.file_list = [d for d in os.listdir(root) if d.find("Area_%d" % test_area) != -1]

    def __getitem__(self, index):
        room_path = os.path.join(self.root, self.file_list[index])
        data_room, label_room = indoor3d_util.room2blocks_wrapper_normalized(room_path, self.block_points)
        return torch.from_numpy(data_room), torch.from_numpy(label_room)

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    d = S3DISDataset(root="../indoor3d_sem_seg_hdf5_data")
    print(len(d))
    data = d[0]
    print(data[0].size(), data[1].size())
