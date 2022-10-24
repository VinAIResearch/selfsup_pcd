import glob
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from .data_utils import (
    center_point_cloud,
    farthest_pair_sample,
    jitter_point_cloud,
    normalize_point_cloud,
    rotate_point_cloud,
    separate_point_sample,
    translate_pointcloud,
)
from PIL import Image
from torchvision import transforms


test_transform = transforms.Compose([transforms.ToTensor()])
list_error = [
    "airplane_0299.",
    "plant_0116.",
    "curtain_0071.",
    "tv_stand_0055.",
    "tv_stand_0012.",
    "bottle_0064.",
    "bottle_0296.",
    "bottle_0106.",
    "bottle_0154.",
    "bottle_0149.",
    "bottle_0036.",
    "radio_0019.",
    "guitar_0045.",
    "guitar_0109.",
    "flower_pot_0110.",
    "cup_0058.",
    "vase_0354.",
    "keyboard_0088.",
]


class ModelNetSSLDataset(data.Dataset):
    def __init__(
        self,
        root,
        npoints=1024,
        num_views=12,
        split="train",
        num_point_contrast=512,
        threshold=128,  # a view at least 128 pair
        fps="random",
        pre_fix="ModelNet40_MV",  # name folder multi-views
        data_augmentation=False,
    ):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.num_views = num_views
        self.cats = {}
        self.threshold = threshold
        self.fps = fps
        self.pre_fix = pre_fix
        print(self.pre_fix)
        self.num_point_contrast = num_point_contrast
        with open(os.path.join(self.root, "modelnet_id.txt")) as f:
            for line in f:
                line = line.split("\t")
                self.cats[line[0]] = int(line[1])
        self.paths = []
        self.classes = dict(zip(sorted(self.cats), range(len(self.cats))))
        path_temp = []
        for cat in self.cats:
            path_temp += glob.glob("%s/%s/%s/*" % (self.root, cat, self.split))
        self.views = []
        for fn in path_temp:
            if self.num_views != 12:
                list_view = []
                folder_mv = fn.replace("ModelNet40_blender_sampling_1024", self.pre_fix).replace(".txt", "")
                for i in range(12):
                    fpp = "%s/pix_point_%s.txt" % (folder_mv, i)
                    with open(fpp, "r") as fp:
                        num_pair = len(fp.readlines())
                        if num_pair > self.threshold:
                            list_view.append(i)
                if len(list_view) >= self.num_views:
                    self.views.append(list_view)
                    self.paths.append(fn)
            else:
                flag = True
                for er in list_error:
                    if er in fn:
                        flag = False
                        continue
                if flag:
                    self.views.append([i for i in range(12)])
                    self.paths.append(fn)

    def __getitem__(self, index):
        fn = self.paths[index]
        point_set = np.loadtxt(fn)[:, [0, 2, 1]]
        folder_mv = fn.replace("ModelNet40_blender_sampling_1024", self.pre_fix).replace(".txt", ".off")
        list_image = []
        views = self.views[index]
        np.random.shuffle(views)
        for i in range(self.num_views):
            fimg = "%s/view_%s.png" % (folder_mv, views[i])
            list_image.append(test_transform(Image.open(fimg)).unsqueeze(0))

        list_id2pix = []
        for i in range(self.num_views):
            fpp = "%s/pix_point_%s.txt" % (folder_mv, views[i])
            indi = np.loadtxt(fpp, dtype=np.int32)
            indi = np.hstack((np.ones((indi.shape[0], 1), dtype=np.int32) * i, indi))
            list_id2pix.append(indi)
        list_id2pix = np.concatenate(list_id2pix, 0)

        # create dict map point to list pixel
        dict_pair = {}
        for i in range(len(list_id2pix)):
            point_id = list_id2pix[i][3]
            if point_id not in dict_pair:
                dict_pair[point_id] = []
            dict_pair[point_id].append(i)
        # shuffle pixel
        for k in dict_pair.keys():
            np.random.shuffle(dict_pair[k])

        if self.fps == "fps":
            # furthest sampling pair
            list_id = farthest_pair_sample(point_set, dict_pair, self.num_point_contrast)
            list_id2pix = list_id2pix[list_id]
        elif self.fps == "sps":
            list_id = separate_point_sample(point_set, dict_pair, self.num_point_contrast)
            list_id2pix = list_id2pix[list_id]
        else:
            # random sampling pair
            if list_id2pix.shape[0] < self.num_point_contrast:
                print("Not enough", list_id2pix.shape[0], fn)
            idx_pts = np.arange(list_id2pix.shape[0])
            np.random.shuffle(idx_pts)
            list_id2pix = list_id2pix[idx_pts[: self.num_point_contrast]]

        point_set = center_point_cloud(point_set)
        point_set = normalize_point_cloud(point_set)
        point_set = point_set[0 : self.npoints, :]
        if self.data_augmentation:
            # point_set = rotate_point_cloud(point_set)
            # point_set = jitter_point_cloud(point_set)
            point_set = translate_pointcloud(point_set)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        return point_set, torch.cat(list_image, dim=0), list_id2pix

    def __len__(self):
        return len(self.paths)
