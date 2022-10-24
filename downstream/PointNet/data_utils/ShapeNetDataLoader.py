import glob
import json
import os
import os.path
import h5py
import numpy as np
import torch
import torch.utils.data as data


class ShapeNetPartSegDataset(data.Dataset):
    def __init__(self, root="shapenet_part_hdf5_data", num_points=2048, split="trainval"):
        super(ShapeNetPartSegDataset, self).__init__()
        self.root = root
        self.num_points = num_points
        self.split = split
        all_obj_cats_file = os.path.join(self.root, "all_object_categories.txt")
        fin = open(all_obj_cats_file, "r")
        lines = [line.rstrip() for line in fin.readlines()]
        self.all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
        fin.close()

        self.cat2id = {}
        id = 0
        for (cat, off) in self.all_obj_cats:
            self.cat2id[cat] = id
            id += 1
        self.seg_classes = json.load(open(os.path.join(self.root, "cats2idpart.json"), "r"))
        self.paths = []
        if self.split == "trainval":
            for cat in self.cat2id:
                self.paths += glob.glob("%s/train/%s/*points.txt" % (self.root, cat))
            for cat in self.cat2id:
                self.paths += glob.glob("%s/val/%s/*points.txt" % (self.root, cat))
        elif self.split == "train":
            for cat in self.cat2id:
                self.paths += glob.glob("%s/train/%s/*points.txt" % (self.root, cat))
        elif self.split == "test":
            for cat in self.cat2id:
                self.paths += glob.glob("%s/test/%s/*points.txt" % (self.root, cat))
        else:
            print("Error: Mistake split")
            exit()

    def __getitem__(self, index):
        fnp = self.paths[index]
        cat = fnp.split("/")[-2]
        cls = self.cat2id[cat]
        fns = fnp.replace("points", "seg")
        points = np.loadtxt(fnp)
        points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
        points = points / dist  # scale
        seg = np.loadtxt(fns)
        points = torch.from_numpy(points.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        seg = torch.from_numpy(seg.astype(np.int64))
        return points, cls, seg

    def __len__(self):
        return len(self.paths)


def convert_data(folder_path="shapenet_part_hdf5_data"):
    all_obj_cats_file = os.path.join(folder_path, "all_object_categories.txt")
    fin = open(all_obj_cats_file, "r")
    lines = [line.rstrip() for line in fin.readlines()]
    all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
    fin.close()
    cat2id = {}
    id2cat = {}
    off2cat = {}
    id = 0
    for (cat, off) in all_obj_cats:
        off2cat[off] = cat
        cat2id[cat] = id
        id2cat[id] = cat
        id += 1
    print(cat2id)
    print(id2cat)
    cat2idpart = {}
    all_cats = json.load(open(os.path.join(folder_path, "overallid_to_catid_partid.json"), "r"))
    id = 0
    for cp in all_cats:
        off = cp[0]
        label = off2cat[off]
        if label not in cat2idpart:
            cat2idpart[label] = []
            cat2idpart[label].append(id)
        else:
            cat2idpart[label].append(id)
        id += 1
    with open("%s/cats2idpart.json" % folder_path, "w") as json_file:
        json.dump(cat2idpart, json_file)
    try:
        os.makedirs("%s/train" % (folder_path))
        os.makedirs("%s/val" % (folder_path))
        os.makedirs("%s/test" % (folder_path))
        for cat in cat2id:
            os.makedirs("%s/train/%s" % (folder_path, cat))
            os.makedirs("%s/val/%s" % (folder_path, cat))
            os.makedirs("%s/test/%s" % (folder_path, cat))
    except OSError:
        pass
    cat_index = {}
    for id in id2cat:
        cat_index[id] = 0
    for train_file in [line.strip() for line in open("%s/train_hdf5_file_list.txt" % folder_path)]:
        print(train_file)
        f = h5py.File("%s/%s" % (folder_path, train_file))
        data = f["data"]
        labels = f["label"]
        segs = f["pid"]
        batch_size = labels.shape[0]
        for i in range(batch_size):
            label = labels[i][0]
            # print(label)
            seg = segs[i]
            points = data[i]
            # print(id2cat[label])
            path_seg = "%s/train/%s/%s_seg.txt" % (folder_path, id2cat[label], cat_index[label])
            path_points = "%s/train/%s/%s_points.txt" % (folder_path, id2cat[label], cat_index[label])
            cat_index[label] += 1
            np.savetxt(path_seg, seg)
            np.savetxt(path_points, points)
    print(cat_index)
    cat_index = {}
    for id in id2cat:
        cat_index[id] = 0
    for train_file in [line.strip() for line in open("%s/val_hdf5_file_list.txt" % folder_path)]:
        print(train_file)
        f = h5py.File("%s/%s" % (folder_path, train_file))
        data = f["data"]
        labels = f["label"]
        segs = f["pid"]
        batch_size = labels.shape[0]
        for i in range(batch_size):
            label = labels[i][0]
            seg = segs[i]
            points = data[i]
            path_seg = "%s/val/%s/%s_seg.txt" % (folder_path, id2cat[label], cat_index[label])
            path_points = "%s/val/%s/%s_points.txt" % (folder_path, id2cat[label], cat_index[label])
            cat_index[label] += 1
            np.savetxt(path_seg, seg)
            np.savetxt(path_points, points)
    print(cat_index)
    cat_index = {}
    for id in id2cat:
        cat_index[id] = 0
    for train_file in [line.strip() for line in open("%s/test_hdf5_file_list.txt" % folder_path)]:
        print(train_file)
        f = h5py.File("%s/%s" % (folder_path, train_file))
        data = f["data"]
        labels = f["label"]
        segs = f["pid"]
        batch_size = labels.shape[0]
        for i in range(batch_size):
            label = labels[i][0]
            seg = segs[i]
            points = data[i]
            path_seg = "%s/test/%s/%s_seg.txt" % (folder_path, id2cat[label], cat_index[label])
            path_points = "%s/test/%s/%s_points.txt" % (folder_path, id2cat[label], cat_index[label])
            cat_index[label] += 1
            np.savetxt(path_seg, seg)
            np.savetxt(path_points, points)
    print(cat_index)
