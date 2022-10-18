# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from pathlib import Path

import numpy as np
from lib.dataset import DatasetPhase, VoxelizationDataset, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import fast_hist, per_class_iu, read_txt
from scipy import spatial


CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


class ScannetVoxelizationDataset(VoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = "z"
    LOCFEAT_IDX = 2
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
    IS_FULL_POINTCLOUD_EVAL = True

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: "scannetv2_train.txt",
        DatasetPhase.Val: "scannetv2_val.txt",
        DatasetPhase.TrainVal: "scannetv2_trainval.txt",
        DatasetPhase.Test: "scannetv2_test.txt",
    }

    def __init__(
        self,
        config,
        prevoxel_transform=None,
        input_transform=None,
        target_transform=None,
        augment_data=True,
        elastic_distortion=False,
        cache=False,
        phase=DatasetPhase.Train,
    ):
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.data.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_paths = read_txt(
            os.path.join(
                "/lustre/scratch/client/vinai/users/bachtx12/PixelPoint3D/downstream/semseg/splits/scannet",
                self.DATA_PATH_FILE[phase],
            )
        )
        # data_paths = read_txt(os.path.join('./splits/scannet_temp', self.DATA_PATH_FILE[phase]))
        logging.info("Loading {}: {}".format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.data.ignore_label,
            return_transformation=config.data.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
        )

    def get_output_id(self, iteration):
        return "_".join(Path(self.data_paths[iteration]).stem.split("_")[:2])

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1), pointcloud[:, 6:])
        )
        return pointcloud

    def test_pointcloud(self, pred_dir):
        print("Running full pointcloud evaluation.")
        eval_path = os.path.join(pred_dir, "fulleval")
        os.makedirs(eval_path, exist_ok=True)
        # Join room by their area and room id.
        # Test independently for each room.
        sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
        hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
        for i, data_path in enumerate(self.data_paths):
            room_id = self.get_output_id(i)
            pred = np.load(os.path.join(pred_dir, "pred_%04d_%02d.npy" % (i, 0)))

            # save voxelized pointcloud predictions
            save_point_cloud(
                np.hstack((pred[:, :3], np.array([SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))),
                f"{eval_path}/{room_id}_voxel.ply",
                verbose=False,
            )

            fullply_f = self.data_root / data_path
            query_pointcloud = read_plyfile(fullply_f)
            query_xyz = query_pointcloud[:, :3]
            query_label = query_pointcloud[:, -1]
            # Run test for each room.
            pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
            _, result = pred_tree.query(query_xyz)
            ptc_pred = pred[result, 3].astype(int)
            # Save prediciton in txt format for submission.
            np.savetxt(f"{eval_path}/{room_id}.txt", ptc_pred, fmt="%i")
            # Save prediciton in colored pointcloud for visualization.
            save_point_cloud(
                np.hstack((query_xyz, np.array([SCANNET_COLOR_MAP[i] for i in ptc_pred]))),
                f"{eval_path}/{room_id}.ply",
                verbose=False,
            )
            # Evaluate IoU.
            if self.IGNORE_LABELS is not None:
                ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
                query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
            hist += fast_hist(ptc_pred, query_label, self.NUM_LABELS)
        ious = per_class_iu(hist) * 100
        print(
            "mIoU: " + str(np.nanmean(ious)) + "\n"
            "Class names: " + ", ".join(CLASS_LABELS) + "\n"
            "IoU: " + ", ".join(np.round(ious, 2).astype(str))
        )


class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02


class ScannetVoxelization4cmDataset(ScannetVoxelizationDataset):
    VOXEL_SIZE = 0.04
