import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../models"))
from dgcnn_sem_segmentation import DGCNN


def copy_parameters(model, pretrained, verbose=True, part_seg=False):
    feat_dict = model.state_dict()
    # load pre_trained self-supervised
    pretrained_dict = pretrained
    try:
        pretrained_dict = pretrained_dict["model_state_dict"]
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}  # remove name module.
    except:
        print("Not OcCo pretrained")
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() if k in feat_dict and pretrained_dict[k].size() == feat_dict[k].size()
    }

    if verbose:
        print("=" * 27)
        print("Restored Params and Shapes:")
        for k, v in pretrained_dict.items():
            print(k, ": ", v.size())
        print("=" * 68)
    feat_dict.update(pretrained_dict)
    model.load_state_dict(feat_dict)
    return model


def to_one_hot(y, num_class):
    new_y = torch.eye(num_class)[
        y.cpu().data.numpy(),
    ]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Part Segmentation")
    parser.add_argument("--exp_name", type=str, default="exp", metavar="N", help="Name of the experiment")
    parser.add_argument(
        "--model", type=str, default="dgcnn", metavar="N", choices=["dgcnn"], help="Model to use, [dgcnn]"
    )
    parser.add_argument("--dataset", type=str, default="S3DIS", metavar="N", choices=["S3DIS"])
    parser.add_argument(
        "--test_area", type=str, default=None, metavar="N", choices=["1", "2", "3", "4", "5", "6", "all"]
    )
    parser.add_argument("--batch_size", type=int, default=32, metavar="batch_size", help="Size of batch)")
    parser.add_argument("--test_batch_size", type=int, default=16, metavar="batch_size", help="Size of batch)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N", help="number of episode to train ")
    parser.add_argument("--use_sgd", type=bool, default=True, help="Use SGD")
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001, 0.1 if using sgd)"
    )
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cos",
        metavar="N",
        choices=["cos", "step"],
        help="Scheduler to use, [cos, step]",
    )
    parser.add_argument("--no_cuda", type=bool, default=False, help="enables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--eval", type=bool, default=False, help="evaluate the model")
    parser.add_argument("--num_points", type=int, default=4096, help="num of points to use")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Dimension of embeddings")
    parser.add_argument("--k", type=int, default=20, metavar="N", help="Num of nearest neighbors to use")
    parser.add_argument("--model_root", type=str, default="", metavar="N", help="Pretrained model root")
    args = parser.parse_args()
    net = DGCNN(args)
    pre_trained = torch.load("/vinai/bachtx12/pre_trained_occo/dgcnn_occo_cls.pth")
    copy_parameters(net, pre_trained)
