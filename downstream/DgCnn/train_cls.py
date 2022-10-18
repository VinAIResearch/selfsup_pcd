import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
sys.path.append(os.path.join(BASE_DIR, "data_utils"))

import json

import sklearn.metrics as metrics
import torch.nn.functional as F
from dgcnn_classification import DGCNN, get_loss
from ModelNetDataLoader import ModelNetDataset, ModelNetDataset_H5PY
from ScanObjectNNDataLoader import ScanObjectNNDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from utils import copy_parameters


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser(description="Point Cloud Recognition")
    parser.add_argument("--log_dir", type=str, default="results_cls", metavar="N", help="Name of the experiment")
    parser.add_argument("--dataset_type", type=str, required=True, metavar="N")
    parser.add_argument("--dataset_path", type=str, required=True, metavar="N")
    parser.add_argument("--batch_size", type=int, default=32, metavar="batch_size", help="Size of batch)")
    parser.add_argument("--nepoch", type=int, default=250, metavar="N", help="number of episode to train ")
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
    parser.add_argument("--manualSeed", type=int, default=None, metavar="S", help="random seed (default: None)")
    parser.add_argument("--ratio", type=int, default=1, metavar="S", help="random seed (default: None)")
    parser.add_argument("--num_point", type=int, default=1024, help="num of points to use")
    parser.add_argument("--dropout", type=float, default=0.5, help="initial dropout rate")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Dimension of embeddings")
    parser.add_argument("--k", type=int, default=20, metavar="N", help="Num of nearest neighbors to use")
    parser.add_argument("--model_path", type=str, default="", metavar="N", help="Pretrained model path")
    parser.add_argument(
        "--data_aug", type=bool, default=True, metavar="N", help="Using data augmentation for training phase"
    )
    return parser.parse_args()


def train():
    args = parse_args()
    print(args.manualSeed)
    if args.manualSeed != None:
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)
    else:
        args.manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)

    if (
        args.dataset_type == "modelnet40"
        or args.dataset_type == "modelnet40_60"
        or args.dataset_type == "modelnet40_70"
        or args.dataset_type == "modelnet40_80"
        or args.dataset_type == "modelnet40_90"
    ):
        dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )
        test_dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="test", data_augmentation=False
        )
    elif (
        args.dataset_type == "modelnet40_10"
        or args.dataset_type == "modelnet40_20"
        or args.dataset_type == "modelnet40_50"
        or args.dataset_type == "modelnet40_5"
    ):
        dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )
        test_dataset = ModelNetDataset(
            root=args.dataset_path, npoints=args.num_point, split="test", data_augmentation=False
        )
    elif args.dataset_type == "modelnet40h5py":
        dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/train.txt", num_point=args.num_point, data_augmentation=args.data_aug
        )

        test_dataset = ModelNetDataset_H5PY(
            filelist=args.dataset_path + "/test.txt", num_point=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnn":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    elif args.dataset_type == "scanobjectnnbg":
        dataset = ScanObjectNNDataset(
            root=args.dataset_path, npoints=args.num_point, split="train", data_augmentation=args.data_aug
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    elif (
        args.dataset_type == "scanobjectnn10"
        or args.dataset_type == "scanobjectnn20"
        or args.dataset_type == "scanobjectnn30"
        or args.dataset_type == "scanobjectnn40"
        or args.dataset_type == "scanobjectnn50"
        or args.dataset_type == "scanobjectnn60"
        or args.dataset_type == "scanobjectnn70"
        or args.dataset_type == "scanobjectnn80"
        or args.dataset_type == "scanobjectnn90"
        or args.dataset_type == "scanobjectnn5"
    ):
        dataset = ScanObjectNNDataset(
            root=args.dataset_path,
            npoints=args.num_point,
            small_data=True,
            ratio=args.ratio,
            split="train",
            data_augmentation=args.data_aug,
        )

        test_dataset = ScanObjectNNDataset(
            root=args.dataset_path, split="test", npoints=args.num_point, data_augmentation=False
        )
    else:
        exit("wrong dataset type")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print(len(dataset), len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes = dataset.num_classes
    print("classes", num_classes)

    name_folder = "%s_%s_%s_%s_seed_%s" % (
        args.dataset_type,
        args.nepoch,
        args.batch_size,
        args.num_point,
        args.manualSeed,
    )
    path_checkpoints = os.path.join(args.log_dir, name_folder, "checkpoints")
    path_logs = os.path.join(args.log_dir, name_folder, "logs")
    path_runs = os.path.join(args.log_dir, name_folder, "runs")

    try:
        os.makedirs(path_checkpoints)
        os.makedirs(path_runs)
        os.makedirs(path_logs)
    except OSError:
        pass

    with open("%s/args.txt" % path_logs, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(path_runs)

    classifier = DGCNN(args, output_channels=num_classes)

    if args.model_path != "":
        classifier = copy_parameters(classifier, torch.load(args.model_path))
    classifier.cuda()
    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(optimizer, args.nepoch, eta_min=1e-3)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

    best_acc = 0.0
    best_epoch = 0
    test_acc = 0
    for epoch in range(1, args.nepoch + 1):
        total_loss = 0.0
        total_point = 0.0

        classifier.train()
        train_pred = []
        train_true = []
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            batch_size = points.size(0)
            total_point += batch_size
            target = target[:, 0]
            points, target = points.cuda().transpose(2, 1), target.cuda()

            optimizer.zero_grad()

            logits = classifier(points)
            loss = get_loss(logits, target)

            total_loss += loss.item() * batch_size
            loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            train_true.append(target.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if args.scheduler == "cos":
            scheduler.step()
        elif args.scheduler == "step":
            if opt.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]["lr"] < 1e-5:
                for param_group in opt.param_groups:
                    param_group["lr"] = 1e-5
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_macc = metrics.balanced_accuracy_score(train_true, train_pred)
        ## Test
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_point = 0.0
            test_true = []
            test_pred = []
            classifier.eval()
            for i, data in tqdm(enumerate(testdataloader, 0)):
                points, target = data
                batch_size = points.size(0)
                test_total_point += batch_size
                target = target[:, 0]
                points, target = points.cuda().transpose(2, 1), target.cuda()

                logits = classifier(points)
                loss = get_loss(logits, target)
                test_total_loss += loss.item() * batch_size

                preds = logits.max(dim=1)[1]
                test_true.append(target.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            test_macc = metrics.balanced_accuracy_score(test_true, test_pred)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(classifier.state_dict(), "%s/cls_best_model.pth" % (path_checkpoints))
        print(
            "[%d] train loss: %f accuracy: %f test_acc: %f best acc: %f best epoch: %d"
            % (epoch, total_loss / total_point, train_acc, test_acc, best_acc, best_epoch)
        )
        writer.add_scalar("Loss/train", total_loss / total_point, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/test", test_acc, epoch)
    torch.save(classifier.state_dict(), "%s/cls_model.pth" % (path_checkpoints))
    ## Test
    with torch.no_grad():
        results = {}
        total_loss = 0.0
        total_point = 0.0
        test_total_loss = 0.0
        classifier.eval()
        test_pred = []
        test_true = []
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            batch_size = points.size(0)
            total_point += batch_size
            target = target[:, 0]
            points, target = points.cuda().transpose(2, 1), target.cuda()

            logits = classifier(points)
            loss = get_loss(logits, target)

            test_total_loss += loss.item() * batch_size

            preds = logits.max(dim=1)[1]
            test_true.append(target.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_macc = metrics.balanced_accuracy_score(test_true, test_pred)
        results["Loss"] = test_total_loss / total_point
        results["Instance acc"] = test_acc
        results["Mean acc"] = test_macc
        results["Best acc"] = best_acc
        with open("%s/test_results.txt" % path_logs, "w") as f:
            json.dump(results, f)
        print(results)


if __name__ == "__main__":
    train()
