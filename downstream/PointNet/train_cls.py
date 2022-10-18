import argparse
import json
import random
import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from ModelNetDataLoader import ModelNetDataset, ModelNetDataset_H5PY
from pointnet_cls import PointNet, get_loss
from ScanObjectNNDataLoader import ScanObjectNNDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import bn_momentum_adjust, copy_parameters, init_weights, init_zeros


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Classification")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size in training [default: 32]")
    parser.add_argument("--nepoch", default=250, type=int, help="number of epoch in training [default: 250]")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training [default: 0.001]"
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number [default: 1024]")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer for training [default: Adam]")
    parser.add_argument("--log_dir", type=str, default=None, help="experiment root")
    parser.add_argument("--model_path", type=str, default="", help="model pre-trained")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--dataset_type", type=str, required=True, help="scanobjectnn|modelnet40|scanobjectnn10")
    parser.add_argument("--lr_decay", type=float, default=0.7, help="decay rate for learning rate")
    parser.add_argument("--decay_step", type=int, default=20, help="decay step for ")
    parser.add_argument("--momentum_decay", type=float, default=0.5, help="momentum_decay decay of batchnorm")
    parser.add_argument("--manualSeed", type=int, default=None, help="random seed")
    parser.add_argument("--data_aug", action="store_true", help="Using data augmentation for training phase")
    parser.add_argument("--weight_decay", action="store_true", help="Using data augmentation for training phase")
    parser.add_argument("--ratio", type=int, default=1, metavar="S", help="random seed (default: None)")
    return parser.parse_args()


def train():
    args = parse_args()
    print(args.manualSeed)
    if args.manualSeed is not None:
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)
    else:
        args.manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", args.manualSeed)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        np.random.seed(args.manualSeed)

    if args.dataset_type == "modelnet40":
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
    else:
        exit("wrong dataset type")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_classes
    print("classes", num_classes)
    name_folder = f"{args.dataset_type}_{args.nepoch}\
                    _{args.batch_size}_{args.num_point}_seed_{args.manualSeed}"
    path_checkpoints = os.path.join(args.log_dir, name_folder, "checkpoints")
    path_logs = os.path.join(args.log_dir, name_folder, "logs")
    path_runs = os.path.join(args.log_dir, name_folder, "runs")

    try:
        os.makedirs(path_checkpoints)
        os.makedirs(path_runs)
        os.makedirs(path_logs)
    except OSError:
        pass

    with open(f"{path_logs}/args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(path_runs)

    classifier = PointNet(3, num_classes)
    classifier.apply(init_weights)
    classifier.stn1.mlp2[-1].apply(init_zeros)
    classifier.stn2.mlp2[-1].apply(init_zeros)
    if args.model_path != "":
        classifier = copy_parameters(classifier, torch.load(args.model_path))
    classifier.cuda()
    if args.weight_decay:
        print("Using weight decay")
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        print("None using weight decay")
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    MOMENTUM_ORIGINAL = 0.5
    MOMENTUM_DECAY = args.momentum_decay
    MOMENTUM_DECAY_STEP = args.decay_step
    LR_ORIGINAL = args.learning_rate
    LR_DECAY = args.lr_decay
    LR_DECAY_STEP = args.decay_step

    best_acc = 0.0
    best_epoch = 0
    test_acc = 0
    for epoch in range(1, args.nepoch + 1):
        total_loss = 0.0
        total_point = 0.0
        total_correct = 0.0
        lr = LR_ORIGINAL * (LR_DECAY ** (epoch // LR_DECAY_STEP))
        if lr < 1e-5:
            lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        step = epoch // MOMENTUM_DECAY_STEP
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY**step)
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f learning rate: %f" % (momentum, lr))
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        writer.add_scalar("Learning rate", lr, epoch)
        classifier.train()
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            total_point += points.size(0)
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()

            optimizer.zero_grad()

            pred, trans, trans_feat = classifier(points)
            loss = get_loss(pred, target, trans_feat)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
        with torch.no_grad():  # Test
            test_total_loss = 0.0
            test_total_point = 0.0
            test_total_correct = 0.0
            classifier.eval()
            for i, data in tqdm(enumerate(testdataloader, 0)):
                points, target = data
                test_total_point += points.size(0)
                target = target[:, 0]
                points, target = points.cuda(), target.cuda()

                pred, trans, trans_feat = classifier(points)
                loss = get_loss(pred, target, trans_feat)
                test_total_loss += loss.item()

                test_pred_choice = pred.data.max(1)[1]
                test_correct = test_pred_choice.eq(target.data).cpu().sum()
                test_total_correct += test_correct.item()
            test_acc = test_total_correct / test_total_point
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(classifier.state_dict(), f"{path_checkpoints}/cls_best_model.pth")
        print(
            "[%d] train loss: %f accuracy: %f test_acc: %f best acc: %f best epoch: %d"
            % (epoch, total_loss / total_point, total_correct / total_point, test_acc, best_acc, best_epoch)
        )
        writer.add_scalar("Loss/train", total_loss, epoch + 1)
        writer.add_scalar("Acc/train", total_correct / total_point, epoch + 1)
        writer.add_scalar("Acc/test", test_acc, epoch + 1)
    torch.save(classifier.state_dict(), "%s/cls_model.pth" % (path_checkpoints))
    # Test
    with torch.no_grad():
        results = {}
        total_loss = 0.0
        total_point = 0.0
        total_correct = 0.0
        classifier.eval()
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            total_point += points.size(0)
            target = target[:, 0]
            points, target = points.cuda(), target.cuda()

            pred, trans, trans_feat = classifier(points)
            loss = get_loss(pred, target, trans_feat)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
        results["Loss"] = total_loss
        results["Instance acc"] = total_correct / total_point
        results["Best acc"] = best_acc
        with open("%s/test_results.txt" % path_logs, "w") as f:
            json.dump(results, f)
        print(results)


if __name__ == "__main__":
    train()
