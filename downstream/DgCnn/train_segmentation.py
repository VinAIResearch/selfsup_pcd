import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import json
import sklearn.metrics as metrics
from dgcnn_sem_segmentation import DGCNN, get_loss
from S3DISDataLoader import S3DISDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from utils import calculate_sem_IoU, copy_parameters


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser(description="Point Cloud Part Segmentation")
    parser.add_argument(
        "--log_dir", type=str, default="results_segmentation", metavar="N", help="Name of the experiment"
    )
    parser.add_argument("--dataset_path", type=str, required=True, metavar="N")
    parser.add_argument("--test_area", type=int, default=6, metavar="N")
    parser.add_argument("--batch_size", type=int, default=32, metavar="batch_size", help="Size of batch)")
    parser.add_argument("--nepoch", type=int, default=100, metavar="N", help="number of episode to train ")
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
    parser.add_argument("--num_point", type=int, default=4096, help="num of points to use")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Dimension of embeddings")
    parser.add_argument("--k", type=int, default=20, metavar="N", help="Num of nearest neighbors to use")
    parser.add_argument("--model_path", type=str, default="", metavar="N", help="Pretrained model root")
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

    dataset = S3DISDataset(root=args.dataset_path, split="train", num_points=args.num_point, test_area=args.test_area)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )

    test_dataset = S3DISDataset(
        root=args.dataset_path, split="test", num_points=args.num_point, test_area=args.test_area
    )
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    print(len(dataset), len(test_dataset))
    num_classes = 13
    print("classes", num_classes)
    name_folder = "%s_%s_%s_area%s" % (args.nepoch, args.batch_size, args.num_point, args.test_area)
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

    classifier = DGCNN(args)
    # load pre-trained model
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
        scheduler = StepLR(optimizer, 20, 0.5, args.nepoch)

    for epoch in range(1, args.nepoch + 1):
        total_seen = 0.0
        total_loss = 0.0
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        classifier.train()
        for i, data in tqdm(enumerate(dataloader, 0)):
            points, seg = data
            batch_size = points.size()[0]
            points = points.cuda().transpose(2, 1)
            seg = seg.long().cuda()

            optimizer.zero_grad()

            seg_pred = classifier(points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = get_loss(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())

            loss.backward()
            optimizer.step()
            pred = seg_pred.max(dim=2)[1]  # (batch_size, num_points)
            total_seen += batch_size
            total_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()  # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))  # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            # break
        if args.scheduler == "cos":
            scheduler.step()
        elif args.scheduler == "step":
            if optimizer.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]["lr"] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = np.mean(calculate_sem_IoU(train_pred_seg, train_true_seg))

        print("Epoch %d: loss: %f acc(each point): %f" % (epoch, total_loss / total_seen, train_acc))
        writer.add_scalar("Loss/train", total_loss / total_seen, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("IoU/train", train_ious, epoch)

    torch.save(classifier.state_dict(), "%s/seg_model.pth" % (path_checkpoints))
    # benchmark mIOU
    with torch.no_grad():
        total_loss = 0.0
        total_seen = 0.0
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        classifier = classifier.eval()
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, seg = data
            batch_size = points.size()[0]
            points, seg = points.cuda().transpose(2, 1), seg.long().cuda()

            seg_pred = classifier(points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = get_loss(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            total_seen += batch_size
            total_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = np.mean(calculate_sem_IoU(test_pred_seg, test_true_seg))
        # break
        test_metrics = {}
        test_metrics["mIoU"] = test_ious
        test_metrics["Acc"] = test_acc
        test_metrics["mcAcc"] = avg_per_class_acc
        print("mIOU : {}, avg acc: {}".format(test_ious, test_acc))
        with open("%s/test_block_area.json" % path_logs, "w") as json_file:
            json.dump(test_metrics, json_file)
        print(test_metrics)


if __name__ == "__main__":
    # for i in range(6):
    #     train(test_area=i+1)
    train()
