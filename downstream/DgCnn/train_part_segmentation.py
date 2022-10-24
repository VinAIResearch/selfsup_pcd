import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from data_utils import ShapeNetPartSegDataset
from models import DGCNN_part_seg, get_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import copy_parameters, to_one_hot


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser(description="Point Cloud Part Segmentation")
    parser.add_argument("--log_dir", type=str, default="results_part", metavar="N", help="Name of the experiment")
    parser.add_argument("--dataset_path", type=str, default="shapenetpart", metavar="N")
    parser.add_argument(
        "--class_choice",
        type=str,
        default=None,
        metavar="N",
        choices=[
            "airplane",
            "bag",
            "cap",
            "car",
            "chair",
            "earphone",
            "guitar",
            "knife",
            "lamp",
            "laptop",
            "motor",
            "mug",
            "pistol",
            "rocket",
            "skateboard",
            "table",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=32, metavar="batch_size", help="Size of batch)")
    parser.add_argument("--nepoch", type=int, default=200, metavar="N", help="number of episode to train ")
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
    parser.add_argument("--manualSeed", type=int, default=None, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--num_point", type=int, default=2048, help="num of points to use")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--emb_dims", type=int, default=1024, metavar="N", help="Dimension of embeddings")
    parser.add_argument("--k", type=int, default=20, metavar="N", help="Num of nearest neighbors to use")
    parser.add_argument("--model_path", type=str, default="", metavar="N", help="Pretrained model path")
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

    dataset = ShapeNetPartSegDataset(root=args.dataset_path, num_points=args.num_point, split="trainval")
    seg_classes = dataset.seg_classes
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    test_dataset = ShapeNetPartSegDataset(root=args.dataset_path, num_points=args.num_point, split="test")
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    print(len(dataset), len(test_dataset))
    num_classes = 16
    num_part_classes = 50
    print("classes", num_classes)

    name_folder = "%s_%s_%s_seed_%s" % (args.nepoch, args.batch_size, args.num_point, args.manualSeed)
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

    classifier = DGCNN_part_seg(args, num_part_classes)
    if args.model_path != "":
        classifier = copy_parameters(classifier, torch.load(args.model_path), part_seg=True)
    classifier.cuda()

    if args.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        optimizer = optim.Adam(optimizer.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == "cos":
        scheduler = CosineAnnealingLR(optimizer, args.nepoch, eta_min=1e-3)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, args.nepoch + 1):
        total_loss = 0.0
        total_seen = 0.0
        total_correct = 0.0
        classifier.train()

        for data in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            points, label, target = data
            batch_size = points.size(0)
            points, label, target = (
                points.cuda().transpose(2, 1),
                label.long().cuda(),
                target.view(-1, 1)[:, 0].long().cuda(),
            )

            optimizer.zero_grad()
            seg_pred = classifier(points, to_one_hot(label.squeeze(), num_classes))
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            loss = get_loss(seg_pred.view(-1, num_part_classes), target.view(-1, 1).squeeze())

            seg_pred = seg_pred.contiguous().view(-1, num_part_classes)
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            total_correct += correct.item()
            total_seen += batch_size * args.num_point
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if args.scheduler == "cos":
            scheduler.step()
        elif args.scheduler == "step":
            if optimizer.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]["lr"] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-5
        train_acc = total_correct / total_seen
        print("Epoch %d: loss: %f acc(each point): %f" % (epoch, total_loss, train_acc))
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)

    torch.save(classifier.state_dict(), "%s/partseg_model.pth" % (path_checkpoints))

    # test
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part_classes)]
        total_correct_class = [0 for _ in range(num_part_classes)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testdataloader)):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda().transpose(2, 1), label.long().cuda(), target.long().cuda()

            seg_pred = classifier(points, to_one_hot(label.squeeze(), num_classes))
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()

            loss = get_loss(seg_pred.view(-1, num_part_classes), target.view(-1, 1).squeeze())
            cur_pred_val = seg_pred.cpu().data.numpy()

            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += cur_batch_size * NUM_POINT

            for l in range(num_part_classes):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target == l))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0
                    ):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l))
                        )
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics["accuracy"] = total_correct / float(total_seen)
        test_metrics["class_avg_accuracy"] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
        )
        for cat in sorted(shape_ious.keys()):
            print("eval mIoU of %s %f" % (cat + " " * (14 - len(cat)), shape_ious[cat]))
        test_metrics["each mIoU"] = shape_ious
        test_metrics["class_avg_iou"] = mean_shape_ious
        test_metrics["inctance_avg_iou"] = np.mean(all_shape_ious)
        with open("%s/test_results.json" % path_logs, "w") as json_file:
            json.dump(test_metrics, json_file)


if __name__ == "__main__":
    train()
