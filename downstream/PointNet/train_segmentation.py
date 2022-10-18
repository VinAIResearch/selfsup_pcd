import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
from pointnet_seg import PointNet, get_loss
from S3DISDataLoader import S3DISDataset
from tqdm import tqdm
from utils import bn_momentum_adjust, copy_parameters, init_weights, init_zeros


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Segmentation")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size in training [default: 24]")
    parser.add_argument("--nepoch", default=100, type=int, help="number of epoch in training [default: 100]")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, help="learning rate in training [default: 0.001]"
    )
    parser.add_argument("--num_point", type=int, default=4096, help="Point Number [default: 4096]")
    parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer for training [default: Adam]")
    parser.add_argument("--log_dir", type=str, default="results_sem_seg", help="experiment root")

    parser.add_argument("--model_path", type=str, default="", help="model pre-trained")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--test_area", type=int, default=6, help="Area using for test")

    # parameter of pointnet
    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")
    parser.add_argument("--momentum_decay", type=float, default=0.5, help="momentum_decay decay of batchnorm")
    parser.add_argument("--decay_step", type=int, default=20, help="Decay step for lr decay ")
    parser.add_argument("--manualSeed", type=int, default=None, help="random seed")
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

    classifier = PointNet(9, num_classes)
    classifier.apply(init_weights)
    classifier.stn1.mlp2[-1].apply(init_zeros)
    classifier.stn2.mlp2[-1].apply(init_zeros)
    # load pre-trained model
    if args.model_path != "":
        classifier = copy_parameters(classifier, torch.load(args.model_path))

    MOMENTUM_ORIGINAL = 0.5
    MOMENTUM_DECAY = args.momentum_decay
    MOMENTUM_DECAY_STEP = args.decay_step
    LR_ORIGINAL = args.learning_rate
    LR_DECAY = args.lr_decay
    LR_DECAY_STEP = args.decay_step

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    classifier.cuda()
    classifier.train()
    for epoch in range(1, args.nepoch + 1):
        total_correct = 0.0
        total_seen = 0.0
        total_loss = 0.0

        lr = LR_ORIGINAL * (LR_DECAY ** (epoch // LR_DECAY_STEP))
        if lr < 1e-5:
            lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print("BN momentum updated to: %f learning rate: %f" % (momentum, lr))
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        writer.add_scalar("Learning rate", lr, epoch)

        for i, data in tqdm(enumerate(dataloader, 0)):
            points, target = data
            B = points.size(0)
            points, target = points.cuda(), target.view(-1, 1)[:, 0].long().cuda()

            optimizer.zero_grad()

            seg_pred, trans, trans_feat = classifier(points)
            seg_pred = seg_pred.view(-1, num_classes)
            pred_choice = seg_pred.data.max(1)[1]
            loss = get_loss(seg_pred, target, trans_feat)

            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_seen += B * args.num_point
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # break
        train_acc = total_correct / total_seen
        print("Epoch %d: loss: %f acc(each point): %f" % (epoch, total_loss, train_acc))
        writer.add_scalar("Loss/train", total_loss, epoch + 1)
        writer.add_scalar("Acc/train", train_acc, epoch + 1)

    torch.save(classifier.state_dict(), "%s/seg_model.pth" % (path_checkpoints))
    # benchmark mIOU
    with torch.no_grad():
        total_correct = 0.0
        total_seen = 0.0
        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        total_iou_deno_class = [0 for _ in range(num_classes)]
        classifier = classifier.eval()
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            BATCH_SIZE = points.size()[0]
            points, target = points.cuda(), target.cuda()

            seg_pred, _, _ = classifier(points)
            pred_choice = seg_pred.data.max(2)[1]

            pred_val = pred_choice.cpu().data.numpy()
            batch_label = target.cpu().data.numpy()
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += BATCH_SIZE * args.num_point
            for l in range(num_classes):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            # break
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        test_metrics = {}
        test_metrics["mIoU"] = mIoU
        test_metrics["mAcc"] = total_correct / total_seen
        test_metrics["mcAcc"] = np.mean(
            np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6)
        )
        print("mIOU : {}, avg acc: {}".format(mIoU, total_correct / total_seen))
        with open("%s/test_block_area.json" % path_logs, "w") as json_file:
            json.dump(test_metrics, json_file)
        print(test_metrics)


if __name__ == "__main__":
    # for i in range(6):
    #     train(test_area=i+1)
    train()
