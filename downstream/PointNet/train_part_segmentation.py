import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_utils'))

from ShapeNetDataLoader import ShapeNetPartSegDataset
from utils import copy_parameters, bn_momentum_adjust, init_weights, init_zeros, to_one_hot
from pointnet_part_seg import PointNet, get_loss
import torch.nn.functional as F
from tqdm import tqdm
import json
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Classification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--nepoch',  default=200, type=int, help='number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='results_part_seg', help='experiment root')
    parser.add_argument('--model_path', type=str, default='', help='model pre-trained')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='decay rate for learning rate')
    parser.add_argument('--decay_step', type=int, default=20, help='decay step for ')
    parser.add_argument('--momentum_decay', type=float, default=0.5, help='momentum_decay decay of batchnorm')
    parser.add_argument('--manualSeed', type=int, default=None, help='random seed')
    parser.add_argument('--weight_decay', action='store_true', help='Using data augmentation for training phase')

    #parameter of pointnet
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

    dataset = ShapeNetPartSegDataset(root = args.dataset_path, num_points= args.num_point, split='trainval')
    seg_classes = dataset.seg_classes
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    test_dataset = ShapeNetPartSegDataset(root = args.dataset_path, num_points= args.num_point, split='test')
    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8)

    print(len(dataset), len(test_dataset))
    # num_classes = len(dataset.classes)
    num_classes = 16
    num_part_classes =50
    print('classes', num_classes)

    name_folder = '%s_%s_%s_seed_%s'%(args.nepoch,args.batch_size,args.num_point,args.manualSeed )
    path_checkpoints = os.path.join(args.log_dir,name_folder,'checkpoints')
    path_logs = os.path.join(args.log_dir,name_folder,'logs')
    path_runs = os.path.join(args.log_dir,name_folder,'runs')

    try:
        os.makedirs(path_checkpoints)
        os.makedirs(path_runs)
        os.makedirs(path_logs)
    except OSError:
        pass

    with open('%s/args.txt'%path_logs, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(path_runs)

    classifier = PointNet(3, num_part_classes)
    classifier.apply(init_weights)
    classifier.stn1.mlp2[-1].apply(init_zeros)
    classifier.stn2.mlp2[-1].apply(init_zeros)
    if args.model_path != '':
        classifier = copy_parameters(classifier,torch.load(args.model_path))
    classifier.cuda()
    if args.weight_decay:
        print('Using weight decay')
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        print('None using weight decay')
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
    for epoch in range(1,args.nepoch+1):
        total_loss = 0.0
        total_seen = 0.0
        total_correct = 0.0
        lr = LR_ORIGINAL * (LR_DECAY ** (epoch // LR_DECAY_STEP))
        if lr < 1e-5:
            lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f learning rate: %f' % (momentum, lr))
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        writer.add_scalar('Learning rate',lr, epoch)
        classifier.train()

        for data in tqdm(dataloader, total=len(dataloader), smoothing=0.9):
            points, label, target = data
            batch_size = points.size(0)
            points, label, target = points.cuda(), label.long().cuda(), \
                            target.view(-1, 1)[:, 0].long().cuda()

            optimizer.zero_grad()
            seg_pred, trans_feat = classifier(points, to_one_hot(label.squeeze(), num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part_classes)
            pred_choice = seg_pred.data.max(1)[1]

            loss = get_loss(seg_pred, target, trans_feat)


            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct+=correct.item()
            total_seen +=batch_size * args.num_point
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
        train_acc = total_correct/total_seen
        print('Epoch %d: loss: %f acc(each point): %f'%(epoch, total_loss, train_acc))
        writer.add_scalar('Loss/train',total_loss, epoch+1)
        writer.add_scalar('Acc/train',train_acc, epoch+1)

    torch.save(classifier.state_dict(), '%s/partseg_model.pth' % (path_checkpoints))

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
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

            seg_pred, _ = classifier(points, to_one_hot(label.squeeze(), num_classes))
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
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part_classes):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['each mIoU'] = shape_ious
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        with open('%s/test_results.json'%path_logs, 'w') as json_file:
            json.dump(test_metrics,json_file)
if __name__=='__main__':
    train()