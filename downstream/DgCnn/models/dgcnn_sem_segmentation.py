import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgcnn_untils import get_graph_feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False), self.bn6, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=False), self.bn7, nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False), self.bn8, nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(
            x, k=self.k, dim9=True
        )  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x


def get_loss(pred, gold, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction="mean")

    return loss


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
    net.cuda()
    x = torch.rand(3, 9, 4096).to("cuda")
    print(net(x).permute(0, 2, 1).contiguous().view(-1, 13).size())
