import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
from pointnet_utils import STN3D_feature, STN3D_input, feature_transform_regularizer


class PointNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D_input(input_channels)
        self.stn2 = STN3D_feature(128)
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn5 = nn.BatchNorm1d(2048)

        self.classifier = nn.Sequential(
            nn.Conv1d(3024, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.num_classes, 1),
        )

    def forward(self, x, labels):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        x = torch.bmm(T1, x)

        out1 = F.relu(self.bn1(self.conv1(x)))

        out2 = F.relu(self.bn2(self.conv2(out1)))

        out3 = F.relu(self.bn3(self.conv3(out2)))

        T2 = self.stn2(out3)

        out4 = torch.bmm(T2, out3)

        out5 = F.relu(self.bn4(self.conv4(out4)))

        global_feat = F.relu(self.bn5(self.conv5(out5)))

        out_max = F.max_pool1d(global_feat, num_points).squeeze(2)
        # print(out_max.size(), labels.size())
        out_concat = torch.cat([out_max, labels], 1)
        out_expand = out_concat.unsqueeze(2).repeat(1, 1, num_points)

        out_final = torch.cat([out1, out2, out3, out4, out5, out_expand], 1)
        output = self.classifier(out_final).transpose(2, 1).contiguous()
        output = F.log_softmax(output, dim=-1)
        return output.view(batch_size, num_points, self.num_classes), T2


def get_loss(pred, target, feat_trans, reg_weight=0.001):
    # cross entropy loss
    loss_cls = F.nll_loss(pred, target)

    # regularize loss
    loss_reg = feature_transform_regularizer(feat_trans)

    return loss_cls + loss_reg * reg_weight

    return 0


if __name__ == "__main__":
    # print(nn.Conv1d(3,64,1).weight.size(),nn.Conv1d(3,64,1).bias.size() )
    pointnet = PointNet(3, 50)
    # pointnet.train()
    labels = torch.ones(4, 16)
    data = torch.ones(4, 1024, 3)
    # print(pointnet(data)[0])
    with torch.no_grad():
        pointnet.eval()
        # print(pointnet)
        print(pointnet(data, labels)[0].size())
