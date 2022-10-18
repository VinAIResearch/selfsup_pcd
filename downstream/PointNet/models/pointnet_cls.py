import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import STN3D_feature, STN3D_input, feature_transform_regularizer


class PointNet(nn.Module):
    def __init__(self, input_channels, num_classes, global_feature=False, feature_transform=True):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.feature_transform = feature_transform
        self.stn1 = STN3D_input(input_channels)
        if self.feature_transform:
            self.stn2 = STN3D_feature(64)
        self.num_classes = num_classes
        self.global_feature = global_feature
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            # nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        x = torch.bmm(T1, x)
        x = self.mlp1(x)
        if self.feature_transform:
            T2 = self.stn2(x)
            f = torch.bmm(T2, x)
            x = self.mlp2(f)
        else:
            x = self.mlp2(x)
            T2 = None
        x = F.max_pool1d(x, num_points).squeeze(2)
        if self.global_feature:
            return x
        x = self.classifier(x)
        return F.log_softmax(x, dim=1), T1, T2


class PointNet_critical(nn.Module):
    def __init__(self, input_channels, feature_transform=True):
        super(PointNet_critical, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D_input(input_channels)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.stn2 = STN3D_feature(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        x = torch.bmm(T1, x)
        x = self.mlp1(x)
        if self.feature_transform:
            T2 = self.stn2(x)
            f = torch.bmm(T2, x)
            x = self.mlp2(f)
        else:
            T2 = None
            x = self.mlp2(x)
        critical_index = torch.argmax(x, dim=2)
        global_feature = F.max_pool1d(x, num_points).squeeze(2)
        return global_feature, critical_index, x


def get_loss(pred, target, feat_trans, reg_weight=0.001):
    # cross entropy loss
    loss_cls = F.nll_loss(pred, target)

    # regularize loss
    loss_reg = feature_transform_regularizer(feat_trans)

    return loss_cls + loss_reg * reg_weight

    return 0
