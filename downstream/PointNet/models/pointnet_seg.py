import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import STN3D_feature, STN3D_input, feature_transform_regularizer


class PointNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(PointNet, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D_input(input_channels)
        self.stn2 = STN3D_feature(64)
        self.num_classes = num_classes
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
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.num_classes, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        num_channels = x.shape[2]

        x = x.transpose(2, 1)
        T1 = self.stn1(x)
        if num_channels > 3:
            x = x.transpose(2, 1)
            features = x[:, :, 3:]
            x = x[:, :, :3]
            x = x.transpose(2, 1)
        x = torch.bmm(T1, x)
        if num_channels > 3:
            x = x.transpose(2, 1)
            x = torch.cat([x, features], dim=2)
            x = x.transpose(2, 1)
        x = self.mlp1(x)
        T2 = self.stn2(x)
        f = torch.bmm(T2, x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze()
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        concat_feat = torch.cat([x, f], 1)

        x = self.classifier(concat_feat)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        x = x.view(batch_size, num_points, self.num_classes)
        return x, T1, T2


def get_loss(pred, target, feat_trans, reg_weight=0.001):
    # cross entropy loss
    loss_cls = F.nll_loss(pred, target)

    # regularize loss
    loss_reg = feature_transform_regularizer(feat_trans)

    return loss_cls + loss_reg * reg_weight

    return 0


if __name__ == "__main__":
    pointnet = PointNet(9, 40)
    data = torch.rand(4, 1024, 9)
    print(pointnet(data))
