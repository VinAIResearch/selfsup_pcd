import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3D_input(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D_input, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3 * 3),
        )

    def forward(self, x):
        num_points = x.shape[2]
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(3).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, 3, 3)
        return x


class STN3D_feature(nn.Module):
    def __init__(self, input_channels=64):
        super(STN3D_feature, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        num_points = x.shape[2]
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.sum(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)) ** 2) / 2
    return loss
