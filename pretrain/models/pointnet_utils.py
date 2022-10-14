
import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
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
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x

class STN3D3k(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D3k, self).__init__()
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
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3 * 3)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(3).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, 3, 3)
        return x

class PointNet_global(nn.Module):
    def __init__(self, input_channels, feature_dim=128, feature_transform=False):
        super(PointNet_global, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.stn2 = STN3D(64)
        self.feature_dim = feature_dim
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
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
            nn.ReLU()
        )
        self.global_project = nn.Sequential(
            nn.Linear(1024, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_dim, bias=True)
        )


    def forward(self, x):
        batch_size = x.shape[0]
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

        # return self.global_project(x), T2
        # return x, T2
        return self.global_project(x), T2
        # return x

class PointNet_point(nn.Module):
    def __init__(self, input_channels, feature_dim=128, feature_transform=False):
        super(PointNet_point, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.stn2 = STN3D(64)
        self.feature_dim = feature_dim
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
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
            nn.ReLU()
        )
        self.point_project = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, self.feature_dim, 1)
        )


    def forward(self, x):
        batch_size = x.shape[0]
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
            x =self.mlp2(x)
            T2 = None
        # return self.point_project(x), T1, T2
        return self.point_project(x), T2

class PointNet_point_global(nn.Module):
    def __init__(self, input_channels, feature_dim=128, feature_transform=False):
        super(PointNet_point_global, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.stn2 = STN3D(64)
        self.feature_dim = feature_dim
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
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
            nn.ReLU()
        )
        self.point_project = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, self.feature_dim, 1)
        )
        self.global_project = nn.Sequential(
            nn.Linear(1024, 512, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.feature_dim, bias=True)
        )


    def forward(self, x):
        batch_size = x.shape[0]
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
            x =self.mlp2(x)
            T2 = None
        # return self.point_project(x), T1, T2
        point_feat = self.point_project(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        global_feat = self.global_project(x)

        return global_feat, point_feat, T2
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.sum(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2))**2)/2
    return loss
if __name__=='__main__':
    net = PointNet_point(3, 128)
    x = torch.rand(2,1024, 3)
    p = net(x)
    print(p[0][:, [2,2]])
