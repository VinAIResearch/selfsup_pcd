import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Model2D_pixel(nn.Module):
    def __init__(self, feature_dim=128, num_views=12):
        super(Model2D_pixel, self).__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views
        modules = []
        for name, module in resnet50().named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if (
                not isinstance(module, nn.Linear)
                and not isinstance(module, nn.MaxPool2d)
                and not isinstance(module, nn.AdaptiveAvgPool2d)
            ):
                modules.append(module)
        self.f = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # upsampling projection layer
        self.convup = nn.Conv2d(2048, feature_dim, kernel_size=(1, 1))
        self.upsample = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=True)
        # projection layer
        self.t = nn.Linear(2048, 1024, bias=False)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, input):
        features = self.f(input)

        global_features = self.pool(features).squeeze()
        global_feats = self.t(global_features)
        global_feats = global_feats.view((int(global_feats.size(0) / self.num_views), self.num_views, -1))
        global_feats = torch.max(global_feats, 1)[0]
        global_feats = self.g(global_feats)

        features = self.convup(features)
        pix_feats = self.upsample(features)
        return global_feats, pix_feats

    # global pooling


class Model2D_MV(nn.Module):
    def __init__(self, feature_dim=128, num_views=12):
        super(Model2D_MV, self).__init__()
        self.f = []
        self.num_views = num_views
        resnet = resnet50()
        for name, module in resnet.named_children():
            if name == "conv1":
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.t = nn.Linear(2048, 1024, bias=False)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.t(feature)
        feature = feature.view((int(feature.size(0) / self.num_views), self.num_views, -1))
        feature = torch.max(feature, 1)[0]
        out = self.g(feature)
        return out


class Model2D_pixel_224(nn.Module):
    def __init__(self, feature_dim=128, num_views=12):
        super(Model2D_pixel_224, self).__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views
        modules = []
        resnet = resnet50()
        resnet.load_state_dict(torch.load("resnet50-0676ba61.pth"))
        for name, module in resnet.named_children():
            if (
                not isinstance(module, nn.Linear)
                and not isinstance(module, nn.MaxPool2d)
                and not isinstance(module, nn.AdaptiveAvgPool2d)
            ):
                modules.append(module)
            # modules.append(module)
        self.f = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # upsampling projection layer
        self.convup = nn.Conv2d(2048, feature_dim, kernel_size=(1, 1))
        self.upsample = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=True)
        # projection layer
        self.t = nn.Linear(2048, 1024, bias=False)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, input):
        features = self.f(input)
        global_features = self.pool(features).squeeze()
        global_feats = self.t(global_features)
        global_feats = global_feats.view((int(global_feats.size(0) / self.num_views), self.num_views, -1))
        global_feats = torch.max(global_feats, 1)[0]
        global_feats = self.g(global_feats)
        features = self.convup(features)
        pix_feats = self.upsample(features)
        return global_feats, pix_feats
