import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class NaivePyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim):
        super(NaivePyramidPoolingModule, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        out = [x]
        out.append(F.interpolate(self.feature(x),
                                 x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
                 naive_ppm=False, criterion=nn.CrossEntropyLoss(ignore_index=0), pretrained=True):
        super(PSPNet, self).__init__()
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        self.layers = layers
        if layers == 18:
            resnet = models.resnet18(pretrained=pretrained)
        elif layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        else:
            resnet = models.resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # fea_dim = resnet.layer4[-1].conv3.out_channels
        fea_dim = 512
        if layers == 18:
            for n, m in self.layer3[0].named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                if 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4[0].named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                if 'downsample.0' in n:
                    m.stride = (1, 1)

        else:
            fea_dim = 2048
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if self.training:
            self.aux_head = nn.Sequential(
                nn.Conv2d(fea_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

        if use_ppm:
            if naive_ppm:
                self.ppm = NaivePyramidPoolingModule(fea_dim, int(fea_dim))
            else:
                self.ppm = PyramidPoolingModule(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls_head = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # use different layers output as auxiliary supervision
        # the reception field of layer3 in Res18 is too small, may influence the
        # segmentation result, so we use the layer4 output for auxiliary supervision.
        if self.layers != 18:
            feat_tmp= self.layer3(x)
            feat = self.layer4(feat_tmp)
            feat = self.ppm(feat)
        else:
            x = self.layer3(x)
            feat_tmp = self.layer4(x)
            feat = self.ppm(feat_tmp)

        pred = self.cls_head(feat)
        if self.zoom_factor != 1:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux_pred = self.aux_head(feat_tmp)
            if self.zoom_factor != 1:
                aux_pred = F.interpolate(aux_pred, size=(h, w), mode='bilinear', align_corners=True)

            main_loss = self.criterion(pred, y)
            aux_loss = self.criterion(aux_pred, y)
            return pred.max(1)[1], main_loss, aux_loss
        else:
            return pred
