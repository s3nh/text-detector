import torch
import torch.nn as nn 
import torch.nn.functional as F
from text_reco.models.craft.basenet.vgg16_bn import vgg16_bn, init_weights

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channel + mid_channel, mid_channel,  kernel_size = 1), 
                nn.BatchNorm2d(mid_channel), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(mid_channel, out_channel, kernel_size = 3, padding = 1), 
                nn.BatchNorm2d(out_channel), 
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        self.basenet = vgg16_bn(pretrained, freeze)
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        n_classes = 2
        self.conv_cls = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(32, 32, 3, 1), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(32, 16, 3, 1), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(16, 16, 1), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(16, n_classes, kernel_size=1),)

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size = sources[2].size()[2:], mode = 'bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size = sources[3].size()[2:], mode = 'bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size = sources[4].size()[2:], mode = 'bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim =1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature

