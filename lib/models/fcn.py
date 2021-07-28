from torchvision.models.vgg import VGG
from torchvision import models
import torch.nn as nn

class FCN_32s(nn.Module):
    def __init__(self, n_class, in_channels=3, pretrained_model=True):
        super(FCN_32s, self).__init__()
        self.n_class = n_class
        self.pretrained_model = pretrained_model
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.classifier = nn.ConvTranspose2d(32, n_class, kernel_size=1)

    def forward(self, x):
        h = self.pretrained_net(x)
        x5 = h['x5']
        score = self.deconv_1(x5)
        score = self.deconv_2(score)
        score = self.deconv_3(score)
        score = self.deconv_4(score)
        score = self.deconv_5(score)
        score = self.classifier(score)
        return score


class FCN_16s(nn.Module):
    def __init__(self, n_class, in_channels=3, pretrained_model=True):
        super(FCN_16s, self).__init__()
        self.n_class = n_class
        self.pretrained_model = pretrained_model
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.classifier = nn.ConvTranspose2d(32, n_class, kernel_size=1)

    def forward(self, x):
        h = self.pretrained_net(x)
        x5 = h['x5']
        x4 = h['x4']
        score = self.deconv_1(x5)
        score = score + x4
        score = self.deconv_2(score)
        score = self.deconv_3(score)
        score = self.deconv_4(score)
        score = self.deconv_5(score)
        score = self.classifier(score)
        return h


class FCN_8s(nn.Module):
    def __init__(self, n_class, in_channels=3, pretrained_model=True):
        super(FCN_8s, self).__init__()
        self.n_class = n_class
        self.pretrained_model = pretrained_model
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.classifier = nn.ConvTranspose2d(32, n_class, kernel_size=1)

    def forward(self, x):
        h = self.pretrained_net(x)
        x5 = h['x5']
        x4 = h['x4']
        x3 = h['x3']
        score = self.deconv_1(x5)
        score = score + x4
        score = self.deconv_2(score)
        score = score + x3
        score = self.deconv_3(score)
        score = self.deconv_4(score)
        score = self.deconv_5(score)
        score = self.classifier(score)
        return h


class FCNs(nn.Module):
    def __init__(self, n_class, pretrained_net, in_channels=3):
        super(FCNs, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1,
                               output_padding=1),
            nn.ReLU(inplace=True))
        self.classifier = nn.ConvTranspose2d(32, n_class, kernel_size=1)

    def forward(self, x):
        h = self.pretrained_net(x)
        x5 = h['x5']
        x4 = h['x4']
        x3 = h['x3']
        x2 = h['x2']
        x1 = h['x1']
        score = self.deconv_1(x5)
        score = score + x4
        score = self.deconv_2(score)
        score = score + x3
        score = self.deconv_3(score)
        score = score + x2
        score = self.deconv_4(score)
        score = score + x1
        score = self.deconv_5(score)
        score = self.classifier(score)
        return h


class VGG_16(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
