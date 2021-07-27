from torchvision.models.vgg import VGG
import torch.nn as nn

class FCN_32s(nn.Module):
    def __init__(self, in_channels, n_class, pretrained_model):
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
    def __init__(self, in_channels, n_class, pretrained_model):
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
    def __init__(self, in_channels, n_class, pretrained_model):
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
    def __init__(self, in_channels, n_class, pretrained_model):
        super(FCNs, self).__init__()
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
        super().__init__()
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


ranges = {'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))}