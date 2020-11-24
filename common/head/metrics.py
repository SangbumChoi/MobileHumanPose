import torch.nn as nn
import torch
from config import cfg

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HeadNet(nn.Module):

    def __init__(self, in_features, joint_num):
        self.inplanes = in_features
        self.outplanes = 256

        super(HeadNet, self).__init__()

        self.deconv_layers = self._make_deconv_layer(3)
        self.final_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=self.outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class CustomNet1(nn.Module):

    def __init__(self, in_features, joint_num):
        self.inplanes = in_features
        self.hidplanes = 64
        self.outplanes = 256

        super(CustomNet1, self).__init__()

        self.deconv_layer_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1, groups=self.hidplanes, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.outplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1, groups=self.outplanes, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.final_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.deconv_layer_1(x)
        x = self.deconv_layer_2(x)
        x = self.deconv_layer_3(x)
        x = self.final_layer(x)
        return x

    def init_weights(self):
        for name, m in self.deconv_layer_1.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layer_2.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layer_3.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class CustomNet2(nn.Module):

    def __init__(self, in_features, joint_num):
        self.inplanes = in_features
        self.hidplanes = 256
        self.outplanes = 256

        super(CustomNet2, self).__init__()

        self.deconv_layer_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1, groups=self.hidplanes, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.outplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1, groups=self.outplanes, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x1 = self.deconv_layer_1(x)
        w1 = self.upsample_layer(x1)
        x2 = self.deconv_layer_2(x1)
        w2 = self.upsample_layer(x2)
        x3 = self.deconv_layer_3(x2+w1)
        x = self.final_layer(x3+w2)
        return x

    def init_weights(self):
        for name, m in self.deconv_layer_1.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layer_2.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.deconv_layer_3.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class PartNet(nn.Module):

    def __init__(self, in_features, joint_num):
        self.inplanes = in_features
        self.outplanes = 256
        self.ratio = [1/4, 3/4]

        self.hidplanes_s = _make_divisible(int(self.outplanes * self.ratio[0]), 8)
        self.hidplanes_l = _make_divisible(int(self.outplanes * self.ratio[1]), 8)

        super(PartNet, self).__init__()

        self.deconv_layer_1_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_1_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.deconv_layer_2_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_s, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_s, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_2_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_l, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_l, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.deconv_layer_3_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_s, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_s, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_3_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_l, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_l, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.final_layer_s = nn.Conv2d(
            in_channels=self.hidplanes_s,
            out_channels=int(joint_num * cfg.depth_dim * self.ratio[0]),
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.final_layer_l = nn.Conv2d(
            in_channels=self.hidplanes_l,
            out_channels=int(joint_num * cfg.depth_dim * self.ratio[1]),
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        xl = self.deconv_layer_1_l(x)
        xs = self.deconv_layer_1_s(x)
        xl = self.deconv_layer_2_l(xl)
        xs = self.deconv_layer_2_s(xs)
        xl = self.deconv_layer_3_l(xl)
        xs = self.deconv_layer_3_s(xs)
        xl = self.final_layer_l(xl)
        xs = self.final_layer_s(xs)

        x = torch.cat((xl, xs), dim=1)
        return x

    def init_weights(self):
        deconv_layers = [self.deconv_layer_1_l, self.deconv_layer_1_s,
                  self.deconv_layer_2_l, self.deconv_layer_2_s,
                  self.deconv_layer_3_l, self.deconv_layer_3_s]
        final_layers = [self.final_layer_l, self.final_layer_s]
        for layer in deconv_layers:
            for name, m in layer.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for layer in final_layers:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

class PartNet2(nn.Module):

    def __init__(self, in_features, joint_num):
        self.inplanes = in_features
        self.outplanes = 256
        self.ratio = [1/4, 2/4]

        self.hidplanes_s = _make_divisible(int(self.outplanes * self.ratio[0]), 8)
        self.hidplanes_l = _make_divisible(int(self.outplanes * self.ratio[1]), 8)

        super(PartNet2, self).__init__()

        self.deconv_layer_1_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_1_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.deconv_layer_2_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_s, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_s, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_2_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_l, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_l, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.deconv_layer_3_s = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_s, out_channels=self.hidplanes_s, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_s, bias=False),
            nn.BatchNorm2d(self.hidplanes_s),
            nn.ReLU(inplace=True))
        self.deconv_layer_3_l = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.hidplanes_l, out_channels=self.hidplanes_l, kernel_size=3, stride=1, padding=1,
                      groups=self.hidplanes_l, bias=False),
            nn.BatchNorm2d(self.hidplanes_l),
            nn.ReLU(inplace=True))
        self.final_layer_s = nn.Conv2d(
            in_channels=self.hidplanes_s,
            out_channels=int(joint_num * cfg.depth_dim * self.ratio[0]),
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.final_layer_l = nn.Conv2d(
            in_channels=self.hidplanes_l,
            out_channels=int(joint_num * cfg.depth_dim * self.ratio[1]),
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        xl = self.deconv_layer_1_l(x)
        xs1 = self.deconv_layer_1_s(x)
        xs2 = self.deconv_layer_1_s(x)
        xl = self.deconv_layer_2_l(xl)
        xs1 = self.deconv_layer_2_s(xs1)
        xs2 = self.deconv_layer_2_s(xs2)
        xl = self.deconv_layer_3_l(xl)
        xs1 = self.deconv_layer_3_s(xs1)
        xs2 = self.deconv_layer_3_s(xs2)
        xl = self.final_layer_l(xl)
        xs1 = self.final_layer_s(xs1)
        xs2 = self.final_layer_s(xs2)

        x = torch.cat((xl, xs1, xs2), dim=1)
        return x

    def init_weights(self):
        deconv_layers = [self.deconv_layer_1_l, self.deconv_layer_1_s,
                  self.deconv_layer_2_l, self.deconv_layer_2_s,
                  self.deconv_layer_3_l, self.deconv_layer_3_s]
        final_layers = [self.final_layer_l, self.final_layer_s]
        for layer in deconv_layers:
            for name, m in layer.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for layer in final_layers:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)