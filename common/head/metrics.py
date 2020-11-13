import torch.nn as nn
from config import cfg

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