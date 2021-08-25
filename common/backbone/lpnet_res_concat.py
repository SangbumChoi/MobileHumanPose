import torch.nn as nn
import torch
from torchsummary import summary
from config import cfg

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
    It can be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_ch , out_ch, kernel_size=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch),
            activation_layer(out_ch),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(out_planes)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, activation_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation_layer=activation_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class LpNetResConcat(nn.Module):
    def __init__(self,
                 input_size,
                 joint_num,
                 input_channel = 48,
                 embedding_size = 2048,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 activation_layer=None,
                 inverted_residual_setting=None):

        super(LpNetResConcat, self).__init__()

        assert input_size[1] in [256]

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.PReLU # PReLU does not have inplace True
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 64, 1, 1],  #[-1, 48, 256, 256]
                [6, 48, 2, 2],  #[-1, 48, 128, 128]
                [6, 48, 3, 2],  #[-1, 48, 64, 64]
                [6, 64, 4, 2],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 2], #[-1, 160, 8, 8]
                [6, 320, 1, 1], #[-1, 320, 8, 8]
            ]

        # building first layer
        inp_channel = [_make_divisible(input_channel * width_mult, round_nearest),
                         _make_divisible(input_channel * width_mult, round_nearest) + inverted_residual_setting[0][1],
                         inverted_residual_setting[0][1] + inverted_residual_setting[1][1],
                         inverted_residual_setting[1][1] + inverted_residual_setting[2][1],
                         inverted_residual_setting[2][1] + inverted_residual_setting[3][1],
                         inverted_residual_setting[3][1] + inverted_residual_setting[4][1],
                         inverted_residual_setting[4][1] + inverted_residual_setting[5][1],
                         inverted_residual_setting[5][1] + inverted_residual_setting[6][1],
                         inverted_residual_setting[6][1] + embedding_size,
                         256 + embedding_size,
                       ]
        self.first_conv = ConvBNReLU(3, inp_channel[0], stride=1, norm_layer=norm_layer, activation_layer=activation_layer)

        inv_residual = []
        # building inverted residual blocks
        j = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                input_channel = inp_channel[j] if i == 0 else output_channel
                inv_residual.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, activation_layer=activation_layer))
            j += 1
        # make it nn.Sequential
        self.inv_residual = nn.Sequential(*inv_residual)

        self.last_conv = ConvBNReLU(inp_channel[j], embedding_size, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)

        self.deonv0 = DoubleConv(inp_channel[j+1], 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deonv1 = DoubleConv(2304, 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deonv2 = DoubleConv(512, 256, norm_layer=norm_layer, activation_layer=activation_layer)

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels= joint_num * cfg.depth_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x0 = self.first_conv(x)
        x1 = self.inv_residual[0:1](x0)
        x2 = self.inv_residual[1:3](torch.cat([x0, x1], dim=1))
        x0 = self.inv_residual[3:6](torch.cat([self.avgpool(x1), x2], dim=1))
        x1 = self.inv_residual[6:10](torch.cat([self.avgpool(x2), x0], dim=1))
        x2 = self.inv_residual[10:13](torch.cat([self.avgpool(x0), x1], dim=1))
        x0 = self.inv_residual[13:16](torch.cat([self.avgpool(x1), x2], dim=1))
        x1 = self.inv_residual[16:17](torch.cat([self.avgpool(x2), x0], dim=1))
        x2 = self.last_conv(torch.cat([x0, x1], dim=1))
        x0 = self.deonv0(torch.cat([x1, x2], dim=1))
        x1 = self.deonv1(torch.cat([self.upsample(x2), x0], dim=1))
        x2 = self.deonv2(torch.cat([self.upsample(x0), x1], dim=1))
        x0 = self.final_layer(x2)
        return x0

    def init_weights(self):
        for i in [self.deconv0, self.deconv1, self.deconv2]:
            for name, m in i.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for j in [self.first_conv, self.inv_residual, self.last_conv, self.final_layer]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = LpNetResConcat((256, 256), 18)
    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    # print(test_outputs.size())
    summary(model, (3, 256, 256))
