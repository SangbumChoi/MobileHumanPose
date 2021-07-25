import torch.nn as nn
import torch
from torchsummary import summary

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

class DeConv(nn.Sequential):
    def __init__(self, in_ch, mid_ch, out_ch, norm_layer=None, activation_layer=None):
        super(DeConv, self).__init__(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            norm_layer(mid_ch),
            activation_layer(mid_ch),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
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

class LpNetSkiConcat(nn.Module):
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

        super(LpNetSkiConcat, self).__init__()

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
                [1, 64, 1, 2],  #[-1, 48, 256, 256]
                [6, 48, 2, 2],  #[-1, 48, 128, 128]
                [6, 48, 3, 2],  #[-1, 48, 64, 64]
                [6, 64, 4, 2],  #[-1, 64, 32, 32]
                [6, 96, 3, 2],  #[-1, 96, 16, 16]
                [6, 160, 3, 1], #[-1, 160, 8, 8]
                [6, 320, 1, 1], #[-1, 320, 8, 8]
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        self.first_conv = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=activation_layer)

        inv_residual = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                inv_residual.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, activation_layer=activation_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.inv_residual = nn.Sequential(*inv_residual)

        self.last_conv = ConvBNReLU(input_channel, embedding_size, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)

        self.deconv0 = DeConv(embedding_size, _make_divisible(inverted_residual_setting[-3][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv1 = DeConv(256, _make_divisible(inverted_residual_setting[-4][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)
        self.deconv2 = DeConv(256, _make_divisible(inverted_residual_setting[-5][-3] * width_mult, round_nearest), 256, norm_layer=norm_layer, activation_layer=activation_layer)

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels= joint_num * 32,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.inv_residual[0:6](x)
        x2 = x
        x = self.inv_residual[6:10](x)
        x1 = x
        x = self.inv_residual[10:13](x)
        x0 = x
        x = self.inv_residual[13:16](x)
        x = self.inv_residual[16:](x)
        z = self.last_conv(x)
        z = torch.cat([x0, z], dim=1)
        z = self.deconv0(z)
        z = torch.cat([x1, z], dim=1)
        z = self.deconv1(z)
        z = torch.cat([x2, z], dim=1)
        z = self.deconv2(z)
        z = self.final_layer(z)
        return z

if __name__ == "__main__":
    model = LpNetSkiConcat((256, 256), 18).cuda()
    test_data = torch.rand(1, 3, 256, 256).cuda()
    test_outputs = model(test_data)
    print(test_outputs.size())
    summary(model, (3, 256, 256))
