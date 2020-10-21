"""
non-official PyTorch implementation of MobileNeXt from paper:
Rethinking Bottleneck Structure for Efficient Mobile Network Design
https://arxiv.org/abs/2007.02269
modified from mobilenetv2 torchvision implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
"""

import math
import torch
from torch import nn

__all__ = ['MobileNeXt', 'mobilenext_pytorch.py']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class LastConvLayer(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(LastConvLayer, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
                ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] + out[:, :self.identity_tensor_channels, :,
                                                                               :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out


class MobileNeXt(nn.Module):
    def __init__(self,
                 width_mult=1.0,
                 identity_tensor_multiplier=1.0,
                 sand_glass_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNeXt main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            identity_tensor_multiplier(float): Identity tensor multiplier - reduce the number of element-wise additions in each block
            sand_glass_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNeXt, self).__init__()

        if block is None:
            block = SandGlass

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280
        res_channel = 2048

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                [2, 96, 1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 2, 1],
                [6, self.last_channel / width_mult, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(sand_glass_setting) == 0 or len(sand_glass_setting[0]) != 4:
            raise ValueError("sand_glass_setting should be non-empty "
                             "or a 4-element list, got {}".format(sand_glass_setting))

        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t,
                                      identity_tensor_multiplier=identity_tensor_multiplier, norm_layer=norm_layer))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # resnet outputshape matching
        self.last_layer = LastConvLayer(last_channel, res_channel)
        # building classifier
        # self.classifier = nn.Linear(self.last_channel, num_classes)

        # weight initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = self.last_layer(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == "__main__":
    model = MobileNeXt(width_mult=1.0, identity_tensor_multiplier=1.0)
    # print(model)

    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    # print(test_outputs.size())