import torch
from torch import nn
from torchsummary import summary

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

class SandGlass(nn.Module):
    def __init__(self, input_dim, output_dim, stride, t):
        super(SandGlass, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.residual = True if stride==1 and input_dim==output_dim else False

        hid_dim = input_dim // t

        layers = []
        # Residual Sub-Block 1
        layers.extend([
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU6(inplace=True)
        ])
        # Residual Sub-Block 2
        layers.extend([
            nn.Conv2d(input_dim, hid_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(hid_dim)
        ])
        # Residual Sub-Block 3
        layers.extend([
            nn.Conv2d(hid_dim, output_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU6(inplace=True)
        ])
        # Residual Sub-Block 4
        layers.extend([
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride, padding=1, groups=output_dim, bias=False),
            nn.BatchNorm2d(output_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.residual == True:
            out = x + out
            return out
        else:
            return out

class MobileNeXt(nn.Module):
    def __init__(self,
                 width_mult=1.0,
                 divisor=8,
                 sand_glass_setting=None):
        super(MobileNeXt, self).__init__()

        block = SandGlass

        init_channel = 32
        last_channel = 1024

        init_channel = _make_divisible(init_channel * width_mult, divisor)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), divisor)

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
        self.block1 = nn.Sequential(
            nn.Conv2d(3, init_channel, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(init_channel),
            nn.ReLU6(inplace=True)
        )

        layers = []
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c, divisor)
            for i in range(b):
                stride = s if i == 0 else 1
                layers.append(block(init_channel, output_channel, stride, t))
                init_channel = output_channel

        self.sandglass_conv = nn.Sequential(*layers)

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

    def forward(self, x):
        x = self.block1(x)
        x = self.sandglass_conv(x)
        return x

if __name__ == "__main__":
    model = MobileNeXt()
    print(model)
    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    print(test_outputs.size())
    summary(model, (3, 256, 256))