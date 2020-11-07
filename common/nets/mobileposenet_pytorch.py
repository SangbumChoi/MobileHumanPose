from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    Sequential, Module
import torch
import torch.nn as nn
from .common import ECA_Layer, SEBlock, CbamBlock, GCT
from torchsummary import summary


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

##################################  MobileFaceNet #############################################################

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, attention, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.attention = attention
        if self.attention == 'eca':
            self.attention_layer = ECA_Layer(out_c)
        elif self.attention == 'se':
            self.attention_layer = SEBlock(out_c)
        elif self.attention == 'cbam':
            self.attention_layer = CbamBlock(out_c)
        elif self.attention == 'gct':
            self.attention_layer = GCT(out_c)

        self.residual = residual

        self.attention = attention  # se, eca, cbam

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.attention != 'none':
            x = self.attention_layer(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, attention, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, attention, residual=True, kernel=kernel, padding=padding, stride=stride,
                                      groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MobilePoseNet(Module):
    def __init__(self, attention='none'):
        super(MobilePoseNet, self).__init__()

        init_channel = 32
        hid_one_channel = 96
        hid_two_channel = 160
        hid_three_channel = 320
        out_channel = 2048

        self.conv1 = Conv_block(3, init_channel, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(init_channel, init_channel, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=init_channel)
        self.conv_23 = Depth_Wise(init_channel, hid_one_channel, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=hid_one_channel)
        self.conv_3 = Residual(hid_one_channel, attention, num_block=4, groups=hid_one_channel, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(hid_one_channel, hid_two_channel, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=hid_two_channel)
        self.conv_4 = Residual(hid_two_channel, attention, num_block=4, groups=hid_two_channel, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(hid_two_channel, hid_three_channel, attention, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=hid_three_channel)
        self.conv_5 = Residual(hid_three_channel, attention, num_block=2, groups=hid_three_channel, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(hid_three_channel, out_channel, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        return conv_features

if __name__ == "__main__":
    model = MobilePoseNet(attention='cbam')
    print(model)
    test_data = torch.rand(1, 3, 256, 256)
    test_outputs = model(test_data)
    print(test_outputs.size())
    summary(model, (3, 256, 256))
