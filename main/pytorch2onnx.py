import onnx
import torch
import argparse
import numpy
import onnxruntime as ort
from onnx_tf.backend import prepare

import tensorflow as tf

from torchsummary import summary

from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
from torch.nn import functional as F
from torch.nn.parallel.data_parallel import DataParallel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--joint', type=int, dest='joint')
    parser.add_argument('--modelpath', type=str, dest='modelpath')
    parser.add_argument('--backbone', type=str, dest='backbone')
    parser.add_argument('--frontbone', type=str, dest='frontbone')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

args = parse_args()

class cfg:
    ## input, output
    input_shape = (256, 256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 64
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

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


"""# Pytorch MobileNeXt"""


class SandGlass(nn.Module):
    def __init__(self, input_dim, output_dim, stride, t):
        super(SandGlass, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.residual = True if stride == 1 and input_dim == output_dim else False

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
            nn.BatchNorm2d(output_dim),
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
                 width_multi=1.0,
                 divisor=8,
                 sand_glass_setting=None):
        super(MobileNeXt, self).__init__()

        block = SandGlass

        init_channel = 32
        last_channel = 2048

        init_channel = _make_divisible(init_channel * width_multi, divisor)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_multi), divisor)

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
                [6, self.last_channel / width_multi, 1, 1],
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


"""# Pytorch ResPoseNet"""


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):

        resnet_spec = {'res18': (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
                       'res34': (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
                       'res50': (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
                       'res101': (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
                       'res152': (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")


class HeadNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
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


class CustomNet(nn.Module):

    def __init__(self, joint_num):
        self.inplanes = 2048
        self.hidplanes = 64
        self.outplanes = 256

        super(CustomNet, self).__init__()

        self.deconv_layer_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.inplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1,
                      groups=1, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.outplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1,
                      groups=self.outplanes, bias=False),
            nn.BatchNorm2d(self.outplanes),
            nn.ReLU(inplace=True))
        self.deconv_layer_3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=self.outplanes, out_channels=self.outplanes, kernel_size=3, stride=1, padding=1,
                      groups=self.outplanes, bias=False),
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


def soft_argmax_pytorch(heatmaps, joint_num):
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim * cfg.output_shape[0] * cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * \
             torch.nn.parallel.comm.broadcast(torch.arange(1, cfg.output_shape[1] + 1).type(torch.cuda.FloatTensor),
                                              devices=[accu_x.device.index])[0]
    accu_y = accu_y * \
             torch.nn.parallel.comm.broadcast(torch.arange(1, cfg.output_shape[0] + 1).type(torch.cuda.FloatTensor),
                                              devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.nn.parallel.comm.broadcast(torch.arange(1, cfg.depth_dim + 1).type(torch.cuda.FloatTensor),
                                                       devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out


class ResPoseNet_PyTorch(nn.Module):
    def __init__(self, backbone, head, joint_num):
        super(ResPoseNet_PyTorch, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)
        hm = self.head(fm)
        coord = soft_argmax_pytorch(hm, self.joint_num)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1] + loss_coord[:, :, 2] * target_have_depth) / 3.

            return loss_coord


def get_pose_net(backbone_str, frontbone_str, is_train, joint_num):
    if backbone_str == 'mobxt':
        print("load MobileNeXt")
        backbone = MobileNeXt(width_multi=1.0)
    # elif backbone_str == 'mobxt_':
    #     print("load MobileNext_")
    #     backbone = MobileNeXt_(width_mult=1.0)
    else:
        print("load ResNet")
        backbone = ResNetBackbone(str)

    if frontbone_str == 'custom':
        print("load CustomNet")
        head_net = CustomNet(joint_num)
    else:
        print("load HeadNet")
        head_net = HeadNet(joint_num)
    if is_train:
        backbone.init_weights()
        head_net.init_weights()

    model = ResPoseNet_PyTorch(backbone, head_net, joint_num)
    return model

dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

pytorch_model = get_pose_net(args.backbone, args.frontbone, False, args.joint)
pytorch_model = DataParallel(pytorch_model).cuda()

#Load the pretrained model
pytorch_model.load_state_dict(torch.load(args.modelpath)['network'])

#Time to transfer weights
single_pytorch_model = pytorch_model.module
single_pytorch_model.eval()
summary(single_pytorch_model, (3, 256, 256))

ONNX_PATH="../output/baseline.onnx"

torch.onnx.export(
    model=single_pytorch_model,
    args=dummy_input,
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output']
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

pytorch_result = single_pytorch_model(dummy_input)
pytorch_result = pytorch_result.cpu().detach().numpy()
print("pytorch_model output {}".format(pytorch_result.shape), pytorch_result)

ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
outputs = numpy.array(outputs[0])
print("onnx_model ouput size{}".format(outputs.shape), outputs)

print("difference", numpy.linalg.norm(pytorch_result-outputs))

TF_PATH = "../output/baseline" # where the representation of tensorflow model will be stored

# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)  # creating TensorflowRep object

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)

TFLITE_PATH = "../output/baseline.tflite"

PB_PATH = "../output/baseline/saved_model.pb"

# make a converter object from the saved tensorflow file
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PB_PATH, input_arrays=['input'], output_arrays=['output'])
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
