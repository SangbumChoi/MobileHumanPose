import torch
import torch.nn as nn
from torch.nn import functional as F
from backbone import *
from head import *
from config import cfg
import os.path as osp

model_urls = {
    'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ResNext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

BACKBONE_DICT = {
    'GhostNet': GhostNet, 'MobileNetV3': MobileNetV3,
    'MobileNeXt': MobileNeXt, 'MobileNetV2': MobileNetV2,
    'MNasNet':MNasNet,
    'ResNet18':ResNet18,
    'ResNet50':ResNet50,
    'MGG':MobGG,
    'MGG_NO_CONCAT':MobGG_No_Concat
    }

HEAD_DICT = {'HeadNet': HeadNet, 'Custom1' : CustomNet1, 'Custom2' : CustomNet2, 'PartNet': PartNet, 'PartNet2' : PartNet2
             }

def soft_argmax(heatmaps, joint_num):

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.nn.parallel.comm.broadcast(torch.arange(1,cfg.output_shape[1]+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.nn.parallel.comm.broadcast(torch.arange(1,cfg.output_shape[0]+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.nn.parallel.comm.broadcast(torch.arange(1,cfg.depth_dim+1).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out

class ResPoseNet(nn.Module):
    def __init__(self, backbone, head, joint_num):
        super(CustomNet, self).__init__()
        self.backbone = backbone
        self.head = head
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)
        hm = self.head(fm)
        coord = soft_argmax(hm, self.joint_num)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.

            return loss_coord

class CustomNet(nn.Module):
    def __init__(self, backbone, joint_num):
        super(CustomNet, self).__init__()
        self.backbone = backbone
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)
        coord = soft_argmax(fm, self.joint_num)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.

            return loss_coord

def get_pose_net(backbone_str, head_str, is_train, joint_num):
    INPUT_SIZE = cfg.input_shape
    EMBEDDING_SIZE = cfg.embedding_size if not cfg.teacher_train else 2048 # feature dimension
    assert INPUT_SIZE == (256, 256)
    backbone = BACKBONE_DICT[backbone_str](INPUT_SIZE, EMBEDDING_SIZE)
    print("=" * 60)
    print("{} Backbone Generated".format(backbone_str))
    print("=" * 60)

    if backbone_str == 'MGG' or 'MGG_NO_CONCAT':
        model = CustomNet(BACKBONE_DICT[backbone_str](INPUT_SIZE), joint_num)
        return model

    head = HEAD_DICT[head_str](in_features = EMBEDDING_SIZE, joint_num = joint_num)
    print("=" * 60)
    print("{} Head Generated".format(head_str))
    print("=" * 60)
    if is_train:
        if cfg.pre_train:
            if cfg.pre_train_name is not None:
                file_path = osp.join(cfg.pretrain_dir, cfg.pre_train_name)
                pretrained_dict = torch.load(file_path)
                backbone.init_weights(model_urls[backbone_str], pretrained_dict)
            else:
                backbone.init_weights(model_urls[backbone_str], None)
            print("=" * 60)
            print("{} pretrain loaded".format(backbone_str))
            print("=" * 60)
        else:
            backbone.init_weights(None)
            print("=" * 60)
            print("{} random initialized".format(backbone_str))
            print("=" * 60)
        head.init_weights()

    model = CustomNet(backbone, head, joint_num)
    return model
