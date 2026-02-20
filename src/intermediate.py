import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

from common.utils.pose_utils import process_bbox
from data.dataset import generate_patch_image
from src.config import cfg
from src.model import get_pose_net

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--input_image', '-i', required=True)
parser.add_argument('--jointnum', type=int, default=18)
parser.add_argument('--gpu', '--backbone', help='Deprecated. Edit config.py')
args = parser.parse_args()
cudnn.benchmark = True

# joint set
joint_num = args.jointnum
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
if joint_num == 18:
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
if joint_num == 21:
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# snapshot load
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % args.epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
model = get_pose_net(cfg.backbone, False, joint_num)
if torch.cuda.is_available():
    model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model = model.module if hasattr(model, 'module') else model
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = args.input_image
assert osp.exists(img_path), 'Cannot find image at ' + img_path
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox_list = [
    [139.41, 102.25, 222.39, 241.57],\
    [287.17, 61.52, 74.88, 165.61],\
    [540.04, 48.81, 99.96, 223.36],\
    [372.58, 170.84, 266.63, 217.19],\
    [0.5, 43.74, 90.1, 220.09]
] # xmin, ymin, width, height
root_depth_list = [11250.5732421875, 15522.8701171875, 11831.3828125, 8852.556640625, 12572.5966796875] # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
assert len(bbox_list) == len(root_depth_list)
person_num = len(bbox_list)

# extractor
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

for n in range(person_num):
    bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
    img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
    img = transform(img).cuda()[None,:,:,:]

    model.backbone.deonv1.register_forward_hook(get_activation('%d' % n))
    # forward
    with torch.no_grad():
        pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

plt.figure(figsize=(32, 32))
a = activation['0'] - activation['1']
b = torch.sum(a, dim=1)
print(b)
for i in range(person_num):
    image = activation['%d'%i]
    print(image.size())
    sum_image = torch.sum(image[0], dim=0)
    print(sum_image.size())
    plt.subplot(1, person_num, i+1)
    plt.imshow(sum_image.cpu(), cmap='gray')
    plt.axis('off')

plt.show()
plt.close()
