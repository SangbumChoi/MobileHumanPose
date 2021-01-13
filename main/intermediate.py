import torch
import argparse
import numpy as np
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchsummary import summary
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton
from dataset import generate_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--epoch', type=int, dest='test_epoch')
    parser.add_argument('--input_image', type=str, dest='image')
    parser.add_argument('--jointnum', type=int, dest='joint')
    parser.add_argument('--backbone', type=str, dest='backbone')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if not args.joint:
        assert print("please insert number of joint")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# joint set
joint_num = args.joint
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
if joint_num == 18:
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
if joint_num == 21:
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# snapshot load
model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
model = get_pose_net(args.backbone, args.frontbone, False, joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model = model.module
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = args.image
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
