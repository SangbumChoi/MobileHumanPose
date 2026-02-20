import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

from common.utils.pose_utils import pixel2cam, process_bbox
from common.utils.vis import vis_3d_multiple_skeleton, vis_keypoints
from data.dataset import generate_patch_image
from src.config import cfg
from src.model import get_pose_net

try:
    from .detection import get_person_bboxes
    from .rootnet import get_root_depths
except ImportError:
    from detection import get_person_bboxes
    from rootnet import get_root_depths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', required=True)
    parser.add_argument('--input_image', '-i', required=True)
    parser.add_argument('--detection', choices=['ultralytics', 'hf', 'onnx'], default='ultralytics',
                        help='Human detection backend for bbox')
    parser.add_argument('--onnx_path', default='', help='For detection=onnx')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('X', 'Y', 'W', 'H'),
                        help='Optional single bbox (x,y,w,h) instead of detection')
    parser.add_argument('--gpu', type=str, help='Deprecated.')
    parser.add_argument('--backbone', type=str, help='Deprecated. Edit src/config.py.')
    return parser.parse_args()

args = parse_args()
if args.backbone:
    import warnings
    warnings.warn('--backbone deprecated; edit src/config.py', DeprecationWarning)
cudnn.benchmark = True

# MuCo joint set
joint_num = 18
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
# 'Pelvis' 'RHip' 'RKnee' 'RAnkle' 'LHip' 'LKnee' 'LAnkle' 'Spine1' 'Neck' 'Head' 'Site' 'LShoulder' 'LElbow' 'LWrist' 'RShoulder' 'RElbow' 'RWrist
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
# skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

# snapshot load
model = get_pose_net(cfg.backbone, False, joint_num)
model = DataParallel(model).cuda()
# print("after DataParallel", model)
ckpt = torch.load(args.model_path)
# print("ckpt", ckpt['network'])
model.load_state_dict(ckpt['network'])
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = args.input_image
assert osp.exists(img_path), 'Cannot find image at ' + img_path
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]
focal = [1500, 1500]
princpt = [original_img_width / 2, original_img_height / 2]

# prepare bbox (detection) and root_depth (RootNet)
if args.bbox is not None:
    bbox_list = [list(args.bbox)]
else:
    bbox_list = get_person_bboxes(
        original_img,
        backend=args.detection,
        conf_thres=args.conf,
        onnx_path=args.onnx_path or osp.join(osp.dirname(__file__), 'models', 'person_detector.onnx'),
    )
if not bbox_list:
    bbox_list = [[0, 0, original_img_width, original_img_height]]

root_depth_list = get_root_depths(
    original_img, bbox_list, focal, princpt,
    process_bbox, generate_patch_image,
)
person_num = len(bbox_list)
print(f'Focal: {focal}, princpt: {princpt}, persons: {person_num}')

# for each cropped and resized human image, forward it to PoseNet
output_pose_2d_list = []
output_pose_3d_list = []
for n in range(person_num):
    bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
    img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
    img = transform(img).cuda()[None,:,:,:]

    # forward
    with torch.no_grad():
        pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

    # inverse affine transform (restore the crop and resize)
    pose_3d = pose_3d[0].cpu().numpy()
    pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
    pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
    pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
    img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
    pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
    output_pose_2d_list.append(pose_3d[:,:2].copy())

    # root-relative discretized depth -> absolute continuous depth
    pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
    pose_3d = pixel2cam(pose_3d, focal, princpt)
    output_pose_3d_list.append(pose_3d.copy())

# visualize 2d poses
vis_img = original_img.copy()
for n in range(person_num):
    vis_kps = np.zeros((3,joint_num))
    vis_kps[0,:] = output_pose_2d_list[n][:,0]
    vis_kps[1,:] = output_pose_2d_list[n][:,1]
    vis_kps[2,:] = 1
    vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
cv2.imwrite('output_pose_2d.jpg', vis_img)

# visualize 3d poses
vis_kps = np.array(output_pose_3d_list)
vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')

