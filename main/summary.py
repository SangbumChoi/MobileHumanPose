import torch
import argparse
import os
import os.path as osp
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_pose_net
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--epoch', type=int, dest='test_epoch')
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

single_model = model.module

summary(single_model, (3, 256, 256))

input = torch.randn(1, 3, 256, 256).cuda()
macs, params = profile(single_model, inputs=(input,))
macs, params = clever_format([macs, params], "%.3f")
flops, params1 = get_model_complexity_info(single_model, (3, 256, 256),as_strings=True, print_per_layer_stat=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print('{:<30}  {:<8}'.format('Number of parameters: ', params1))
