"""
Config: single source of truth. Edit here, not via CLI.
Backbone and dataset set in this file. Use torchrun for multi-GPU (see runs/train.sh).
"""
import os
import os.path as osp

# -----------------------------------------------------------------------------
# Model (edit here; --backbone deprecated)
backbone = 'LPSKI'  # LPSKI | LPRES | LPWO

# -----------------------------------------------------------------------------
# Dataset
trainset_3d = ['Dummy']   # Human36M | MuCo | Dummy
trainset_2d = ['MSCOCO']  # MSCOCO | MPII (HF whyen-wang/coco_keypoints)
testset = 'MSCOCO'        # Human36M | MuPoTS | MSCOCO | Dummy

# -----------------------------------------------------------------------------
# Paths (config lives in src/3dpose_estimator/; root is project root)
cur_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.join(cur_dir, '..', '..')
data_dir = osp.join(root_dir, 'data')
output_dir = osp.join(root_dir, 'output')
model_dir = osp.join(output_dir, 'model_dump')
log_dir = osp.join(output_dir, 'log')
result_dir = osp.join(output_dir, 'result')
vis_dir = osp.join(output_dir, 'vis')
pretrain_dir = osp.join(output_dir, 'pre_train')

# -----------------------------------------------------------------------------
# Input/output
input_shape = (256, 256)
output_shape = (input_shape[0] // 8, input_shape[1] // 8)
width_multiplier = 1.0
depth_dim = 32
bbox_3d_shape = (2000, 2000, 2000)
pixel_mean = (0.485, 0.456, 0.406)
pixel_std = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Training
embedding_size = 2048
lr_dec_epoch = [5, 10]   # fewer for MSCOCO + Dummy smoke test
end_epoch = 12
lr = 1e-3
lr_dec_factor = 10
batch_size = 32

# -----------------------------------------------------------------------------
# Testing
test_batch_size = 32
flip_test = True
use_gt_info = True

# -----------------------------------------------------------------------------
# System (torchrun sets world_size; else single process)
num_thread = min(8, os.cpu_count() or 8)
continue_train = False

# Resolve num_gpus: from torch.distributed if using torchrun, else 1
def _get_num_gpus():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1

# Lazy init after torch
num_gpus = 1  # updated in train/test

# -----------------------------------------------------------------------------
# Setup paths and imports
for d in (model_dir, log_dir, result_dir, vis_dir):
    os.makedirs(d, exist_ok=True)

from common.utils.dir_utils import add_pypath

add_pypath(data_dir)
for ds in trainset_3d + trainset_2d + [testset]:
    add_pypath(osp.join(data_dir, ds))

# Backward-compat cfg namespace
class Cfg:
    backbone = backbone
    trainset_3d = trainset_3d
    trainset_2d = trainset_2d
    testset = testset
    root_dir = root_dir
    data_dir = data_dir
    output_dir = output_dir
    model_dir = model_dir
    log_dir = log_dir
    result_dir = result_dir
    vis_dir = vis_dir
    input_shape = input_shape
    output_shape = output_shape
    width_multiplier = width_multiplier
    depth_dim = depth_dim
    bbox_3d_shape = bbox_3d_shape
    pixel_mean = pixel_mean
    pixel_std = pixel_std
    embedding_size = embedding_size
    lr_dec_epoch = lr_dec_epoch
    end_epoch = end_epoch
    lr = lr
    lr_dec_factor = lr_dec_factor
    batch_size = batch_size
    test_batch_size = test_batch_size
    flip_test = flip_test
    use_gt_info = use_gt_info
    num_thread = num_thread
    continue_train = continue_train
    num_gpus = 1

cfg = Cfg()
