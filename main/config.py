import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # training set
    # 3D: Human36M, MuCo
    # 2D: MSCOCO, MPII 
    trainset_3d = ['MuCo']
    trainset_2d = ['MSCOCO']

    # testing set
    # Human36M, MuPoTS, MSCOCO
    testset = 'MuPoTS'

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    pretrain_dir = osp.join(output_dir, 'pre_train')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    
    ## input, output
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//8, input_shape[1]//8)
    width_multiplier = 1.0
    depth_dim = 32
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    embedding_size = 2048
    lr_dec_epoch = [17, 21]
    end_epoch = 25
    lr = 1e-3
    lr_dec_factor = 10
    batch_size = 64

    ## testing config
    test_batch_size = 32
    flip_test = True
    use_gt_info = True

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

