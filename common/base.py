"""
Base Trainer/Tester. Minimal, hackable. Use config.py for backbone, dataset.
"""
import glob
import os
import os.path as osp

import torch
import torchvision.transforms as transforms
from config import cfg
from dataset import DatasetLoader
from logger import colorlogger
from model import get_pose_net
from multiple_datasets import MultipleDatasets
from timer import Timer
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


# Dataset registry
def get_ds(name, split):
    from utils.dir_utils import add_pypath
    add_pypath(osp.join(cfg.data_dir, name))
    mod = __import__(name, fromlist=[name])
    return getattr(mod, name)(split)


def save_ckpt(model, optimizer, epoch):
    path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar')
    unwrap = model.module if hasattr(model, 'module') else model
    torch.save({'epoch': epoch, 'network': unwrap.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    return path


def load_ckpt(model_dir):
    files = glob.glob(osp.join(model_dir, '*.pth.tar'))
    if not files:
        return None, 0
    latest = max(files, key=lambda f: int(f.split('snapshot_')[-1].split('.')[0]))
    ckpt = torch.load(latest, map_location='cpu')
    return ckpt, ckpt['epoch'] + 1


class Trainer:
    def __init__(self):
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        self.logger = colorlogger(cfg.log_dir, 'train_logs.txt')
        self.backbone = cfg.backbone

    def _make_data(self):
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        loaders_3d = []
        ref = None
        for name in cfg.trainset_3d:
            ds = get_ds(name, 'train')
            loaders_3d.append(DatasetLoader(ds, ref, True, t))
            ref = ds.joints_name if ref is None else ref

        loaders_2d = [DatasetLoader(get_ds(name, 'train'), ref, True, t) for name in cfg.trainset_2d]

        self.joint_num = loaders_3d[0].joint_num
        d3 = MultipleDatasets(loaders_3d, make_same_len=False)
        combined = MultipleDatasets([d3, MultipleDatasets(loaders_2d, make_same_len=False)], make_same_len=True) if loaders_2d else MultipleDatasets([d3], make_same_len=True)
        batch = cfg.batch_size * cfg.num_gpus if not torch.distributed.is_initialized() else cfg.batch_size
        sampler = torch.utils.data.distributed.DistributedSampler(combined, shuffle=True) if torch.distributed.is_initialized() else None
        self.loader = DataLoader(combined, batch_size=batch, shuffle=(sampler is None), sampler=sampler,
                                 num_workers=cfg.num_thread, pin_memory=torch.cuda.is_available())
        self.itr_per_epoch = len(combined) // batch

    def _make_model(self):
        model = get_pose_net(cfg.backbone, True, self.joint_num)
        model = model.cuda() if torch.cuda.is_available() else model
        if cfg.num_gpus > 1 and torch.distributed.is_initialized():
            model = DDP(model)
        elif cfg.num_gpus > 1 or (torch.cuda.is_available() and not torch.distributed.is_initialized()):
            model = DataParallel(model).cuda() if torch.cuda.is_available() else model

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        start = 0
        if cfg.continue_train:
            ckpt, start = load_ckpt(cfg.model_dir)
            if ckpt:
                state = ckpt['network']
                if next(iter(state.keys())).startswith('module.'):
                    state = {k.replace('module.', ''): v for k, v in state.items()}
                model.module.load_state_dict(state) if hasattr(model, 'module') else model.load_state_dict(state)
                opt.load_state_dict(ckpt['optimizer'])

        self.model = model
        self.optimizer = opt
        self.start_epoch = start

    def _lr(self, epoch):
        idx = 0
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
            idx += 1
        lr = cfg.lr / (cfg.lr_dec_factor ** idx)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


class Tester:
    def __init__(self):
        self.backbone = cfg.backbone
        self.logger = colorlogger(cfg.log_dir, 'test_logs.txt')

    def _make_data(self):
        ds = get_ds(cfg.testset, 'test')
        self.testset = ds
        loader = DatasetLoader(ds, None, False, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
        ]))
        batch = cfg.test_batch_size * cfg.num_gpus
        self.loader = DataLoader(loader, batch_size=batch, shuffle=False,
                                 num_workers=cfg.num_thread, pin_memory=torch.cuda.is_available())
        self.joint_num = loader.joint_num
        self.skeleton = loader.skeleton
        self.flip_pairs = ds.flip_pairs

    def _make_model(self, epoch):
        path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar')
        assert os.path.exists(path), f'No checkpoint at {path}'
        model = get_pose_net(cfg.backbone, False, self.joint_num)
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['network']
        if next(iter(state.keys())).startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        if torch.cuda.is_available():
            model = DataParallel(model).cuda()
        model.eval()
        self.model = model


class Transformer:
    """For ONNX/CoreML export."""
    def __init__(self, joint_num, model_path):
        self.backbone = cfg.backbone
        self.joint_num = joint_num
        self.model_path = model_path

    def _make_model(self):
        model = get_pose_net(cfg.backbone, False, self.joint_num)
        ckpt = torch.load(self.model_path, map_location='cpu')
        state = ckpt['network']
        if next(iter(state.keys())).startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()
        self.model = model.module if hasattr(model, 'module') else model
