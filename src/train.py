"""
Train MobileHumanPose. Edit config.py for backbone, dataset.
Single GPU:  python train.py
Multi-GPU:   bash runs/train.sh  (uses torchrun)
"""
import argparse
import os

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from common.base import Trainer, save_ckpt
from src.config import cfg


def setup_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        cfg.num_gpus = dist.get_world_size()
        return True
    cfg.num_gpus = 1
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont', action='store_true', help='resume from latest ckpt')
    args = parser.parse_args()
    if args.cont:
        cfg.continue_train = True

    setup_dist()
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_data()
    trainer._make_model()

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        if hasattr(trainer.loader, 'sampler') and hasattr(trainer.loader.sampler, 'set_epoch'):
            trainer.loader.sampler.set_epoch(epoch)
        lr = trainer._lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (img, coord, vis, have_depth) in enumerate(trainer.loader):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()
            target = {'coord': coord, 'vis': vis, 'have_depth': have_depth}
            loss = trainer.model(img, target).mean()

            loss.backward()
            trainer.optimizer.step()

            trainer.gpu_timer.toc()
            trainer.logger.info(
                f'Epoch {epoch}/{cfg.end_epoch} itr {itr}/{trainer.itr_per_epoch} '
                f'lr={lr:.2e} speed={trainer.tot_timer.average_time:.2f}s/itr '
                f'loss={loss.item():.4f}'
            )
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        path = save_ckpt(trainer.model, trainer.optimizer, epoch)
        trainer.logger.info(f'Saved {path}')


if __name__ == '__main__':
    main()
