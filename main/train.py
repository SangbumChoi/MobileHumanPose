import argparse
from config import cfg
from tqdm import tqdm
import os.path as osp
import numpy as np
import torch
from base import Trainer
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn


def main():
    
    # argument parse and create log
    cudnn.fastest = True
    cudnn.benchmark = True

    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (input_img, joint_img, joint_vis, joints_have_depth) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            target = {'coord': joint_img, 'vis': joint_vis, 'have_depth': joints_have_depth}
            loss_coord = trainer.model(input_img, target)
            loss_coord = loss_coord.mean()

            # backward
            loss = loss_coord
            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_coord', loss_coord.detach()),
                ]
            trainer.logger.info(' '.join(screen))
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

if __name__ == "__main__":
    main()
