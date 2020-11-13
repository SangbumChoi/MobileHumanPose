import argparse
from config import cfg
from tqdm import tqdm
import os.path as osp
import numpy as np
import torch
from base import Trainer
from base import Tester
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--backbone', type=str, dest='backbone')
    parser.add_argument('--frontbone', type=str, dest='frontbone')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.fastest = True
    cudnn.benchmark = True

    trainer = Trainer(args.backbone, args.frontbone)
    trainer._make_batch_generator()
    trainer._make_model()

    tester = Tester(args.backbone, args.frontbone)
    tester._make_batch_generator()

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

        preds = []
        tester._make_model(osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch))))

        with torch.no_grad():
            for itr, input_img in enumerate(tqdm(tester.batch_generator)):

                # forward
                coord_out = tester.model(input_img)

                if cfg.flip_test:
                    flipped_input_img = flip(input_img, dims=3)
                    flipped_coord_out = tester.model(flipped_input_img)
                    flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                    for pair in tester.flip_pairs:
                        flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1],
                                                                                             :].clone(), flipped_coord_out[
                                                                                                         :, pair[0],
                                                                                                         :].clone()
                    coord_out = (coord_out + flipped_coord_out) / 2.
                coord_out = coord_out.cpu().numpy()
                preds.append(coord_out)
        # evaluate
        preds = np.concatenate(preds, axis=0)
        result = tester._evaluate(preds, cfg.result_dir)
        trainer.logger.info(''.joint(result))

if __name__ == "__main__":
    main()
