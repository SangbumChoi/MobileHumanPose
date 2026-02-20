"""
Test MobileHumanPose. Edit config.py for backbone, dataset.
Usage: python test.py [--test_epoch 0] [--epochs 0-1]
"""
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from base import Tester
from config import cfg
from utils.pose_utils import flip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', '--epochs', type=str, default='0')
    args = parser.parse_args()

    # Parse epoch range
    s = args.test_epoch
    if '-' in s:
        a, b = s.split('-')
        epochs = range(int(a), int(b) + 1)
    else:
        epochs = [int(s)]

    cfg.num_gpus = 1
    cudnn.benchmark = True

    tester = Tester()
    tester._make_data()

    for epoch in epochs:
        tester._make_model(epoch)
        preds = []

        with torch.no_grad():
            for img in tester.loader:
                out = tester.model(img)
                if cfg.flip_test:
                    flip_img = flip(img, dims=3)
                    out_f = tester.model(flip_img)
                    out_f[:, :, 0] = cfg.output_shape[1] - out_f[:, :, 0] - 1
                    for (i, j) in tester.flip_pairs:
                        out_f[:, i], out_f[:, j] = out_f[:, j].clone(), out_f[:, i].clone()
                    out = (out + out_f) / 2
                preds.append(out.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tester.testset.evaluate(preds, cfg.result_dir)


if __name__ == '__main__':
    main()
