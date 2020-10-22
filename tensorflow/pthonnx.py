import torch
import torchvision
import argparse
import os
import onnx
import numpy as np

from config import cfg
from torch.nn.parallel.data_parallel import DataParallel
from model import get_pose_net
from onnx_tf.backend import prepare

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, dest='model')
    parser.add_argument('--back_bone', type=str, dest='backbone')
    parser.add_argument('--front_bone', type=str, dest='frontbone')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    args = parse_args()
    model_path = os.path.join(args.model)
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path

    model = get_pose_net(args.backbone, args.frontbone, True, args.joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])

    filename_onnx = 'output.onnx'
    filename_tf = 'output.pb'

    torch.onnx.export(model, torch.ones(1, 3, cfg.input_shape[0], cfg.input_shape[1]), filename_onnx, input_names=['main_input'], output_names=['main_output'])

    onnx_model = onnx.load("input.onnx")  # load onnx model
    tf_rep = prepare(onnx_model)  # import the onnx model to tf

    print(tf_rep.inputs)
    print('-----------')
    print(tf_rep.outputs)
    print('-----------')
    print(tf_rep.tensor_dict)

    # inference
    output = tf_rep.run(np.random.randn(1, 3, cfg.input_shape[0], cfg.input_shape[1]))._0
    # output.shape == (1, 1000)

    tf_rep.export_graph(filename_tf)

if __name__ == "__main__":
    main()