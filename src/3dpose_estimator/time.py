import argparse

import torch

from common.base import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--joint', type=int, default=18)
parser.add_argument('--modelpath', type=str, required=True)
parser.add_argument('--gpu', '--backbone', help='Deprecated. Edit config.py')
args = parser.parse_args()

optimal_batch_size = 64
transformer = Transformer(args.joint, args.modelpath)
transformer._make_model()

model = transformer.model

device = torch.device("cuda")

dummy_input = torch.randn(optimal_batch_size, 3, 256, 256, dtype=torch.float).to(device)

repetitions=100
total_time = 0

with torch.no_grad():
    for rep in range(repetitions):
        starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time
Throughput = (repetitions*optimal_batch_size)/total_time
print('Final Throughput:',Throughput)
