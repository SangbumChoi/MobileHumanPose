import argparse

import coremltools as ct
import torch
from base import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--joint', type=int, default=18)
parser.add_argument('--modelpath', type=str, required=True)
parser.add_argument('--gpu', '--backbone', help='Deprecated. Edit config.py')
args = parser.parse_args()

transformer = Transformer(args.joint, args.modelpath)
transformer._make_model()

single_pytorch_model = transformer.model

device = torch.device('cpu')
single_pytorch_model.to(device)

dummy_input = torch.randn(1, 3, 256, 256)

traced_model = torch.jit.trace(single_pytorch_model, dummy_input)

# Convert to Core ML using the Unified Conversion API
model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_1", shape=dummy_input.shape)], #name "input_1" is used in 'quickstart'
)

model.save("test.mlmodel")
