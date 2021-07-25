import torch
import argparse
import coremltools as ct


from config import cfg
from torch.nn.parallel.data_parallel import DataParallel
from base import Transformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--joint', type=int, dest='joint')
    parser.add_argument('--modelpath', type=str, dest='modelpath')
    parser.add_argument('--backbone', type=str, dest='backbone')
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

args = parse_args()

# modelpath as definite path
transformer = Transformer(args.backbone, args.joint, args.modelpath)
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
