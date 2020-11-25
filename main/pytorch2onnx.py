import onnx
import torch
import argparse
import numpy
import onnxruntime as ort
import tensorflow as tf
import torch.nn.utils.prune as prune

from torchsummary import summary
from torch.nn.parallel.data_parallel import DataParallel
from model import get_pose_net
from onnx_tf.backend import prepare
from config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--joint', type=int, dest='joint')
    parser.add_argument('--modelpath', type=str, dest='modelpath')
    parser.add_argument('--backbone', type=str, dest='backbone')
    parser.add_argument('--frontbone', type=str, dest='frontbone')
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

dummy_input = torch.randn(1, 3, 256, 256, device='cuda')

pytorch_model = get_pose_net(args.backbone, args.frontbone, False, args.joint)
pytorch_model = DataParallel(pytorch_model).cuda()

#Load the pretrained model
pytorch_model.load_state_dict(torch.load(args.modelpath)['network'])

#Time to transfer weights
single_pytorch_model = pytorch_model.module
single_pytorch_model.eval()
summary(single_pytorch_model, (3, 256, 256))

ONNX_PATH="../output/baseline.onnx"

torch.onnx.export(
    model=single_pytorch_model,
    args=dummy_input,
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output']
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

pytorch_result = single_pytorch_model(dummy_input)
pytorch_result = pytorch_result.cpu().detach().numpy()
print("pytorch_model output {}".format(pytorch_result.shape), pytorch_result)

ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
outputs = numpy.array(outputs[0])
print("onnx_model ouput size{}".format(outputs.shape), outputs)

print("difference", numpy.linalg.norm(pytorch_result-outputs))

TF_PATH = "../output/baseline" # where the representation of tensorflow model will be stored

# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)  # creating TensorflowRep object

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)

TFLITE_PATH = "../output/baseline.tflite"

PB_PATH = "../output/baseline/saved_model.pb"

# make a converter object from the saved tensorflow file
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PB_PATH, input_arrays=['input'], output_arrays=['output'])
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
