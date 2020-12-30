import os
import onnx
import time
import torch
import argparse
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch.nn.utils.prune as prune

from config import cfg
from torch.nn.parallel.data_parallel import DataParallel
from base import Transformer
from onnx_tf.backend import prepare

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--joint', type=int, dest='joint')
    parser.add_argument('--modelpath', type=str, dest='modelpath')
    parser.add_argument('--backbone', type=str, dest='backbone')
    parser.add_argument('--frontbone', type=str, dest='frontbone')
    parser.add_argument('--quantization',type=str, dest='quantization')
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

dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True, device='cuda')

transformer = Transformer(args.backbone, args.frontbone, args.joint, args.modelpath)
transformer._make_model()

single_pytorch_model = transformer.model

print(args)
# static quantization
if args.quantization == 'True':
    print("=" * 60)
    print("starting quantization")
    print("=" * 60)
    model = torch.quantization.quantize_dynamic(single_pytorch_model, {torch.nn.Conv2d}, dtype=torch.qint8)
    # single_pytorch_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # signle_pytorch_model_fused = torch.quantization.fuse_modules(single_pytorch_model, ['backbone'])
    # single_pytorch_model_prepared = torch.quantization.prepare(single_pytorch_model_fused)
    # model = torch.quantization.convert(single_pytorch_model_prepared)
else:
    print("=" * 60)
    print("loading model")
    print("=" * 60)
    model = single_pytorch_model

ONNX_PATH="../output/{}.onnx".format(args.backbone + args.frontbone + str(args.quantization))

torch.onnx.export(
    model=model,
    args=dummy_input,
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

pytorch_result = model(dummy_input)
pytorch_result = pytorch_result.cpu().detach().numpy()

ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(None, {'input': dummy_input.detach().cpu().numpy()})
outputs = np.array(outputs[0])

print("=" * 60)
print("pytorch_model output {}".format(pytorch_result.shape), pytorch_result)
print("onnx_model ouput size{}".format(outputs.shape), outputs)
print("difference", np.linalg.norm(pytorch_result-outputs))
print("=" * 60)

TF_PATH = "../output/{}".format(args.backbone + args.frontbone + str(args.quantization)) # where the representation of tensorflow model will be stored

# prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)  # creating TensorflowRep object choose either "CUDA:0" or "CPU"

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)

TFLITE_PATH = "../output/{}.tflite".format(args.backbone + args.frontbone + str(args.quantization))

PB_PATH = "../output/{}/saved_model.pb".format(args.backbone + args.frontbone + str(args.quantization))

# make a converter object from the saved tensorflow file
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(PB_PATH, input_arrays=['input'], output_arrays=['output'])
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

# tell converter which type of optimization techniques to use
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

# converter.experimental_new_converter = True

# I had to explicitly state the ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tf_lite_model = converter.convert()
print("=" * 60)
print("converting to tflite")
print("=" * 60)
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)

# Load TFLite model and allocate tensors.
print("=" * 60)
print(tf.__version__)
print("=" * 60)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("input_details : ", input_details)
print("output_details : ", output_details)
print("=" * 60)

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

print("output_shape : ", output_data.shape)


