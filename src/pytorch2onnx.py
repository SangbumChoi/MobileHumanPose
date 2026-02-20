import argparse

import imageio
import numpy
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
from onnx_tf.backend import prepare
from torchsummary import summary

from common.base import Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--joint', type=int, default=18)
parser.add_argument('--modelpath', type=str, required=True)
parser.add_argument('--gpu', '--backbone', help='Deprecated. Edit config.py')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dummy_input = torch.randn(1, 3, 256, 256, device=device)

transformer = Transformer(args.joint, args.modelpath)
transformer._make_model()

single_pytorch_model = transformer.model

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
    output_names=['output'],
    opset_version=11
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

# converter.experimental_new_converter = True
#
# # I had to explicitly state the ops
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]

def representative_dataset():

    dataset_size = 10

    for i in range(dataset_size):
        print(i)
        data = imageio.imread("../sample_images/" + "00000" + str(i) + ".jpg")
        data = numpy.resize(data, [1, 3, 256, 256])
        yield [data.astype(numpy.float32)]


converter.experimental_new_converter = True
converter.experimental_new_quantizer = True

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# input_arrays = converter.get_input_arrays()
# converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}

tf_lite_model = converter.convert()
# Save the model.
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
