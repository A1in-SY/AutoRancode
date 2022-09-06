import torch
import tensorflow
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
from model_resnet import mymodel_resnet

use_model = 'model_resnet_100'

input_pth = './{}.pth'.format(use_model)
output_onnx = './{}.onnx'.format(use_model)
output_pb = './{}.pb'.format(use_model)

# Load the trained model from file
trained_model = mymodel_resnet()
trained_model.load_state_dict(torch.load(input_pth))
print('Model loading complete.')

# Export the trained model to ONNX
# one black and white 50 x 100 picture will be the input to the model
dummy_input = Variable(torch.randn(1, 1, 50, 100))
torch.onnx.export(trained_model, dummy_input, output_onnx)

# Load the ONNX file
model = onnx.load(output_onnx)

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)
tf_rep.export_graph(output_pb)
