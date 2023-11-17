import onnx
import numpy as np
from onnx import helper, numpy_helper

# 创建输入和输出张量信息
input_shape = [1, 3, 192, 192]
output_shape_conv = [1, 16, 48, 48]
output_shape_pad = [1, 32, 48, 48]

# 创建Conv节点
conv_weight = np.random.randn(16, 3, 3, 3).astype(np.float32)
conv_node = helper.make_node('Conv', inputs=['input', 'conv_weight'], outputs=['conv_output'], name='conv', kernel_shape=[3, 3], pads=[1, 1, 1, 1])

# 创建Pad节点
pad_node = helper.make_node('Pad', inputs=['conv_output'], outputs=['output'], mode='constant', pads=[0, 0, 0, 0, 8, 8, 0, 0], value=0)

# 创建图结构
graph_def = helper.make_graph(
    [conv_node, pad_node],
    'pad_graph',
    [helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape),
     helper.make_tensor_value_info('conv_weight', onnx.TensorProto.FLOAT, conv_weight.shape)],
    [helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output_shape_pad)]
)

# 创建模型
model_def = helper.make_model(graph_def, producer_name='onnx-example')

# 保存ONNX模型
onnx.save(model_def, 'pad_model.onnx')
