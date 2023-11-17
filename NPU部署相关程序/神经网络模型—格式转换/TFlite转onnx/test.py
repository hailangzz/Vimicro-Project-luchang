import onnx
import onnx.helper as helper
import numpy as np

# 输入尺寸
input_shape = [1, 16, 48, 48]
# 目标输出尺寸
output_shape = [1, 32, 48, 48]

# 创建Pad节点
pad_value = 0  # 可以根据需要更改填充值
pads = [0, 0, 8, 8, 0, 0, 0, 0]  # 根据输入输出尺寸的差异计算填充值
pad_node = helper.make_node('Pad', inputs=['input'], outputs=['output'], mode='constant', pads=pads, value=pad_value)

# 创建输入输出张量信息
input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input_shape)
output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output_shape)

# 创建ONNX图
graph_def = helper.make_graph([pad_node], 'pad_graph', [input_tensor], [output_tensor])
model_def = helper.make_model(graph_def, producer_name='onnx-pad-example')

# 保存ONNX模型
onnx.save(model_def, 'padded_model.onnx')
