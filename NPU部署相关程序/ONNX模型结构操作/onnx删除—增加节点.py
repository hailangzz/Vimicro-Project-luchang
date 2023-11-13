import onnx
from onnx import helper

# 加载ONNX模型
model_path = r"C:\Users\zhangzuo\Desktop/face_landmark1.onnx"
onnx_model = onnx.load(model_path)

# 查找节点类型
delect_node_name_dict={}
graph = onnx_model.graph
node = graph.node
for i in range(len(node)):
    if node[i].op_type == 'Pad':
        print(node[i],i)
        delect_node_name_dict[i] = node[i]


for node_inde,node in delect_node_name_dict.items():
    onnx_model.graph.node.remove(node)
    if node_inde==13:
        node1 = onnx.helper.make_node(
            op_type='Transpose',
            name='to_Transpose1',
            inputs=['max_pooling2d_1'],
            outputs=['to_Transpose1'],
            perm=[0,2,3,1],
        )
        onnx_model.graph.node.insert(11, node1)

        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['to_Transpose1'],
            outputs=['to_Pad1'],
            mode='constant',
            pads=[0, 0, 0, 0, 0, 0, 0, 16],
            name='to_Pad1',
        )
        onnx_model.graph.node.insert(12, node2)

        node3 = onnx.helper.make_node(
            op_type='Transpose',
            name='channel_padding_1',
            inputs=['to_Pad1'],
            outputs=['channel_padding_1'],
            perm=[0, 3, 1, 2],
        )
        onnx_model.graph.node.insert(13, node3)

    if node_inde==27:
        node1 = onnx.helper.make_node(
            op_type='Transpose',
            name='to_pad2',
            inputs=['max_pooling2d_2'],
            outputs=['to_pad2'],
            perm=[0,2,3,1],
        )
        onnx_model.graph.node.insert(25, node1)

        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['to_pad2'],
            outputs=['to_tran2'],
            mode='constant',
            pads=[0, 0, 0, 0, 0, 0, 0, 32],
            name='to_tran2',

        )
        onnx_model.graph.node.insert(26, node2)

        node3 = onnx.helper.make_node(
            op_type='Transpose',
            name='channel_padding_2',
            inputs=['to_tran2'],
            outputs=['channel_padding_2'],
            perm=[0, 3, 1, 2],
        )
        onnx_model.graph.node.insert(27, node3)

    if node_inde==41:
        node1 = onnx.helper.make_node(
            op_type='Transpose',
            name='to_pad3',
            inputs=['max_pooling2d_3'],
            outputs=['to_pad3'],
            perm=[0,2,3,1],
        )
        onnx_model.graph.node.insert(39, node1)

        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['to_pad3'],
            outputs=['to_tran3'],
            mode='constant',
            pads=[0, 0, 0, 0, 0, 0, 0, 64],
            name='to_tran3',
        )
        onnx_model.graph.node.insert(40, node2)

        node3 = onnx.helper.make_node(
            op_type='Transpose',
            name='channel_padding_3',
            inputs=['to_tran3'],
            outputs=['channel_padding_3'],
            perm=[0, 3, 1, 2],
        )
        onnx_model.graph.node.insert(41, node3)

#      onnx_model.graph.node.remove(node)
#
# # 如果需要，更新输入和输出
# # ...
# # 保存修改后的模型
output_model_path = r"C:\Users\zhangzuo\Desktop/modified_model.onnx"
onnx.save(onnx_model, output_model_path)
onnx.checker.check_model(onnx_model)