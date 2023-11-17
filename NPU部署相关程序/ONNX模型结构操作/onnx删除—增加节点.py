import onnx
from onnx import helper
import onnxsim

# 加载ONNX模型
model_path = r"C:\Users\zhangzuo\Downloads\facemesh_face_landmark_simple.onnx"
onnx_model = onnx.load(model_path)

# 查找节点类型
delect_node_name_dict={}
graph = onnx_model.graph
node = graph.node
for i in range(len(node)):
    if node[i].op_type == 'Pad':
        print(i,node[i])
        delect_node_name_dict[i] = node[i]


for node_inde,node in delect_node_name_dict.items():

    if node_inde==13:
        onnx_model.graph.node.remove(node)
        # node1 = onnx.helper.make_node(
        #     'Transpose',
        #     name='Transpose1',
        #     inputs=['max_pooling2d_1'],
        #     outputs=['Transpose1'],
        #     perm=[0,2,3,1],
        # )
        # onnx_model.graph.node.insert(12, node1)

        pad_value = 0  # 可以根据需要更改填充值
        pads = [0, 0, 8, 8, 0, 0, 0, 0]  # 根据输入输出尺寸的差异计算填充值

        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['max_pooling2d'],
            outputs=['channel_padding'],
            mode='constant',
            pads=pads,
            value=pad_value,
        )
        onnx_model.graph.node.insert(13, node2)
# # #
#         node3 = onnx.helper.make_node(
#             op_type='Transpose',
#             name='channel_padding_1',
#             inputs=['Pad_1'],
#             outputs=['channel_padding_1'],
#             perm=[0, 3, 1, 2],
#         )
#         onnx_model.graph.node.insert(12, node3)
#
    if node_inde==27:
        onnx_model.graph.node.remove(node)
    #     node1 = onnx.helper.make_node(
    #         op_type='Transpose',
    #         name='channel_padding_2',
    #         inputs=['max_pooling2d_2'],
    #         outputs=['channel_padding_2'],
    #         perm=[0,2,3,1],
    #     )
    #     onnx_model.graph.node.insert(26, node1)

        pad_value = 0  # 可以根据需要更改填充值
        pads = [0, 0, 16, 16, 0, 0, 0, 0]  # 根据输入输出尺寸的差异计算填充值

        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['max_pooling2d_1'],
            outputs=['channel_padding_1'],
            mode='constant',
            pads=pads,
            name='channel_padding_1',
            value=pad_value

        )
        onnx_model.graph.node.insert(27, node2)
#
#         node3 = onnx.helper.make_node(
#             op_type='Transpose',
#             name='channel_padding_2',
#             inputs=['to_tran2'],
#             outputs=['channel_padding_2'],
#             perm=[0, 3, 1, 2],
#         )
#         onnx_model.graph.node.insert(27, node3)
#
    if node_inde==41:
        onnx_model.graph.node.remove(node)
    #     node1 = onnx.helper.make_node(
    #         op_type='Transpose',
    #         name='channel_padding_3',
    #         inputs=['max_pooling2d_3'],
    #         outputs=['channel_padding_3'],
    #         perm=[0,2,3,1],
    #     )
    #     onnx_model.graph.node.insert(40, node1)

        pad_value = 0  # 可以根据需要更改填充值
        pads = [0, 0, 32, 32, 0, 0, 0, 0]  # 根据输入输出尺寸的差异计算填充值
        node2 = onnx.helper.make_node(
            op_type='Pad',
            inputs=['max_pooling2d_2'],
            outputs=['channel_padding_2'],
            mode='constant',
            pads=pads,
            name='channel_padding_2',
            value=pad_value
        )
        onnx_model.graph.node.insert(41, node2)
#
#         node3 = onnx.helper.make_node(
#             op_type='Transpose',
#             name='channel_padding_3',
#             inputs=['to_tran3'],
#             outputs=['channel_padding_3'],
#             perm=[0, 3, 1, 2],
#         )
#         onnx_model.graph.node.insert(41, node3)
#
# #      onnx_model.graph.node.remove(node)
# #
# # # 如果需要，更新输入和输出
# # # ...
# # # 保存修改后的模型
output_model_path = r"C:\Users\zhangzuo\Desktop/modified_model.onnx"

# simplified_model, check = onnxsim.simplify(onnx_model)
# print("onnx model simplify Ok!")
# onnx.save(simplified_model, output_model_path)

onnx.save(onnx_model, output_model_path)
onnx.checker.check_model(onnx_model)