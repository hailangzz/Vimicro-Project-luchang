import tflite2onnx
import onnx
tflite_path = r"C:\Users\zhangzuo\Desktop/face_landmark10.tflite"
onnx_path = r"C:\Users\zhangzuo\Desktop/face_landmark.onnx" #modelname.onnx
tflite2onnx.convert(tflite_path,onnx_path)

onnx_model = onnx.load(r"C:\Users\zhangzuo\Desktop/face_landmark.onnx")
graph = onnx_model.graph
node = graph.node

for i in range(len(node)):
    if node[i].op_type == 'Pad':
        print(node[i],i)

#onnx.helper.make_node
#node.remove()
