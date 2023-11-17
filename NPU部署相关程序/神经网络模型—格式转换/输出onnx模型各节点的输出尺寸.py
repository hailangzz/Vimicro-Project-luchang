import onnxruntime as ort

# 加载模型
model_path = r"C:\Users\zhangzuo\Downloads\facemesh_face_landmark.onnx"  # 替换为你的ONNX模型路径
ort_session = ort.InferenceSession(model_path)

# 获取模型输出信息
outputs_info = ort_session.get_model_outputs_info()

# 打印每一层的输出大小信息
for idx, output_info in enumerate(outputs_info):
    output_name = output_info.name
    output_shape = output_info.shape
    print(f"Layer {idx + 1}: Output '{output_name}' Shape: {output_shape}")
