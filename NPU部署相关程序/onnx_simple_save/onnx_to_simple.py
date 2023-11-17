import onnx
import onnxsim

model = onnx.load(r"C:\Users\zhangzuo\Downloads\face_landmark4.onnx")
onnx.checker.check_model(model)

print("====> Simplifying...")
# model_opt = onnxsim.simplify(args.onnx_model)
simplified_model, check = onnxsim.simplify(model)
# print("model_opt", model_opt)
onnx.save(simplified_model, r"C:\Users\zhangzuo\Downloads\facemesh_face_landmark_simple.onnx")
print("onnx model simplify Ok!")