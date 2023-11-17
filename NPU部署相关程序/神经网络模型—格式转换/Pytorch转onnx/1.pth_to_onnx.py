import torch
from blazeface import BlazeFace, MediaPipeBlazeFace

x = torch.randn(1, 3, 128, 128)
model = BlazeFace()
model_statedict = torch.load(r"E:\BlazeFace-PyTorch-master\blazeface.pth",map_location=lambda storage,loc:storage)   #导入Gpu训练模型，导入为cpu格式
model.load_state_dict(model_statedict)  #将参数放入model_test中
model.eval()  # 测试，看是否报错
#下面开始转模型，cpu格式下
device = torch.device("cpu")
dummy_input = torch.randn(1, 3, 128, 128,device=device)
input_names = ["input"]  # 名字随意
output_names = ["box_out","class_out"] # 名字随意

torch.onnx.export(model,
                  dummy_input,
                  "model.onnx",  #  输出的onnx路径
                  verbose=False,opset_version=9,
                  input_names=input_names,
                  output_names=output_names)

def onnx_model_simplify(origin_onnx_model_path="blazeface.onnx"):
    import onnx
    from onnxsim import simplify
    model = onnx.load(origin_onnx_model_path)
    simplify_model,check = simplify(model)
    onnx.save(simplify_model,"blazeface_simplify.onnx")

onnx_model_simplify(origin_onnx_model_path="model.onnx")