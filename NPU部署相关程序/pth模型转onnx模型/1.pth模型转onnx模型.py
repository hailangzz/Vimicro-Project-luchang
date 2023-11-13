import torch
import torchvision.models as models
model = models.resnet50()
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model.load_state_dict(torch.load('./resnet50-0676ba61.pth'))
model = model.cuda()
model.eval()
input_names = [ "input" ]
output_names = [ "output" ]
torch.onnx.export(model, dummy_input, "resnet50-0676ba61.onnx", verbose=True,
input_names=input_names, output_names=output_names)