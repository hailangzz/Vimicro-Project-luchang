import argparse
from pathlib import Path
import os,sys
import yaml
from models.yolo import Model
import torch
from torchsummary import summary
from models.common import Bottleneck
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'E:\Downloads\yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / './data/my_test.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path') #模型超参数信息
    parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    parser.add_argument('--nc', type=int, default=2, help='the detection datasets number class!!')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def get_hyp_info(hyp):# 获取模型超参数信息！
    # Hyperparameters


    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict #读取超参文件配置信息；hyp(str)->hyp(dict)

    return  hyp.copy()  # for saving hyps to checkpoints


opt = parse_opt()
cfg = opt.cfg
nc = opt.nc
hyp = get_hyp_info(opt.hyp)
device = opt.device
model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
# 输出网络模型结构
print(model)
# 打印网络各层的输入、输出维度信息
summary(model, (3, 576, 1024))  # 输入是（N,C,H,W）

#打印权重和偏执系数及batchnorm（）中的相关参数
print(model.state_dict().keys())
for param_tensor in model.state_dict():
    #打印 key value字典
    print(param_tensor,'\t',model.state_dict()[param_tensor].size(),model.state_dict()[param_tensor])

# 打印的是Weights和Bais参数的值
for param in model.parameters():
    print(param.size())

#迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
#并且可以更改参数的可训练属性
for name, parameters in model.named_parameters():
    print(name, ';', parameters.size(),parameters.requires_grad)

origin_bn_list = []
ignore_bn_list = []
batch_norm_list = []
for k, m in model.named_modules():
    if isinstance(m, Bottleneck):
        print(m.add)
        print(k, "\t--> model info\t-->", m)
        if m.add:
            origin_bn_list.append(k)

            ignore_bn_list.append(k.rsplit(".", 2)[0] + ".cv1.bn")
            ignore_bn_list.append(k + '.cv1.bn')
            ignore_bn_list.append(k + '.cv2.bn')
    if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
        if int(k.split('.')[1])>6:
            batch_norm_list.append(k)

print(origin_bn_list)
print(ignore_bn_list)
print(batch_norm_list)