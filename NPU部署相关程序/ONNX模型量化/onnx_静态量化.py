import os
import numpy as np
from PIL import Image
from paddle.vision.transforms import Compose, Resize, CenterCrop, Normalize
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers

# 模型路径
model_fp32 = 'models/yolop-640-640.onnx'
model_quant_static = 'models/yolop-640-640_quant_static.onnx'

# 数据预处理
'''
    缩放 -> 中心裁切 -> 类型转换 -> 转置 -> 归一化 -> 添加维度
'''
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transforms = Compose(
    [
        Resize(640, interpolation="bilinear"),
        CenterCrop(640),
        lambda x: np.asarray(x, dtype='float32').transpose(2, 0, 1) / 255.0,
        Normalize(mean, std),
        lambda x: x[None, ...]
    ]
)

# 用于校准的图像数据
'''
    读取图像 -> 预处理 -> 组成数据字典
'''
img_dir = 'F:\AiTotalDatabase\ADAS_test_images\origin_images'
img_num = 100
datas = [
    val_transforms(
        Image.open(os.path.join(img_dir, img)).convert('RGB')
    ) for img in os.listdir(img_dir)[:img_num]
]


# 数据批次读取器
def batch_reader(datas, batch_size):
    _datas = []
    length = len(datas)
    for i, data in enumerate(datas):
        if batch_size == 1:
            yield {'inputs': data}
        elif (i + 1) % batch_size == 0:
            _datas.append(data)
            yield {'inputs': np.concatenate(_datas, 0)}
            _datas = []
        elif i < length - 1:
            _datas.append(data)
        else:
            _datas.append(data)
            yield {'inputs': np.concatenate(_datas, 0)}


# 构建校准数据读取器
'''
    实质是一个迭代器
    get_next 方法返回一个如下样式的字典
    {
        输入 1: 数据 1, 
        ...
        输入 n: 数据 n
    }
    记录了模型的各个输入和其对应的经过预处理后的数据
'''


class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size):
        self.datas = batch_reader(datas, batch_size)

    def get_next(self):
        return next(self.datas, None)


# 实例化一个校准数据读取器
data_reader = DataReader(datas, 1)

# 静态量化
quantize_static(
    model_input=model_fp32,  # 输入模型
    model_output=model_quant_static,  # 输出模型
    calibration_data_reader=data_reader,  # 校准数据读取器
    quant_format=QuantFormat.QDQ,  # 量化格式 QDQ / QOperator
    activation_type=QuantType.QInt8,  # 激活类型 Int8 / UInt8
    weight_type=QuantType.QInt8,  # 参数类型 Int8 / UInt8
    calibrate_method=CalibrationMethod.MinMax,  # 数据校准方法 MinMax / Entropy / Percentile
    optimize_model=False  # 是否优化模型
        )