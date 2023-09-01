import os
import numpy as np
from PIL import Image
from paddle.vision.transforms import Compose, Resize, CenterCrop, Normalize
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers

# ģ��·��
model_fp32 = 'models/yolop-640-640.onnx'
model_quant_static = 'models/yolop-640-640_quant_static.onnx'

# ����Ԥ����
'''
    ���� -> ���Ĳ��� -> ����ת�� -> ת�� -> ��һ�� -> ���ά��
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

# ����У׼��ͼ������
'''
    ��ȡͼ�� -> Ԥ���� -> ��������ֵ�
'''
img_dir = 'F:\AiTotalDatabase\ADAS_test_images\origin_images'
img_num = 100
datas = [
    val_transforms(
        Image.open(os.path.join(img_dir, img)).convert('RGB')
    ) for img in os.listdir(img_dir)[:img_num]
]


# �������ζ�ȡ��
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


# ����У׼���ݶ�ȡ��
'''
    ʵ����һ��������
    get_next ��������һ��������ʽ���ֵ�
    {
        ���� 1: ���� 1, 
        ...
        ���� n: ���� n
    }
    ��¼��ģ�͵ĸ�����������Ӧ�ľ���Ԥ����������
'''


class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size):
        self.datas = batch_reader(datas, batch_size)

    def get_next(self):
        return next(self.datas, None)


# ʵ����һ��У׼���ݶ�ȡ��
data_reader = DataReader(datas, 1)

# ��̬����
quantize_static(
    model_input=model_fp32,  # ����ģ��
    model_output=model_quant_static,  # ���ģ��
    calibration_data_reader=data_reader,  # У׼���ݶ�ȡ��
    quant_format=QuantFormat.QDQ,  # ������ʽ QDQ / QOperator
    activation_type=QuantType.QInt8,  # �������� Int8 / UInt8
    weight_type=QuantType.QInt8,  # �������� Int8 / UInt8
    calibrate_method=CalibrationMethod.MinMax,  # ����У׼���� MinMax / Entropy / Percentile
    optimize_model=False  # �Ƿ��Ż�ģ��
        )