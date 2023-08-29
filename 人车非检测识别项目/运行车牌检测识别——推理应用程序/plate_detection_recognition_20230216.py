"""
    ClassySORT
    
    YOLO v5(image segmentation) + vanilla SORT(multi-object tracker) implementation 
    that is aware of the tracked object category.
    
    This is for people who want a real-time multiple object tracker (MOT) 
    that can track any kind of object with no additional training.
    
    If you only need to track people, then I recommend YOLOv5 + DeepSORT implementations.
    DeepSORT adds a separately trained neural network on top of SORT, 
    which increases accuracy for human detections but decreases performance slightly.
    
    
    Copyright (C) 2020-2021 Jason Sohn tensorturtle@gmail.com
    
    
    === start GNU License ===
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    === end GNU License ===
"""

# python interpreter searchs these subdirectories for modules
import sys

import argparse
import os
import platform
import time
from pathlib import Path
import cv2
import torch
import torchvision
import numpy as np
import copy
import onnxruntime

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

total_num = 0
obj_num = 0


def get_character():
    dict_character = []
    with open('./plate_char_label.txt', "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            dict_character += list(line)
    dict_char = {}
    for i, char in enumerate(dict_character):
        dict_char[char] = i + 1
    character = ['[blank]'] + dict_character + [' ']
    return character


def decode(preds, raw=False):
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)
    result_list = []
    for word, prob in zip(preds_idx, preds_prob):
        if raw:
            character = get_character()
            result_list.append((''.join([character[int(i)] for i in word]), prob))
        else:
            result = []
            conf = []
            for i, index in enumerate(word):
                if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                    character = get_character()
                    result.append(character[int(index)])
                    conf.append(prob[i])
            result_list.append((''.join(result), conf))
    return result_list


def normalize_img(image):
    image_deal = (image.astype(np.float32) / 255 - 0.5) / 0.5
    return image_deal


def resize(image):
    resize_ratio_height = 48 / image.shape[0]
    width_vir = resize_ratio_height * image.shape[1]
    if width_vir < 120:
        img = cv2.resize(image, (0, 0), fx=resize_ratio_height, fy=resize_ratio_height,
                         interpolation=cv2.INTER_LINEAR)
        gen_image = np.zeros((48, 120, 3), dtype=np.uint8)
        off_x = (gen_image.shape[1] - img.shape[1]) // 2
        gen_image[:, off_x:off_x + img.shape[1], :] = img
    elif width_vir >= 120:
        gen_image = cv2.resize(image, (120, 48), interpolation=cv2.INTER_LINEAR)
    return gen_image


def plate_rec_inference_onnx(img_numpyTensor):
    txts = []
    img_numpyTensor = cv2.cvtColor(img_numpyTensor, cv2.COLOR_BGR2RGB)
    img_resize_numpyTensor = resize(img_numpyTensor)
    img_Normalize_numpyTensor = normalize_img(img_resize_numpyTensor)
    img_Normalize_numpyTensor_1 = img_Normalize_numpyTensor.transpose([2, 0, 1])
    img_torchTensor = torch.from_numpy(img_Normalize_numpyTensor_1).float()
    img_batch_torchTensor = img_torchTensor.unsqueeze(dim=0)
    img_batch_numpyTensor = img_batch_torchTensor.numpy()
    out = plate_rec_session.run(None, {"inputOP": img_batch_numpyTensor})
    out_1 = torch.from_numpy(out[0]).float()
    out_2 = torch.squeeze(out_1, dim=0)
    out_3 = out_2.permute(1, 2, 0)
    out_4 = out_3.softmax(dim=2)
    out_5 = out_4.cpu().numpy()
    txts.extend(decode(out_5))
    rec_result = txts[0][0]

    # if (txts[0][0] == label_name):
    #     correct_num = correct_num + 1
    return rec_result


class YOLOV5_ONNX(object):
    def __init__(self, onnx_path):
        '''initialize onnx'''
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        '''get input node name'''
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        '''get output node name'''
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, image_tensor):
        '''get input tensor'''
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_tensor

        return input_feed

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)

        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def nms(self, prediction, conf_thres=0.4, iou_thres=0.45, agnostic=False):
        if prediction.dtype is torch.float16:
            prediction = prediction.float()  # to FP32
        xc = prediction[..., 4] > conf_thres  # candidates
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])

            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]

        return output

    def clip_coords(self, boxes, img_shape):
        '''check if it is out of bounds'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        # Rescale coords (xyxy) from img1_shape to img0_shape
        '''

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                    img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def infer(self, img_path, save_path):
        '''model forward inference'''
        anchors = len(anchor_list[0]) // 2
        anchor = np.array(anchor_list).astype(np.float32).reshape(3, -1, 2)

        image_height, image_width = img_size
        area = image_height * image_width
        size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
        feature = [[int(j / stride[i]) for j in img_size] for i in range(3)]

        # read image
        src_img = cv2.imread(img_path)
        src_size = src_img.shape[:2]

        # Resize and pad image while meeting stride-multiple constraints
        img = self.letterbox(src_img, img_size, stride=32)[0]
        vehicle_img_copy = copy.copy(img)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = img.astype(dtype=np.float32)
        img /= 255.0

        # dimensions expand
        img = np.expand_dims(img, axis=0)

        # forward inference
        start = time.time()
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)

        for i in range(3):
            bs, _, ny, nx = pred[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            pred[i] = pred[i].reshape(bs, anchors, 5 + class_num, ny, nx).transpose(0, 1, 3, 4, 2)

        # extract features
        y = []
        y.append(torch.tensor(pred[0].reshape(-1, size[0] * 3, 5 + class_num)))
        y.append(torch.tensor(pred[1].reshape(-1, size[1] * 3, 5 + class_num)))
        y.append(torch.tensor(pred[2].reshape(-1, size[2] * 3, 5 + class_num)))

        grid = []
        for k, f in enumerate(feature):
            grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

        z = []
        for i in range(3):
            src = y[i]

            xy = src[..., 0:2] * 2. - 0.5
            wh = (src[..., 2:4] * 2) ** 2
            dst_xy = []
            dst_wh = []
            for j in range(3):
                dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + torch.tensor(grid[i])) * stride[i])
                dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
            src[..., 0:2] = torch.from_numpy(np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1))
            src[..., 2:4] = torch.from_numpy(np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1))
            z.append(src.view(1, -1, 5 + class_num))

        results = torch.cat(z, 1)
        results = self.nms(results, conf_thres, iou_thres)
        cast = time.time() - start
        print("cast time:{}".format(cast))

        # mapping original image
        img_shape = img.shape[2:]
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = det[:, :4]
                # det[:, :4] = self.scale_coords(img_shape, det[:, :4], src_size).round()

        # save_path = img_path.replace('images', 'result')

        # Write results
        plate_num = 0
        plate_name=''
        if det is not None:
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                # label = f'{names[c]} {conf:.2f}'
                if c == 1:
                    plate_x1, plate_y1, plate_x2, plate_y2 = [int(item) for item in xyxy]
                    piece_im = vehicle_img_copy[plate_y1:plate_y2, plate_x1:plate_x2, :]
                    if piece_im.size:
                        plate_infer = plate_rec_inference_onnx(piece_im)
                        
                        # save_piece_path = os.path.join(dst_dir, f'{total_num}_{names[cat]}_{obj_num}_{car_id}.jpg')
                        plate_name += plate_infer
                        plate_num += 1
                    # if (plate_y1 + plate_y2) / 2 > image_height / 2:
                        # piece_im = src_img[plate_y1:plate_y2, plate_x1:plate_x2, :]
                        
        name, ext = os.path.splitext(save_path)                
        res_file_path = f'_plate_{plate_num}_{plate_name}'.join([name, ext])
        cv2.imwrite(res_file_path, src_img)

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0]] -= pad[0]  # x padding
    coords[:, [1]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1

    return coords


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


if __name__ == '__main__':
    """
    usage:
    nohup python track_strategy_plate_detection_recognition.py --source ./test_videos_1/fact_20181019141901_20181019142439.avi >> logging_10.9.2.txt 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows-plate-marker-person-weight', type=str, default=r'D:\迅雷下载\AI数据集汇总\人车非数据及项目\模型代码及数据\模型代码及数据\plate_detection_recognition\weights\\renchefeis2.onnx', help='model weight')
    parser.add_argument('--plate-rec-weight', type=str, default=r'D:\迅雷下载\AI数据集汇总\人车非数据及项目\模型代码及数据\模型代码及数据\plate_detection_recognition\weights\\plate_rec.onnx', help='model weight')
    parser.add_argument('--source', type=str,
                        default='./test_imgs', help='source')
    parser.add_argument('--output', type=str, default='./test_result',
                        help='output folder')  # output folder
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[i for i in range(4)], help='filter by class')  # 80 classes in COCO dataset
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    parser.add_argument('--names', nargs='+', type=str, default=['windows', 'plate', 'marker', 'person'],
                        help='class name')
    parser.add_argument('--vehicle-img-size', nargs='+', type=int, default=(320, 320), help='image size')
    parser.add_argument('--class-num', type=int, default=4, help='class number')
    parser.add_argument('--stride', nargs='+', type=int, default=[8, 16, 32], help='model stride')
    parser.add_argument('--anchor-list', nargs='+', type=int,
                        default=[[12, 8, 65, 23, 74, 26], [67, 58, 84, 66, 215, 66], [216, 95, 258, 91, 241, 109]],
                        help='anchor list')

    args = parser.parse_args()
    class_num = args.class_num
    stride = args.stride
    anchor_list = args.anchor_list
    img_size = args.vehicle_img_size
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    source = args.source
    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    windows_plate_marker_person_weight = args.windows_plate_marker_person_weight
    windows_plate_marker_person_model = YOLOV5_ONNX(onnx_path=windows_plate_marker_person_weight)
    plate_rec_onnx_model_path = args.plate_rec_weight
    plate_rec_session = onnxruntime.InferenceSession(plate_rec_onnx_model_path)

    if os.path.isdir(source):
        for tmp_file in os.listdir(source):
            img_input_path = os.path.join(source, tmp_file)
            img_output_path = os.path.join(output, tmp_file)
            print(f'current file: {img_input_path}')
            # !!!!! result savedir
            windows_plate_marker_person_model.infer(img_input_path, img_output_path)





"""
    args = parser.parse_args()
    args.img_size = [check_img_size(x) for x in args.img_size]  # verify img_size are gs-multiples
    # args.img_size = check_img_size(args.img_size)
    # print(args)
    names = args.names
    img_size = args.vehicle_img_size
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    class_num = args.class_num
    stride = args.stride
    anchor_list = args.anchor_list
    windows_plate_marker_person_weight = args.windows_plate_marker_person_weight
    windows_plate_marker_person_model = YOLOV5_ONNX(onnx_path=windows_plate_marker_person_weight)
    
    plate_rec_onnx_model_path = args.plate_rec_weight
    plate_rec_session = onnxruntime.InferenceSession(plate_rec_onnx_model_path)

    new_source = copy.copy(args.source)

    with torch.no_grad():
        if os.path.isdir(new_source):
            for tmp_file in os.listdir(new_source):
                args.source = os.path.join(new_source, tmp_file)
                args.output = os.path.splitext(tmp_file)[0]
                print(f'current file: {args.source}')
                print(f'current output dir: {args.output}')
                # !!!!! result savedir
                dst_dir = args.output
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                detect(args)
        else:
            tmp_file = os.path.basename(new_source)
            args.output = os.path.splitext(tmp_file)[0]
            dst_dir = args.output
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            detect(args)
"""