# -*- coding: utf-8 -*-
import numpy as np
import torch
import os


def cvtx0y0whTox1y1x2y2(x0, y0, w, h, imgShape=(576,1024,3)):
    # "0.530921 0.666667 0.622368 0.666667"=>(167, 169, 639, 507)
    # labelme 的COCO标注格式就是 中心点x+中心点y+宽+高 （归一化的）
    # 此函数出来的就是 左上点  右下点  （未归一化的）
    height, width, c = imgShape
    x1, y1, x2, y2 = int((x0 - w * 0.5) * width), \
                     int((y0 - h * 0.5) * height), \
                     int((x0 + w * 0.5) * width), \
                     int((y0 + h * 0.5) * height)
    return x1, y1, x2, y2

   
def get_one_label(file):
    # print(file)
    if not os.path.isfile(file):
        print("Label file %s is not exist." %file)
    with open(file, 'r') as f:
        l = torch.tensor(np.array([x.split() for x in f.read().splitlines()], dtype=np.float32))
    if len(l) == 0:
        l = torch.Tensor(0, 5).zero_().float()
    tcls = l[:, 0]
    tbox = l[:, 1:5]
    # tbox = l[:, 1:5]*640 # 640对应bin的尺寸
    # print(l[:, 1])

    for box_index in range(tbox.shape[0]):
        # print(tbox[box_index],tbox[box_index][0].item())
        tbox[box_index][0],tbox[box_index][1],tbox[box_index][2],tbox[box_index][3]=cvtx0y0whTox1y1x2y2(tbox[box_index][0],tbox[box_index][1],tbox[box_index][2],tbox[box_index][3])

    # tbox[0] = tbox[0] * 1024
    # tbox[2] = tbox[2] * 1024

    # tbox = l[:, 4] * 1024
    # print('tcls:',tcls)
    # print('tbox:',tbox)
    return tcls, tbox

def xywh2xyxy(x): # 坐标x1、y1、x2、y2转coco标注
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # box2 = xywh2xyxy(box2) # 将标注坐标转为coco
    area1 = box_area(box1.t())
    area2 = box_area(box2.t())
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def eveluate( tcls, tbox, pred):
    iouv = torch.linspace(0.5, 0.95, 10)
    label_num = len(tcls)
    detected = []
    correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool)

    # print(tcls,torch.unique(tcls))
    for cls in torch.unique(tcls):
        # print((cls == tcls).nonzero(as_tuple=False).view(-1)) #计算不同种类目标预测正确个数
        # print((cls == pred[:, 5]).nonzero(as_tuple=False).view(-1))
        # print(pred[:, 5])

        ti = (cls == tcls).nonzero(as_tuple=False).view(-1)
        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
        if pi.shape[0] == 0:
            continue
        # print(pred[pi, :4], tbox[ti]) #预测框体集和标注框体集
        ious,i = box_iou(pred[pi, :4], tbox[ti]).max(1)
        for j in (ious > iouv[0]).nonzero(as_tuple=False):
            d = ti[i[j]]  # detected target
            if d in detected:
                continue
            detected.append(d)
            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
            if len(detected) == label_num:  # all targets already located in image
                break
    return correct

def _ap_per_class( tp, conf, pred_cls, target_cls):
    #pr_conf_thres = 0.1
    pr_conf_thres = np.linspace(0, 1, 1000)
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = pr_conf_thres  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    nc = unique_classes.shape[0]
    #ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        if n_p == 0 or n_gt == 0:
            continue
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        # Recall
        recall = tpc / (n_gt + 1e-16)  # recall curve
        r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = _compute_ap(recall[:, j], precision[:, j])
    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1

def _compute_ap( recall, precision):
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    return ap

def read_result(file):
    box = []
    with open(file,'r') as f:
        result = f.readlines()

    if len(result)==0:
        box=[[0,0,0,0,0,9999]]
        # print(file)
    else:
        for line in result:
            line_parts = line.split(' ')
            cls = int((float(line_parts[1])))
            score = float(line_parts[2])
            x1 = float(line_parts[3])
            y1 = float(line_parts[4])
            x2 = float(line_parts[5])
            y2 = float(line_parts[6])
            box += [[x1, y1, x2, y2, score, cls]]

    return torch.tensor(np.array(box))

def compute_result():
    for i,file in enumerate(os.listdir(result_dir)):
        pred = read_result(result_dir + file)
        tcls, tbox = get_one_label(label_dir + file.replace('.bin_result','')) #获取标注类别及标注coco框体
        # print(tcls, tbox)
        if pred is None:
            result_vec.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        correct =eveluate(tcls, tbox, pred)
        result_vec.append((correct, pred[:, 4], pred[:, 5], tcls))

def calc_accuracy(method="11point"):
    stats = [np.concatenate(x, 0) for x in zip(*result_vec)]  # to numpy
    p, r, ap, f1= _ap_per_class(*stats)
    p50, r50, ap50, f1, ap = p[:, 0], r[:, 0], ap[:, 0], f1, ap.mean(1)    # [P, R, AP@0.5, AP@0.5:0.95]
    mp50, mr50, map50, map, f1= p.mean(), r.mean(), ap50.mean(), ap.mean(), f1.mean()
    print(('%12s' * 5) % ('P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1'))
    print(('%12.3g' * 5) % (mp50, mr50, map50, map, f1))

label_dir = './labels/' #labels里的label应跟输入bin的尺寸对应，例bin是416*416*3尺寸，label应对应416*416，如果不对应416*416，计算结果不准确
result_dir = './result/'#以result里面的结果个数去计算精度，例result_dir里有10个结果，精度就评估10个结果的精度
result_vec = []
compute_result()
calc_accuracy()