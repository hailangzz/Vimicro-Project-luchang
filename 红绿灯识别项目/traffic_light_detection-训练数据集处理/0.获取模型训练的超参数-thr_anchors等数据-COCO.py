'''
Arguments:
          dataset: 数据的yaml路径
          n: 类簇的个数
          img_size: 训练过程中的图片尺寸（32的倍数）
          thr: anchor的长宽比阈值，将长宽比限制在此阈值之内
          gen: k-means算法最大迭代次数（不理解的可以去看k-means算法）
          verbose: 打印参数

Usage:
          from utils.autoanchor import *; _ = kmean_anchors()

'''

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans

from tqdm import tqdm, trange
import os
import random

def get_thr_value(path='./data/ccpd.yaml'): # 获取标注框的长宽比范围值
    with open(path, 'rb') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        print(data_dict)

    total_images_list = os.listdir(data_dict['train'])

    w_h_min_rate=10
    w_h_max_rate=0
    for single_images_name in total_images_list:
        box_string2=single_images_name.split('-')[2].split('_') #[129&449 598&635]
        weight=int(box_string2[1].split('&')[0])-int(box_string2[0].split('&')[0])
        hight = int(box_string2[1].split('&')[1]) - int(box_string2[0].split('&')[1])

        if weight>0 and hight>0:
            w_h_rate=weight/hight
            if w_h_min_rate>w_h_rate:
                w_h_min_rate=w_h_rate
            if w_h_max_rate<w_h_rate:
                w_h_max_rate=w_h_rate
    print('w_h_min_rate:{},w_h_max_rate:{}'.format(w_h_min_rate,w_h_max_rate))
    return w_h_max_rate

def get_thr_coco_value(path='./data/ccpd.yaml'): # 获取标注框的长宽比范围值
    with open(path, 'rb') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        print(data_dict)

    total_images_list = os.listdir(data_dict['train']+'/../labels')

    w_h_min_rate=10
    w_h_max_rate=0
    for single_images_name in total_images_list:
        with open(os.path.join(data_dict['train']+'/../labels',single_images_name),'r') as label_cur:
            all_target_row_info = label_cur.readlines()
            for single_row in all_target_row_info:
                # print(single_row.split(' '))
                weight = float(single_row.split(' ')[3])*1280
                hight = float(single_row.split(' ')[4])*720

                if weight>0 and hight>0:
                    w_h_rate=weight/hight
                    if w_h_min_rate>w_h_rate:
                        w_h_min_rate=w_h_rate
                    if w_h_max_rate<w_h_rate:
                        w_h_max_rate=w_h_rate
    print('w_h_min_rate:{},w_h_max_rate:{}'.format(w_h_min_rate,w_h_max_rate))
    return w_h_max_rate


def kmean_anchors(path='./data/ccpd.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):

    # 将计算ancor计算结果写入到文本文件中：
    anchors_info_cur = open(r'./anchors.txt', 'a+')

    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.general import *; _ = kmean_anchors()
    """

    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):

        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg

            # 写入anchors数据
            try:
                # print('%i,%i' % (round(x[0]), round(x[1])))
                anchors_info_cur.write('%s,%s' % (str(round(x[0])), str(round(x[1]))))
                anchors_info_cur.write('\n')
            except Exception as e:
                print(e)
                continue

        return k

    if isinstance(path, str):  # *.yaml file
        with open(path,'rb') as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
            print(data_dict)
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)



    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


thr = get_thr_coco_value()
kmean_anchors(path='./data/ccpd_DVR.yaml', n=9, img_size=640, thr=thr)
