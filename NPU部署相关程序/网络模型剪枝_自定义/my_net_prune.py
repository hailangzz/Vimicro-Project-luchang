
import torch
import torch.nn as nn
import numpy as np

class net(nn.Module):
    def __init__(self, cfg=None):
        super(net, self).__init__()
        if cfg:
            self.features = self.make_layer(cfg)
            self.linear = nn.Linear(cfg[2], 2)
        else:
            layers = []
            layers += [nn.Conv2d(3, 64, 7, 2, 1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]

            layers += [
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ]
            layers += [
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ]
            layers += [nn.AvgPool2d(2)]
            self.features = nn.Sequential(*layers)
            self.linear = nn.Linear(256, 2)

    def make_layer(self, cfg):
        layers = []
        layers += [nn.Conv2d(3, cfg[0], 7, 2, 1, bias=False),
                   nn.BatchNorm2d(cfg[0]),
                   nn.ReLU(inplace=True)]

        layers += [
            nn.Conv2d(cfg[0], cfg[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cfg[1]),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(cfg[1], cfg[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(cfg[2]),
            nn.ReLU(inplace=True)
        ]
        layers += [nn.AvgPool2d(2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = net()
# 加载稀疏训练的模型
model.load_state_dict(torch.load("net.pth"))
total = 0  # 统计所有BN层的参数量
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        print(m.weight.data.shape[0],m.weight.data.shape)  # 每个BN层权重w参数量：64/128/256
        print(m.weight.data)
        total += m.weight.data.shape[0]

print("所有BN层总weight数量：", total)

# bn_data 用于存储所有需要进行剪枝的Batch Norm层的权值系数结果，将用来找出剪枝所需的权值系数阈值
bn_data = torch.zeros(total)
index = 0
for m in model.modules():
    # 将各个BN层的参数值拷贝到bn中
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn_data[index:(index + size)] = m.weight.data.abs().clone()
        index = size
# 对bn中的weight值排序
data, id = torch.sort(bn_data)
percent = 0.7  # 保留70%的BN层通道数
thresh_index = int(total * percent)
thresh = data[thresh_index]  # 取bn排序后的第thresh_index索引值为bn权重的截断阈值

# 制作mask
pruned_num = 0  # 统计当前的BN层被剪枝通道个数
cfg = []  # 统计整个模型的BN保存通道数（数组）
cfg_mask = []  # BN层权值系数相对于剪枝阈值的bool形mask位列数组记录，剪枝的通道记为0，未剪枝通道记为1

for k, m in enumerate(model.modules()):  #循环每个模型层的索引k 、及每层的算子名称 m
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone() # weight_copy 拷贝当前BN的权值系数层，并将和计算的权值系数阈值进行对比，确定此BN层的剪枝Mask项。
        # print(weight_copy)
        mask = weight_copy.gt(thresh).float()  # 阈值分离权重 #确定了当前BN层的剪枝bool形位列数组
        # print(mask)
        # exit()
        pruned_num += mask.shape[0] - torch.sum(mask)  # #当前BN层的被剪枝个数计算
        # print(pruned_num)
        m.weight.data.mul_(mask)  # 更新BN层的权重，剪枝通道的权重值为0 #这个函数的作用是，将当前层的BN权值系数直接按照bool形剪枝位列数组来进行归零操作。
        m.bias.data.mul_(mask)

        cfg.append(int(torch.sum(mask)))  # 记录未被剪枝的通道数量 #记录当前层的保留BN通道数
        cfg_mask.append(mask.clone()) #记录每层BN的剪枝bool形mask位列数组信息
        print("layer index:{:d}\t total channel:{:d}\t remaining channel:{:d}".format(k, mask.shape[0],
                                                                                      int(torch.sum(mask))))
    elif isinstance(m, nn.AvgPool2d): #当前层算子非BN且是AvgP时将往下执行
        cfg.append("A")

pruned_ratio = pruned_num / total  #计算BN层的剪枝比率：
print("剪枝通道占比：", pruned_ratio)
print(cfg)
newmodel = net(cfg) # 根据BN层阈值剪枝后的保留特征输出情况，重建原模型结构基础上的剪枝网络。
# print(newmodel)
# from torchsummary import summary
# print(summary(newmodel,(3,20,20),1))

layer_id_in_cfg = 0  # 层
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]  # 第一个BN层对应的mask
# print(cfg_mask)
# print(end_mask)

# 逐层对原模型算子、新剪枝模型算子进行，赋值操作（剪枝不改变模型深度、仅对模型层的特征输出数进行正则化调整）
for (m0, m1) in zip(model.modules(), newmodel.modules()):  # 以最少的为准
    if isinstance(m0, nn.BatchNorm2d):
        # idx1=np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))#获得mask中非零索引即未被减掉的序号
        # print(idx1)
        # exit()
        # idx1=np.array([1])
        # # print(idx1)
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
            # print(idx1)
        # exit()
        # 将旧模型的参数值拷贝到新模型中 #拷贝BN层的非剪权值系数、偏置项、批标准化均值、标准差等参数信息
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()

        #更新下一个BN的mask层的剪枝索引信息，并继续循环进行后续的剪枝处理操作
        layer_id_in_cfg += 1  # 下一个mask 更新到下一个mask层
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):  # 输入
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))  # 输入非0索引 #原始三通道输出时，全部保留的索引权值矩阵
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.numpy())))  # 输出非0索引  #由mask剪枝bool位列数组，确认的剪枝权值矩阵索引

        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        #此处为拷贝原训练模型当前层的全部系数矩阵到W1上
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()  #提取W1全部权值矩阵上，非剪枝索引下的权值矩阵到W1
        m1.weight.data = w1.clone() #拷贝保存好的非剪枝权值矩阵参数，更新了剪枝模型参数系数
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.numpy())))  # 输入非0索引
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save(newmodel.state_dict(), "prune_net.pth")
print(newmodel)