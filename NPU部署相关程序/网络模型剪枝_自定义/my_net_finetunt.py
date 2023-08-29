# 接项目中my_net_prune.py之后进行在训练操作

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
        # print(m.weight.data.shape[0])  # 每个BN层权重w参数量：64/128/256
        # print(m.weight.data)
        total += m.weight.data.shape[0]

print("所有BN层总weight数量：", total)

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
pruned_num = 0  # 统计BN层剪枝通道数
cfg = []  # 统计保存通道数
cfg_mask = []  # BN层权重矩阵，剪枝的通道记为0，未剪枝通道记为1

for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        # print(weight_copy)
        mask = weight_copy.gt(thresh).float()  # 阈值分离权重
        # print(mask)
        # exit()
        pruned_num += mask.shape[0] - torch.sum(mask)  #
        # print(pruned_num)
        m.weight.data.mul_(mask)  # 更新BN层的权重，剪枝通道的权重值为0
        m.bias.data.mul_(mask)

        cfg.append(int(torch.sum(mask)))  # 记录未被剪枝的通道数量
        cfg_mask.append(mask.clone())
        print("layer index:{:d}\t total channel:{:d}\t remaining channel:{:d}".format(k, mask.shape[0],
                                                                                      int(torch.sum(mask))))
    elif isinstance(m, nn.AvgPool2d):
        cfg.append("A")

pruned_ratio = pruned_num / total
print("剪枝通道占比：", pruned_ratio)
print(cfg)
newmodel = net(cfg)


newmodel.load_state_dict(torch.load("prune_net.pth"))
#
optimer=torch.optim.Adam(model.parameters())
loss_fn=torch.nn.CrossEntropyLoss()
for e in range(100):
    x = torch.rand((1, 3, 20, 20))
    y=torch.tensor(np.random.randint(0,2,(1))).long()
    out=newmodel(x)
    loss=loss_fn(out,y)
    optimer.zero_grad()
    loss.backward()
    optimer.step()
torch.save(newmodel.state_dict(),"prune_net_finetunt.pth")