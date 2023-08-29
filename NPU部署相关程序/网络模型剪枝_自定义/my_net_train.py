import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

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


def updateBN(model, s=0.0001):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = net().to(device)
    summary(model,(3,20,20))
    x = torch.rand((1, 3, 20, 20)).to(device)
    print(model(x).to(device))
    optimer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(100):
        x = torch.rand((1, 3, 20, 20)).to(device)
        y = torch.tensor(np.random.randint(0, 2, (1))).long().to(device)
        out = model(x)
        loss = loss_fn(out, y)
        optimer.zero_grad()
        loss.backward()
        # BN权重稀疏化
        updateBN(model)
        optimer.step()
    torch.save(model.state_dict(), "net.pth")

