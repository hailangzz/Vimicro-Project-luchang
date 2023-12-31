import torch.nn as nn
import torch
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            small_basic_block(ch_in=128, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(1, 13), stride=1, padding=[0, 6]),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.connected = nn.Sequential(
            nn.Linear(class_num * 88, 128),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=128 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.backbone(x)
        pattern = x.flatten(1, -1)
        pattern = self.connected(pattern)
        width = x.size()[-1]
        pattern = torch.reshape(pattern, [-1, 128, 1, 1])
        pattern = pattern.repeat(1, 1, 1, width)
        x = torch.cat([x, pattern], dim=1)
        x = self.container(x)
        logits = x.squeeze(2)

        return logits




    # https://blog.csdn.net/weixin_39027619/article/details/106143755
    # def forward(self, x):
    #     x = self.backbone(x)
    #     pattern = x.flatten(1, -1)
    #     pattern = self.connected(pattern)
    #     width = x.size()[-1]
    #     pattern = torch.reshape(pattern, [-1, 128, 1, 1])
    #     pattern = pattern.repeat(1, 1, 1, width)
    #     x = torch.cat([x, pattern], dim=1)
    #     x = self.container(x)
    #     logits = x.squeeze(2)
    #     return logits


# def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):
#
#     Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)
#
#     if phase == "train":
#         return Net.train()
#     else:
#         return Net.eval()
