import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides[0], padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ResNetBasicBlock(16, 16 ,[1, 1]),
            ResNetBasicBlock(16, 16, [1, 1]),
            ResNetBasicBlock(16, 16, [1, 1])
        )
        self.layer3 = nn.Sequential(
            ResNetBasicBlock(16, 32, [2, 1]),
            ResNetBasicBlock(32, 32, [1, 1]),
            ResNetBasicBlock(32, 32, [1, 1])
        )
        self.layer4 = nn.Sequential(
            ResNetBasicBlock(32, 64, [2, 1]),
            ResNetBasicBlock(64, 64, [1, 1]),
            ResNetBasicBlock(64, 64, [1, 1])
        )
        self.net = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.net(x)
        return out

if __name__ == '__main__':
    model = ResNet()
    print(model)