import torch
import torch.nn as nn
import torch.functional as F

"""resnet"""


class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""

    def __init__(self, in_planes, planes, kernel_size=(1, 3), stride=(1, 1)):
        """

        :param in_planes: 输入通道
        :param planes:  输出通道
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        """
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
                                            , nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Sequential()

    def forward(self, inx):
        x = self.relu(self.bn1(self.conv1(inx)))
        x = self.bn2(self.conv2(x))
        out = x + self.downsample(inx)
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, basicBlock, blockNums, nb_classes):
        super(Resnet, self).__init__()
        self.in_planes = 64
        # 输入层
        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.layer1 = self._make_layers(basicBlock, blockNums[0], 64, (1, 1))
        self.layer2 = self._make_layers(basicBlock, blockNums[1], 128, (1, 2))
        self.layer3 = self._make_layers(basicBlock, blockNums[2], 256, (1, 2))
        self.layer4 = self._make_layers(basicBlock, blockNums[3], 512, (1, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(12, 1))
        self.fc = nn.Linear(512 * 12, nb_classes)

    def _make_layers(self, basicBlock, blockNum, plane, stride):
        """

        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param plane: 输出通道数
        :param stride: 卷积步长
        :return:
        """
        layers = []
        for i in range(blockNum):
            if i == 0:
                layer = basicBlock(self.in_planes, plane, (1, 3), stride=stride)
            else:
                layer = basicBlock(plane, plane, (1, 3), stride=(1, 1))
            layers.append(layer)
        self.in_planes = plane
        return nn.Sequential(*layers)

    def forward(self, inx):
        x = self.maxpool(self.relu(self.bn1(self.conv1(inx))))
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = nn.ReLU(out)

        return out


class ResNet(nn.Module):
    def __init__(self, esBlock, num_classes=9):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=(1, 1))
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=(1, 2))
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=(1, 2))
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=(1, 2))
        self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride[1]] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        print(x.shape)
        out = self.layer1(out)
        print(x.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
