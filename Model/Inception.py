import torch
import torch.nn.functional as F


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        # 第一个分支 1*1  输出通道数16
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))

        # 第二个分支 输出通道数24
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=(1, 5), padding=(0, 2))

        # 第三个分支 输出通道数24
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=(1, 3), padding=(0, 1))

        # 第四个分支 输出通道数 24
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

    def forward(self, x):
        brach1x1 = self.branch1x1(x)

        brach5x5 = self.branch5x5_1(x)
        brach5x5 = self.branch5x5_2(brach5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [brach1x1, brach5x5, branch3x3, branch_pool]

        return torch.cat(output, dim=1)


class inception(torch.nn.Module):
    def __init__(self):
        super(inception, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(1, 5), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=(1, 5), padding=(0, 0))

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(kernel_size=(1, 3))

        #self.fc = torch.nn.Linear(877536, 9)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(877536, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            #torch.nn.Dropout(0.2),

            torch.nn.Linear(128, 9)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.incep1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
