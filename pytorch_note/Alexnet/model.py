# @Time    : 2020/5/20 15:29
# @Author  : Libuda
# @FileName: model.py
# @Software: PyCharm
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self,num_class = 1000,inti_weights=False):
        """
        :param num_class: 分类个数
        :param inti_weights: 是否初始化权重
        """
        super(AlexNet, self).__init__()

        # 提取特征层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,padding=2,stride=4),
            # 增加计算量 降低内存使用
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

            # stride 默认为1 可以不写
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2,stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 256*6*6 为 上一层的输出 【6,6,256】
            nn.Linear(256*6*6,2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_class),
        )

    def forward(self,x):
        """
        前向传播
        :return:
        """
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
    alexnet = AlexNet()
    x = alexnet(input)
    print(alexnet)
    print(x.shape)