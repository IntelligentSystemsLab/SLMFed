# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/7/20 15:49
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/7/20 15:49

from torch import nn


class create_cnn_for_germants(nn.Module):
    def __init__(self):
        super(create_cnn_for_germants, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-6)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-6)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-6)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, eps=1e-6)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 256)
        self.bn5 = nn.BatchNorm1d(256, eps=1e-6)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(256, 43)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x
