# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/7/20 14:49
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/7/20 14:49

import torch.nn as nn


class create_cnn_for_fmnist(nn.Module):
    def __init__(self):
        super(create_cnn_for_fmnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu4(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x
