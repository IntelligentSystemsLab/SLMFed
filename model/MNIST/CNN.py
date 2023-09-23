# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/5/24 16:15
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/5/24 16:15

from torch import nn


class create_cnn_for_mnist(nn.Module):
    def __init__(self):
        super(create_cnn_for_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)