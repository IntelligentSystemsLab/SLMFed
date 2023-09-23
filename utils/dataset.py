# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 10:46
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 10:46

from numpy import ndarray
from torch.utils.data import Dataset


class simple_dataset(Dataset):

    def __init__(
            self,
            data: ndarray,
            label: ndarray
    ) -> None:
        # Initialize object properties.
        self.data = data
        self.label = label
        self.length = data.shape[0]
        # if len(self.label.shape)==1:
        #     self.label.resize((self.length,1))

    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return label, data

    def __len__(self):
        return self.length
