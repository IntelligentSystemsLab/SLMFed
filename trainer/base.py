# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 9:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 9:56

from abc import abstractmethod

import torch
from numpy import ndarray
from torch import nn

from utils.data import deepcopy_list


class BaseTrainer(object):

    def __init__(
            self,
            mode: str,
            model: nn.Module,
            optimizer: object,
            learning_rate: float,
            loss: object,
            meta: bool = False,
            batch_size: int = 16,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    ) -> None:
        # Initialize object properties.
        self.mode = mode
        self.meta = meta
        self.trained_num = 0
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.batch_size = batch_size
        self.train_epoch = train_epoch
        self.device = device
        self.upload_model = []
        self.upload_para_size = 0.0
        self.update_epoch=1

    @abstractmethod
    def train(self,
              *args
              ) -> None:
        pass

    def test(self,
             *args
             ) -> None:
        pass

    def predict(
            self,
            data: ndarray
    ) -> ndarray:
        # Call model objects to perform prediction.
        return self.model.predict(data)

    def update_local_model(
            self,
            new_global: list,
    ) -> None:
        # Call model objects to update model weight.

        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_global):
                p.data.copy_(d.data)

    def get_model(self) -> list:
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    @abstractmethod
    def get_upload_para(self) -> list:
        pass
