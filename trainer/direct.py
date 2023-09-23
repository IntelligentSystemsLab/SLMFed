# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 9:57
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 9:57
import copy

import torch
from numpy import ndarray
import numpy as np
from ptflops import get_model_complexity_info
from torch import nn
from torch.utils.data import DataLoader

from trainer.base import BaseTrainer
from utils.data import deepcopy_list
from utils.dataset import simple_dataset
from utils.loss import NTD_Loss


class DirectTrainer(BaseTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: object,
            learning_rate: float,
            loss: object,
            batch_size: int = 16,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            cal_cost_base=(3, 540, 540),
            cal_cost_base_option=True
    ) -> None:
        # Super class init.
        super().__init__(
            mode='direct',
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            batch_size=batch_size,
            train_epoch=train_epoch,
            device=device
        )
        self.cal_cost_base=cal_cost_base
        if cal_cost_base_option:
            _, model_para_size = get_model_complexity_info(self.model, self.cal_cost_base, print_per_layer_stat=False)
            self.upload_para_size = float(model_para_size[:-1])

    def train(self,
              train_data: ndarray,
              train_label: ndarray
              ) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Switch mode of model.
        self.model.train()

        # if train_data.shape[-1] == 1:
        #     train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        train_dataset = simple_dataset(data=train_data, label=train_label)
        if train_data.shape[0] % self.batch_size == 1:

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)

        # Client model training.
        for epoch in range(self.train_epoch):
            training_loss = 0.0
            training_acc = 0.0
            training_count = 0
            training_total = 0
            for data in train_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                self.model.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss(output, label.long())
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                _, pre = torch.max(output.data, dim=1)
                training_acc += (pre == label).sum().item()
                training_count += 1
                training_total += label.shape[0]

            print('Epoch [%d/%d]:    Loss: %.4f       Accuracy: %.2f' % (
                epoch+1, self.train_epoch, training_loss / training_count, training_acc / training_total * 100))
        self.optimizer.zero_grad()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.upload_model=deepcopy_list(weight)
        torch.cuda.empty_cache()

    def test(self,
             test_data: ndarray,
             test_label: ndarray,
             ):
        # Switch mode of model.
        self.model.eval()
        self.model.to(self.device)
        # if test_data.shape[-1] == 1:
        #     test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        test_dataset = simple_dataset(data=test_data, label=test_label)
        if test_data.shape[0] % self.batch_size == 1:

            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        # Client model testing.
        with torch.no_grad():
            testing_loss = 0.0
            testing_acc = 0.0
            testing_count = 0
            testing_total = 0
            for data in test_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                output = self.model(input)
                loss = self.loss(output, label.long())
                testing_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.data, dim=1)
                testing_acc += (pre == label).sum().item()
                testing_total += label.shape[0]

            loss_mean = testing_loss / testing_total
            acc_mean = testing_acc / testing_total * 100
            torch.cuda.empty_cache()
            print('Test:    Loss: %.4f       Accuracy: %.2f' % (
                loss_mean, acc_mean))
        return loss_mean, acc_mean

    def get_upload_para(self) -> list:
        return deepcopy_list(self.upload_model)

class ProxTrainer(BaseTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: object,
            learning_rate: float,
            loss: object,
            batch_size: int = 16,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            cal_cost_base=(3, 540, 540),
            cal_cost_base_option=True,
            miu=1
    ) -> None:
        # Super class init.
        super().__init__(
            mode='direct',
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            batch_size=batch_size,
            train_epoch=train_epoch,
            device=device
        )
        self.cal_cost_base=cal_cost_base
        self.miu=miu
        if cal_cost_base_option:
            _, model_para_size = get_model_complexity_info(self.model, self.cal_cost_base, print_per_layer_stat=False)
            self.upload_para_size = float(model_para_size[:-1])

    def train(self,
              train_data: ndarray,
              train_label: ndarray
              ) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Switch mode of model.
        self.model.train()

        # if train_data.shape[-1] == 1:
        #     train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        train_dataset = simple_dataset(data=train_data, label=train_label)
        if train_data.shape[0] % self.batch_size == 1:

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)

        # Client model training.
        global_model_p=self.get_upload_para()
        for epoch in range(self.train_epoch):
            training_loss = 0.0
            training_acc = 0.0
            training_count = 0
            training_total = 0
            for data in train_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                self.model.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                proximal_term=0.0
                for w, w_t in zip(self.model.parameters(), global_model_p):
                    proximal_term += (w - w_t.to(self.device)).norm(2)
                loss = self.loss(output, label.long())+proximal_term*self.miu/2
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                _, pre = torch.max(output.data, dim=1)
                training_acc += (pre == label).sum().item()
                training_count += 1
                training_total += label.shape[0]

            print('Epoch [%d/%d]:    Loss: %.4f       Accuracy: %.2f' % (
                epoch+1, self.train_epoch, training_loss / training_count, training_acc / training_total * 100))
        self.optimizer.zero_grad()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.upload_model=deepcopy_list(weight)
        torch.cuda.empty_cache()

    def test(self,
             test_data: ndarray,
             test_label: ndarray,
             ):
        # Switch mode of model.
        self.model.eval()
        self.model.to(self.device)
        # if test_data.shape[-1] == 1:
        #     test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        test_dataset = simple_dataset(data=test_data, label=test_label)
        if test_data.shape[0] % self.batch_size == 1:

            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        # Client model testing.
        with torch.no_grad():
            testing_loss = 0.0
            testing_acc = 0.0
            testing_count = 0
            testing_total = 0
            for data in test_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                output = self.model(input)
                loss = self.loss(output, label.long())
                testing_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.data, dim=1)
                testing_acc += (pre == label).sum().item()
                testing_total += label.shape[0]

            loss_mean = testing_loss / testing_total
            acc_mean = testing_acc / testing_total * 100
            torch.cuda.empty_cache()
            print('Test:    Loss: %.4f       Accuracy: %.2f' % (
                loss_mean, acc_mean))
        return loss_mean, acc_mean

    def get_upload_para(self) -> list:
        return deepcopy_list(self.upload_model)


class FedNTDTrainer(BaseTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: object,
            learning_rate: float,
            loss: object,
            batch_size: int = 16,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            cal_cost_base=(3, 540, 540),
            cal_cost_base_option=True
    ) -> None:
        # Super class init.
        super().__init__(
            mode='direct',
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            batch_size=batch_size,
            train_epoch=train_epoch,
            device=device
        )
        self.cal_cost_base=cal_cost_base
        if cal_cost_base_option:
            _, model_para_size = get_model_complexity_info(self.model, self.cal_cost_base, print_per_layer_stat=False)
            self.upload_para_size = float(model_para_size[:-1])
        self.criterion=NTD_Loss()
        self.dg_model = copy.deepcopy(self.model)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False

    def train(self,
              train_data: ndarray,
              train_label: ndarray
              ) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Switch mode of model.
        self.model.train()

        # if train_data.shape[-1] == 1:
        #     train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        train_dataset = simple_dataset(data=train_data, label=train_label)
        if train_data.shape[0] % self.batch_size == 1:

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)

        # Client model training.
        for epoch in range(self.train_epoch):
            training_loss = 0.0
            training_acc = 0.0
            training_count = 0
            training_total = 0
            for data in train_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                self.model.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                dg_logits=self._get_dg_logits(input)
                try:
                    loss = self.criterion(output, label.long(),dg_logits)
                except:
                    self.criterion.num_classes=43
                    loss = self.criterion(output, label.long(), dg_logits)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                _, pre = torch.max(output.data, dim=1)
                training_acc += (pre == label).sum().item()
                training_count += 1
                training_total += label.shape[0]

            print('Epoch [%d/%d]:    Loss: %.4f       Accuracy: %.2f' % (
                epoch+1, self.train_epoch, training_loss / training_count, training_acc / training_total * 100))
        self.optimizer.zero_grad()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.upload_model=deepcopy_list(weight)
        torch.cuda.empty_cache()

    def test(self,
             test_data: ndarray,
             test_label: ndarray,
             ):
        # Switch mode of model.
        self.model.eval()
        self.model.to(self.device)
        # if test_data.shape[-1] == 1:
        #     test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        test_dataset = simple_dataset(data=test_data, label=test_label)
        if test_data.shape[0] % self.batch_size == 1:

            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        # Client model testing.
        with torch.no_grad():
            testing_loss = 0.0
            testing_acc = 0.0
            testing_count = 0
            testing_total = 0
            for data in test_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                output = self.model(input)
                loss = self.loss(output, label.long())
                testing_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.data, dim=1)
                testing_acc += (pre == label).sum().item()
                testing_total += label.shape[0]

            loss_mean = testing_loss / testing_total
            acc_mean = testing_acc / testing_total * 100
            torch.cuda.empty_cache()
            print('Test:    Loss: %.4f       Accuracy: %.2f' % (
                loss_mean, acc_mean))
        return loss_mean, acc_mean

    def get_upload_para(self) -> list:
        return deepcopy_list(self.upload_model)

    def _get_dg_logits(self, data):

        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits



class SoteriaFLTrainer(BaseTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: object,
            learning_rate: float,
            loss: object,
            batch_size: int = 16,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            cal_cost_base=(3, 540, 540),
            cal_cost_base_option=True,
            gamma=0.5

    ) -> None:
        # Super class init.
        super().__init__(
            mode='direct',
            model=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            batch_size=batch_size,
            train_epoch=train_epoch,
            device=device
        )
        self.cal_cost_base=cal_cost_base
        self.initial_reference = [0 for i in range(len(list(self.model.parameters())))]
        self.gamma=gamma
        if cal_cost_base_option:
            _, model_para_size = get_model_complexity_info(self.model, self.cal_cost_base, print_per_layer_stat=False)
            self.upload_para_size = float(model_para_size[:-1])

    def train(self,
              train_data: ndarray,
              train_label: ndarray
              ) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Switch mode of model.
        self.model.train()
        layer_num=len(list(self.model.parameters()))
        # if train_data.shape[-1] == 1:
        #     train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        train_dataset = simple_dataset(data=train_data, label=train_label)
        if train_data.shape[0] % self.batch_size == 1:

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)

        # Client model training.
        for epoch in range(self.train_epoch):
            training_loss = 0.0
            training_acc = 0.0
            training_count = 0
            training_total = 0
            for data in train_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                self.model.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss(output, label.long())
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                _, pre = torch.max(output.data, dim=1)
                training_acc += (pre == label).sum().item()
                training_count += 1
                training_total += label.shape[0]

            print('Epoch [%d/%d]:    Loss: %.4f       Accuracy: %.2f' % (
                epoch + 1, self.train_epoch, training_loss / training_count, training_acc / training_total * 100))
        self.optimizer.zero_grad()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        for l in range(layer_num):
            weight[l]=weight[l]*2-self.initial_reference[l]
            self.initial_reference[l]=(1-self.gamma)*self.initial_reference[l]+self.gamma*weight[l]
        self.upload_model = deepcopy_list(weight)
        torch.cuda.empty_cache()

    def test(self,
             test_data: ndarray,
             test_label: ndarray,
             ):
        # Switch mode of model.
        self.model.eval()
        self.model.to(self.device)
        # if test_data.shape[-1] == 1:
        #     test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        test_dataset = simple_dataset(data=test_data, label=test_label)
        if test_data.shape[0] % self.batch_size == 1:

            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        # Client model testing.
        with torch.no_grad():
            testing_loss = 0.0
            testing_acc = 0.0
            testing_count = 0
            testing_total = 0
            for data in test_dataloader:
                torch.cuda.empty_cache()
                input = data[1].float().to(self.device)
                label = data[0].float().to(self.device)
                output = self.model(input)
                loss = self.loss(output, label.long())
                testing_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.data, dim=1)
                testing_acc += (pre == label).sum().item()
                testing_total += label.shape[0]

            loss_mean = testing_loss / testing_total
            acc_mean = testing_acc / testing_total * 100
            torch.cuda.empty_cache()
            print('Test:    Loss: %.4f       Accuracy: %.2f' % (
                loss_mean, acc_mean))
        return loss_mean, acc_mean

    def get_upload_para(self) -> list:
        return deepcopy_list(self.upload_model)
