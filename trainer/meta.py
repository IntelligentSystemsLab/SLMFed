# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 9:57
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 9:57
import gc
import random
from typing import Tuple
import numpy as np

import torch
from numpy import ndarray
from ptflops import get_model_complexity_info
from torch import nn
from torch.utils.data import DataLoader

from trainer.base import BaseTrainer
from utils.data import get_cos_similar_matrix
from utils.dataset import simple_dataset


class MetaTrainer(BaseTrainer):

    def __init__(
            self,
            model: nn.Module,
            inner_learning_rate: float,
            loss: object,
            batch_size: int = 16,
            update_epoch: int = 1,
            train_epoch: int = 1,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            momentum: bool = False,
            batch_step: bool = False,
            layer_wise: bool = False,
            data_sequential: bool = True,
            layer_wise_limit: float = 1.0,
            meta_mode: str = 'FOMAML',
            cal_cost_base=(3, 540, 540),

    ) -> None:
        # Super class init.
        super().__init__(
            mode='direct',
            model=model,
            optimizer=None,
            learning_rate=0,
            loss=loss,
            batch_size=batch_size,
            train_epoch=train_epoch,
            device=device,
            meta=True
        )
        self.update_epoch = update_epoch
        self.inner_learning_rate = inner_learning_rate
        self.upload_model = None
        self.cal_cost_base = cal_cost_base
        _, model_para_size = get_model_complexity_info(self.model,cal_cost_base , print_per_layer_stat=False)
        self.all_upload_para_size = float(model_para_size[:-1])
        self.upload_para_size = float(model_para_size[:-1])
        self.momentum = momentum
        self.batch_step = batch_step
        self.layer_wise = layer_wise
        self.server_corr = []
        self.data_sequential = data_sequential
        self.last_weight = []
        self.last_grads = []
        self.layer_wise_limit = layer_wise_limit
        self.meta_mode = meta_mode

    def train(self,
              support_data: ndarray,
              support_label: ndarray,
              query_data: ndarray,
              query_label: ndarray
              ) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # if support_data.shape[-1] == 1:
        #     support_data = support_data.reshape(support_data.shape[0], 1, support_data.shape[1], support_data.shape[2])
        if not self.data_sequential:
            support_index_all = [i for i in range(support_label.shape[0])]
            query_index_all = [i for i in range(query_label.shape[0])]
            random.shuffle(support_index_all)
            random.shuffle(query_index_all)
            support_data = support_data[support_index_all]
            support_label = support_label[support_index_all]
            query_data = query_data[query_index_all]
            query_label = query_label[query_index_all]
        support_dataset = simple_dataset(data=support_data, label=support_label)
        if support_data.shape[0] % self.batch_size == 1:

            support_dataloader = DataLoader(dataset=support_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            support_dataloader = DataLoader(dataset=support_dataset, batch_size=self.batch_size)

        # if query_data.shape[-1] == 1:
        #     query_data = query_data.reshape(query_data.shape[0], 1, query_data.shape[1], query_data.shape[2])
        query_dataset = simple_dataset(data=query_data, label=query_label)
        if query_data.shape[0] % self.batch_size == 1:

            query_dataloader = DataLoader(dataset=query_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            query_dataloader = DataLoader(dataset=query_dataset, batch_size=self.batch_size)

        if self.meta_mode != 'FOMAML':
            self.current_parameters = [p.clone().cpu() for p in self.model.parameters()]

        # Client model training.
        for epoch in range(self.train_epoch):
            # Switch mode of model.
            self.model.train()
            self.model.to(self.device).half()
            support_loss_list = []
            support_acc_list = []
            for u_epoch in range(self.update_epoch):
                s_loss, s_acc = self.support_one_epoch(support_dataloader=support_dataloader, momentum=self.momentum,
                                                       batch_step=self.batch_step)
                support_loss_list.append(s_loss)
                support_acc_list.append(s_acc)
            if self.meta_mode == 'Reptile':
                query_grads = [torch.sub(p.cpu(), g).detach().cpu().numpy() for p, g in
                               zip(self.model.parameters(), self.current_parameters)]
                print('Epoch [%d/%d]:    Loss: %.4f    Acc: %.2f' % (
                    epoch + 1, self.train_epoch, support_loss_list[-1], support_acc_list[-1]))

            else:
                query_grads, q_loss, q_acc = self.query_one_epoch(query_dataloader=query_dataloader, train=True)

                print(
                    'Epoch [%d/%d]:    Support Loss: %.4f    Support Acc: %.2f   \n                   Query Loss: %.4f    '
                    'Query Acc: %.2f' % (
                        epoch + 1, self.train_epoch, support_loss_list[-1], support_acc_list[-1], q_loss, q_acc))

            if epoch + 1 < self.train_epoch:
                for p, g in zip(self.model.parameters(), query_grads):
                    p.data.add_(torch.from_numpy(g).to(self.device).data, alpha=-self.inner_learning_rate)
                    del g
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
        if self.layer_wise:
            if len(self.last_grads) != 0:
                all_size = sum([q.size for q in query_grads])
                real_size = all_size
                for i in range(len(self.last_grads)):
                    # similarity = get_cos_similar_matrix(query_grads[i], self.last_grads[i].data.detach().cpu().numpy())
                    similarity = torch.cosine_similarity(torch.from_numpy(query_grads[i]).reshape((1, -1)),
                                                         self.last_grads[i].reshape((1, -1)))
                    if similarity.item() >= self.layer_wise_limit:
                        real_size -= query_grads[i].size
                        query_grads[i] = self.last_grads[i].data.detach().cpu().numpy()
                self.upload_para_size = self.all_upload_para_size * real_size / all_size

            self.upload_model = query_grads
            torch.cuda.empty_cache()

        else:
            self.upload_model = query_grads
            torch.cuda.empty_cache()

    def test(self,
             support_data: ndarray,
             support_label: ndarray,
             query_data: ndarray,
             query_label: ndarray
             ) -> Tuple[float, float]:
        """

        Testing.

        """

        # if support_data.shape[-1] == 1:
        #     support_data = support_data.reshape(support_data.shape[0], 1, support_data.shape[1], support_data.shape[2])
        support_dataset = simple_dataset(data=support_data, label=support_label)
        if support_data.shape[0] % self.batch_size == 1:

            support_dataloader = DataLoader(dataset=support_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            support_dataloader = DataLoader(dataset=support_dataset, batch_size=self.batch_size)

        # if query_data.shape[-1] == 1:
        #     query_data = query_data.reshape(query_data.shape[0], 1, query_data.shape[1], query_data.shape[2])
        query_dataset = simple_dataset(data=query_data, label=query_label)
        if query_data.shape[0] % self.batch_size == 1:

            query_dataloader = DataLoader(dataset=query_dataset, batch_size=self.batch_size, drop_last=True)
        else:
            query_dataloader = DataLoader(dataset=query_dataset, batch_size=self.batch_size)

        # Switch mode of model.
        self.model.train()
        self.model.to(self.device).half()

        # Client model testing.
        support_loss_list = []
        support_acc_list = []
        for u_epoch in range(self.update_epoch):
            s_loss, s_acc = self.support_one_epoch(support_dataloader=support_dataloader, momentum=False,
                                                   batch_step=self.batch_step)
            support_loss_list.append(s_loss),
            support_acc_list.append(s_acc)

        self.model.eval()
        q_loss, q_acc = self.query_one_epoch(query_dataloader=query_dataloader, train=False)
        print('Test:    Support Loss: %.4f    Support Acc: %.2f   \n         Query Loss: %.4f    Query Acc: %.2f' % (
            support_loss_list[-1], support_acc_list[-1], q_loss, q_acc))

        return q_loss, q_acc

    def get_upload_para(self) -> list:
        return self.upload_model

    def support_one_epoch(self, support_dataloader, momentum=False, batch_step=False):
        support_loss = 0.0
        support_acc = 0.0
        support_total = 0
        batch_id = 0
        if batch_step:
            for data in support_dataloader:
                if momentum:
                    batch_id += 1
                    miu = np.exp(-batch_id / len(support_dataloader))
                else:
                    miu = 1
                input = data[1].float().to(self.device).half()
                label = data[0].float().to(self.device).half()
                output = self.model(input)
                loss = self.loss(output, label.long())
                support_loss += loss.item() * label.shape[0]
                _, pre = torch.max(output, dim=1)
                support_acc += pre.eq(label).sum().item()
                support_total += label.shape[0]
                support_grads = torch.autograd.grad(loss, list(self.model.parameters()),
                                                    create_graph=True, retain_graph=True)

                for p, g in zip(self.model.parameters(), support_grads):
                    p.data.add_(g.data, alpha=-self.inner_learning_rate * miu)
                    del g
                del input, label, output, loss, pre, _, support_grads
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
                torch.cuda.empty_cache()
        else:
            support_grads = None
            for data in support_dataloader:
                if momentum:
                    batch_id += 1
                    miu = np.exp(-batch_id / len(support_dataloader))
                else:
                    miu = 1
                input = data[1].float().to(self.device).half()
                label = data[0].float().to(self.device).half()
                output = self.model(input)
                loss = self.loss(output, label.long())
                support_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.detach().data, dim=1)
                support_acc += (pre == label).sum().item()
                support_total += label.shape[0]
                batch_grads = torch.autograd.grad(loss * label.shape[0] * miu, list(self.model.parameters()),
                                                  create_graph=True, retain_graph=True)
                if support_grads is None:
                    support_grads = [batch_g.detach().cpu().numpy() for batch_g in batch_grads]
                else:
                    for i in range(len(batch_grads)):
                        support_grads[i] += batch_grads[i].detach().cpu().numpy()
                del data, input, label, output, loss, pre, _, batch_grads
                torch.cuda.empty_cache()

            support_grads = [support_g / support_total for support_g in support_grads]
            for p, g in zip(self.model.parameters(), support_grads):
                p.data.add_(torch.from_numpy(g).to(self.device).data, alpha=-self.inner_learning_rate)
                del g

            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            del support_grads
        torch.cuda.empty_cache()
        return support_loss / support_total, support_acc / support_total * 100

    def query_one_epoch(self, query_dataloader, train=False):
        query_loss = 0.0
        query_acc = 0.0
        query_total = 0
        if train:
            query_grads = None
            for data in query_dataloader:
                input = data[1].float().to(self.device).half()
                label = data[0].float().to(self.device).half()
                output = self.model(input)
                loss = self.loss(output, label.long())
                query_loss += loss.detach().item() * label.shape[0]
                _, pre = torch.max(output.data, dim=1)
                query_acc += (pre == label).sum().item()
                query_total += label.shape[0]
                if self.meta_mode == 'MAML':
                    batch_grads = torch.autograd.grad(loss * label.shape[0], list(self.current_parameters),
                                                      create_graph=True, retain_graph=True)
                elif self.meta_mode == 'FOMAML':
                    batch_grads = torch.autograd.grad(loss * label.shape[0], list(self.model.parameters()),
                                                      create_graph=True, retain_graph=True)
                if query_grads is None:
                    query_grads = [batch_g.detach().cpu().numpy() for batch_g in batch_grads]
                else:
                    for i in range(len(batch_grads)):
                        query_grads[i] += batch_grads[i].detach().cpu().numpy()
                del input, label, output, pre, _, batch_grads
                torch.cuda.empty_cache()
            query_grads = [query_g / query_total for query_g in query_grads]
            return query_grads, query_loss / query_total, query_acc / query_total * 100
        else:
            with torch.no_grad():
                for data in query_dataloader:
                    input = data[1].float().to(self.device).half()
                    label = data[0].float().to(self.device).half()
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    query_loss += loss.detach().item() * label.shape[0]
                    _, pre = torch.max(output.data, dim=1)
                    query_acc += (pre == label).sum().item()
                    query_total += label.shape[0]
                    del input, label, output, pre, _
                    torch.cuda.empty_cache()
            return query_loss / query_total, query_acc / query_total * 100

    def update_local_model(
            self,
            new_global: list,
    ) -> None:
        # Call model objects to update model weight.
        if self.layer_wise:
            if self.trained_num == 0:
                with torch.no_grad():
                    self.last_weight = [p.data.clone().detach() for p in new_global]
            else:
                with torch.no_grad():
                    self.last_grads = []
                    for p, d in zip(self.last_weight, new_global):
                        self.last_grads.append(torch.sub(p, d))
                    with torch.no_grad():
                        self.last_weight = [p.data.clone().detach() for p in new_global]
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_global):
                p.data.copy_(d.data)
