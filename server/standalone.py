# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 12:24
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 12:24
import os
import time
from queue import Queue
import random
import numpy as np
import torch
from numpy import ndarray
from ptflops import get_model_complexity_info
from torch import nn
from torch.utils.data import DataLoader

from aggregation.base import BaseFed
from client.standalone import StandAloneClient

from selection.base import BaseSelect
from selection.function import random_select_with_percentage, softmax_prob_from_indicators, rank_select_with_percentage
from utils.client import client_increment
from utils.data import deepcopy_list, calculate_SIC, fedprofkl
from utils.dataset import simple_dataset
from utils.model import compute_rda_alllayers, compute_rc_simp, choose_layer
from utils.zip import zip_model, read_zip


class StandAloneServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: BaseSelect,
            eval_target: float,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            client_pool: list = [],
            eval_client: list = [],
            init_weight: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.client_pool = client_pool
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.round_time = []
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        _, model_para_size = get_model_complexity_info(self.model, cal_cost_base, print_per_layer_stat=False)
        self.model_para_size = float(model_para_size[:-1])

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround

        self.select.client_list = self.client_pool

        # Get selected clients.
        self.client_selected = self.select.select()

        # Start Task.
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)

        # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
        self.client_queue = Queue()
        for selected_client in self.client_selected:
            self.client_queue.put(selected_client)
            selected_client.field = self.aggregation.get_field()
        self.update_clients_model()

        # Circular queue to execute task workflow.
        while not self.task_stop:
            # Queue first client object out of the queue.
            self.client_one_solve()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_from_queue.get_upload_para(),
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(3, 20),
            client_true_time=time.time() - begin_time,
            client_cost=client_from_queue.trainer.upload_para_size,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))
        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)

            # Judge the evaluation situation.
            if self.run_evaluation():
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.
                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_pool.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.test()
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            self.clients_increment()
            if self.current_round() % self.round_increment == 0:
                self.clients_update_increment()
            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()


class SLMFedStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround

        select_prob = [0.5 for c in self.client_now]
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=select_prob,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                SIC_client = []
                datasize_client = []
                for c in self.client_now:
                    SIC_client.append(
                        calculate_SIC(new_label=c.increment_train_label,
                                      old_label=c.train_label,
                                      c=1.01))
                    c.update_client_increment_data()
                    datasize_client.append(c.datasize)
                    c.round_increment = False
                client_select_prob = softmax_prob_from_indicators([SIC_client, datasize_client])
                if self.select == 'random':
                    select_prob = [0.5 for c in self.client_now]
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'random_perc':
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=client_select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'rank_perc':
                    client_select = rank_select_with_percentage(client_list=self.client_now,
                                                                client_selected_probability=client_select_prob,
                                                                select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()
        if len(self.eval_record) != 0 and self.eval_record[-1] > self.base_acc:
            client_rc = []
            local_rdm_alllayer = compute_rda_alllayers(
                layer_num=len(client_from_queue.trainer.model.state_dict()),
                model=client_from_queue.trainer.model,
                data_stimulus=self.data_stimulus,
                distance_measure='correlation',
                device=self.device
            )
            for layer_idx in range(len(local_rdm_alllayer)):
                if type(local_rdm_alllayer[layer_idx]) == int or type(
                        self.global_rdm_alllayer[layer_idx]) == int:
                    client_rc.append(999)
                else:
                    Pearson_coefficient_onelayer = compute_rc_simp(
                        local_rdm_alllayer[layer_idx],
                        self.global_rdm_alllayer[layer_idx])
                    representational_consistency_onelayer = \
                        Pearson_coefficient_onelayer[0] * \
                        Pearson_coefficient_onelayer[0]
                    client_rc.append(representational_consistency_onelayer)
            upload_prob = softmax_prob_from_indicators([client_rc])
            client_model = choose_layer(client_model, upload_prob)

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.
                if len(self.eval_record) != 0 and self.eval_record[-1] > self.base_acc:
                    self.global_rdm_alllayer = compute_rda_alllayers(
                        layer_num=len(self.model.state_dict()),
                        model=self.model,
                        data_stimulus=self.data_stimulus,
                        distance_measure='correlation',
                        device=self.device
                    )
                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))
        return sim_zip_cc


class NormalStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround

        select_prob = [0.5 for c in self.client_now]
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=select_prob,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                for c in self.client_now:
                    # SIC_client.append(
                    #     calculate_SIC(new_label=c.increment_train_label,
                    #                   old_label=c.train_label,
                    #                   c=1.01))
                    c.update_client_increment_data()
                    # datasize_client.append(c.datasize)
                    c.round_increment = False
                # client_select_prob = softmax_prob_from_indicators([SIC_client, datasize_client])
                if self.select == 'random':
                    select_prob = [0.5 for c in self.client_now]
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'NOMA':
                    r_kef = []
                    ph_sum = 0
                    tau = 1
                    omiga = -174
                    p_k = 0.1
                    for c in self.client_now:
                        h_k = self.select_percentage * np.random.normal(loc=0, scale=1)
                        r_k = np.log2(1 + (p_k * h_k * h_k) / (tau * (ph_sum + omiga)))
                        ph_sum += p_k * h_k * h_k
                        t_k = self.aggregation.aggregation_version
                        r_kef.append(5 * r_k * t_k / c.datasize)
                    select_prob = softmax_prob_from_indicators([r_kef])
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'FedProf':
                    kl = []
                    for c in self.client_now:
                        kl.append(np.exp(-10 * fedprofkl(c.train_label, self.client_now[0].total_train_label)))
                    client_selected = random_select_with_percentage(client_list=self.client_now,
                                                                    client_selected_probability=kl,
                                                                    select_percentage=self.select_percentage)

                # elif self.select == 'random_perc':
                #     client_select = random_select_with_percentage(client_list=self.client_now,
                #                                                   client_selected_probability=client_select_prob,
                #                                                   select_percentage=self.select_percentage)
                # elif self.select == 'rank_perc':
                #     client_select = rank_select_with_percentage(client_list=self.client_now,
                #                                                 client_selected_probability=client_select_prob,
                #                                                 select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc


class FedProfStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )
        self.sim_data_cc = len(list(self.client_now[0].trainer.model.parameters())) * 2 * 32

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround
        kl = []
        for c in self.client_now:
            kl.append(np.exp(-10 * fedprofkl(c.train_label, self.client_now[0].total_train_label)))
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=kl,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.round_cost += self.sim_data_cc * len(self.client_now)
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                for c in self.client_now:
                    # SIC_client.append(
                    #     calculate_SIC(new_label=c.increment_train_label,
                    #                   old_label=c.train_label,
                    #                   c=1.01))
                    c.update_client_increment_data()
                    # datasize_client.append(c.datasize)
                    c.round_increment = False
                # client_select_prob = softmax_prob_from_indicators([SIC_client, datasize_client])

                kl = []
                for c in self.client_now:
                    kl.append(np.exp(-10 * fedprofkl(c.train_label, self.client_now[0].total_train_label)))
                client_select = random_select_with_percentage(client_list=self.client_now,
                                                              client_selected_probability=kl,
                                                              select_percentage=self.select_percentage)

                # elif self.select == 'random_perc':
                #     client_select = random_select_with_percentage(client_list=self.client_now,
                #                                                   client_selected_probability=client_select_prob,
                #                                                   select_percentage=self.select_percentage)
                # elif self.select == 'rank_perc':
                #     client_select = rank_select_with_percentage(client_list=self.client_now,
                #                                                 client_selected_probability=client_select_prob,
                #                                                 select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.round_cost += self.sim_data_cc * len(self.client_now)
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc


class OortStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround
        util_clients = [(random.uniform(10, 100) / 100) ** 2 for c in self.client_now]
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=util_clients,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                for c in self.client_now:
                    # SIC_client.append(
                    #     calculate_SIC(new_label=c.increment_train_label,
                    #                   old_label=c.train_label,
                    #                   c=1.01))
                    c.update_client_increment_data()
                    # datasize_client.append(c.datasize)
                    c.round_increment = False
                # client_select_prob = softmax_prob_from_indicators([SIC_client, datasize_client])

                client_select = rank_select_with_percentage(client_list=self.client_now,
                                                            client_selected_probability=util_clients,
                                                            select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            util_clients = []

            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

                util_clients.append(loss_incre * (random.uniform(10, 100) / 100) ** 2)
            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc


class E3CSStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.E3CS_exponential_weight = dict()
        self.E3CS_p = dict()
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.sigma = (self.select_percentage + 0) / 2
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround
        select_prob = []
        exponential_weight = []
        for c in self.client_now:
            exponential_weight.append(1)
            select_prob.append(
                self.sigma + (len(self.client_now) * self.select_percentage - len(self.client_now) * self.sigma) * 1 / (
                        len(self.client_now) - 1))
            self.E3CS_p[c.client_name] = self.sigma + (
                    len(self.client_now) * self.select_percentage - len(self.client_now) * self.sigma) * 1 / (
                                                 len(self.client_now) - 1)
        select_prob = softmax_prob_from_indicators([select_prob])
        # exponential_weight = softmax_prob_from_indicators([exponential_weight])
        temp_id = 0
        for c in self.client_now:
            self.E3CS_exponential_weight[c.client_name] = exponential_weight[temp_id]
            temp_id += 1

        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=select_prob,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                select_prob = []
                exponential_weight = []
                for c in self.client_now:
                    if c.client_name not in self.E3CS_exponential_weight.keys():
                        exponential_weight.append(1)
                    else:
                        exponential_weight.append(self.E3CS_exponential_weight[c.client_name])
                # exponential_weight = softmax_prob_from_indicators([exponential_weight])
                temp_id = 0
                for c in self.client_now:
                    self.E3CS_exponential_weight[c.client_name] = exponential_weight[temp_id]
                    exponential_weight.append(1 / len(self.client_now))
                    select_prob.append(self.sigma + (
                            len(self.client_now) * self.select_percentage - len(self.client_now) * self.sigma) *
                                       exponential_weight[temp_id] / (
                                               sum(exponential_weight) - exponential_weight[temp_id]))
                    self.E3CS_p[c.client_name] = self.sigma + (
                            len(self.client_now) * self.select_percentage - len(self.client_now) * self.sigma) * \
                                                 exponential_weight[temp_id] / (
                                                         sum(exponential_weight) - exponential_weight[temp_id])
                    temp_id += 1
                select_prob = softmax_prob_from_indicators([select_prob])
                client_select = random_select_with_percentage(client_list=self.client_now,
                                                              client_selected_probability=select_prob,
                                                              select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field={'E3CS_weight': self.E3CS_exponential_weight[client_from_queue.client_name]},
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            for c in self.client_now:
                if c.client_name not in self.E3CS_exponential_weight.keys():
                    self.E3CS_exponential_weight[c.client_name] = 1
                else:
                    try:
                        self.E3CS_exponential_weight[c.client_name] = self.E3CS_exponential_weight[
                                                                          c.client_name] * np.exp((
                                                                                                          (len(
                                                                                                              self.client_now) * self.select_percentage - len(
                                                                                                              self.client_now) * self.sigma) * 0.5 * len(
                                                                                                      self.client_now) * self.select_percentage / len(
                                                                                                      self.client_now) /
                                                                                                          self.E3CS_p[
                                                                                                              c.client_name]) / 200)
                    except:
                        self.E3CS_exponential_weight[c.client_name] = self.E3CS_exponential_weight[
                                                                          c.client_name] * np.exp((
                                                                                                          (len(
                                                                                                              self.client_now) * self.select_percentage - len(
                                                                                                              self.client_now) * self.sigma) * 0.5 * len(
                                                                                                      self.client_now) * self.select_percentage / len(
                                                                                                      self.client_now) / np.mean(
                                                                                                      list(
                                                                                                          self.E3CS_p.values()))) / 200)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc


class FedBalancerStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround

        select_prob = [0.5 for c in self.client_now]
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=select_prob,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                select_prob = []
                for c in self.client_now:
                    loss, _ = c.trainer.test(test_data=c.train_data,
                                             test_label=c.train_label)
                    self.round_cost += 64 * c.train_label.shape[0] / c.trainer.batch_size
                    select_prob.append(len(self.client_now) * (loss * loss / len(self.client_now)) ** 0.5)
                client_select_prob = softmax_prob_from_indicators([select_prob])
                client_select = random_select_with_percentage(client_list=self.client_now,
                                                              client_selected_probability=client_select_prob,
                                                              select_percentage=self.select_percentage)

                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc


class SoteriaFLStageServer(object):

    def __init__(
            self,
            aggregation: BaseFed,
            model: nn.Module,
            select: str,
            eval_target: float,
            select_percentage: float,
            base_acc: float,
            client_incre_min_perc: float,
            client_incre_max_perc: float,
            client_incre_prob: float,
            target_pnew: float,
            target_ue: float,
            stage_total: int,
            data_stimulus=None,
            eval_data: ndarray = np.array([]),
            eval_label: ndarray = np.array([]),
            convergence: float = 0.001,
            convergence_round: int = 10,
            server_name: str = 'Server1',
            eval_client: list = [],
            init_weight: list = [],
            client_now: list = [],
            client_rest: list = [],
            ration: int = 0,
            batch_size: int = 16,
            device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0'),
            loss=None,
            record_file=None,
            round_increment: int = 100,
            cal_cost_base=(3, 540, 540)

    ) -> None:

        # Initialize object properties.
        self.aggregation = aggregation
        self.synchronous = aggregation.synchronous
        self.model = model
        self.select = select
        self.client_incre_min_perc = client_incre_min_perc
        self.client_incre_max_perc = client_incre_max_perc
        self.client_incre_prob = client_incre_prob
        self.select_percentage = select_percentage
        self.eval_data = eval_data
        self.eval_label = eval_label
        self.target_pnew = target_pnew
        self.target_ue = target_ue
        self.base_acc = base_acc
        self.data_stimulus = data_stimulus
        self.eval_target = eval_target
        self.convergence = convergence
        self.loss = loss
        self.stage_total = stage_total
        self.record_file = record_file
        self.convergence_round = convergence_round
        self.server_name = server_name
        self.eval_client = eval_client
        self.ration = ration
        self.batch_size = batch_size
        self.device = device
        self.round_increment = round_increment
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []
        self.maxround = 0
        self.task_stop = False
        self.global_rdm_alllayer = None
        self.round_time = []
        self.client_now = client_now
        self.client_rest = client_rest
        self.round_true_time = []
        self.time_record = []
        self.true_time_record = []
        self.round_cost = 0
        self.stage_run_iter = 0
        self.cost_record = []
        self.eval_record = []
        self.loss_record = []
        self.cal_cost_base = cal_cost_base
        if len(init_weight) != 0:
            self.update_model(new_model=init_weight)
        self.client_queue = Queue()
        with torch.no_grad():
            weight = [p.cpu().data.clone().detach() for p in self.model.parameters()]
        self.model_para_size = self.sim_zip_for_cc(
            model=deepcopy_list(weight),
            filename=self.server_name,
        )

    def start_task(
            self,
            maxround: int,
    ) -> None:

        self.maxround = maxround

        select_prob = [0.5 for c in self.client_now]
        self.client_selected = random_select_with_percentage(client_list=self.client_now,
                                                             client_selected_probability=select_prob,
                                                             select_percentage=self.select_percentage)

        # Start Task.
        stage_id = 1
        stage_trans = False
        self.round_time = []
        self.round_true_time = []
        self.round_cost = 0
        self.task_stop = False
        print("Task on Server %s starts." % self.server_name)
        print('Stage: ' + str(stage_id) + '\n')
        self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
        self.record_file.flush()
        self.stage_run_iter = 0

        while stage_id <= self.stage_total:
            self.aggregation.aggregation_version = 0
            self.task_stop = False
            self.aggregation.initial_reference = None
            for c in self.client_now:
                c.trainer.initial_reference = [0 for i in range(len(list(c.trainer.model.parameters())))]
            if stage_id > 1 and stage_trans:
                print('Stage: ' + str(stage_id) + '\n')
                self.record_file.writelines('Stage: ' + str(stage_id) + '\n')
                self.record_file.flush()
                stage_trans = False
                self.round_time = []
                self.round_true_time = []
                self.round_cost = 0
                self.time_record = []
                self.true_time_record = []
                self.round_cost = 0
                self.cost_record = []
                self.eval_record = []
                self.loss_record = []
                # SIC_client = []
                # datasize_client = []
                for c in self.client_now:
                    # SIC_client.append(
                    #     calculate_SIC(new_label=c.increment_train_label,
                    #                   old_label=c.train_label,
                    #                   c=1.01))
                    c.update_client_increment_data()
                    # datasize_client.append(c.datasize)
                    c.round_increment = False
                # client_select_prob = softmax_prob_from_indicators([SIC_client, datasize_client])
                if self.select == 'random':
                    select_prob = [0.5 for c in self.client_now]
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'NOMA':
                    r_kef = []
                    ph_sum = 0
                    tau = 1
                    omiga = -174
                    p_k = 0.1
                    for c in self.client_now:
                        h_k = self.select_percentage * np.random.normal(loc=0, scale=1)
                        r_k = np.log2(1 + (p_k * h_k * h_k) / (tau * (ph_sum + omiga)))
                        ph_sum += p_k * h_k * h_k
                        t_k = self.aggregation.aggregation_version
                        r_kef.append(5 * r_k * t_k / c.datasize)
                    select_prob = softmax_prob_from_indicators([r_kef])
                    client_select = random_select_with_percentage(client_list=self.client_now,
                                                                  client_selected_probability=select_prob,
                                                                  select_percentage=self.select_percentage)
                elif self.select == 'FedProf':
                    kl = []
                    for c in self.client_now:
                        kl.append(np.exp(-10 * fedprofkl(c.train_label, self.client_now[0].total_train_label)))
                    client_selected = random_select_with_percentage(client_list=self.client_now,
                                                                    client_selected_probability=kl,
                                                                    select_percentage=self.select_percentage)

                # elif self.select == 'random_perc':
                #     client_select = random_select_with_percentage(client_list=self.client_now,
                #                                                   client_selected_probability=client_select_prob,
                #                                                   select_percentage=self.select_percentage)
                # elif self.select == 'rank_perc':
                #     client_select = rank_select_with_percentage(client_list=self.client_now,
                #                                                 client_selected_probability=client_select_prob,
                #                                                 select_percentage=self.select_percentage)
                self.client_selected = deepcopy_list(client_select)

            # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
            self.client_queue = Queue()
            for selected_client in self.client_selected:
                self.client_queue.put(selected_client)
                selected_client.field = self.aggregation.get_field()
            self.update_clients_model()

            # Circular queue to execute task workflow.
            while not self.task_stop:
                # Queue first client object out of the queue.
                self.client_one_solve()
            self.client_now, self.client_rest, num_change_client, flat = client_increment(self.client_now,
                                                                                          self.client_rest,
                                                                                          self.client_incre_min_perc,
                                                                                          self.client_incre_max_perc,
                                                                                          self.client_incre_prob)
            data_incre_client_num = 0
            experience_client_num = 0
            for c in self.client_now:
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss_init, _ = c.trainer.test(test_data=c.train_data,
                                              test_label=c.train_label)
                loss_incre, _ = c.trainer.test(test_data=c.increment_train_data,
                                               test_label=c.increment_train_label)

                if c.round_increment:
                    data_incre_client_num += 1

                if loss_incre <= loss_init:
                    experience_client_num += 1

            client_now_num = len(self.client_now)
            percentage_client_with_newdata = data_incre_client_num / client_now_num
            user_experience = experience_client_num / client_now_num
            self.stage_run_iter += 1
            if stage_id < self.stage_total and ((self.target_pnew < percentage_client_with_newdata) and \
                                                (user_experience < self.target_ue) or self.stage_run_iter >= 3):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()

            elif stage_id == self.stage_total and \
                    (self.eval_record[-1] >= self.eval_target or
                     self.stage_run_iter >= 2 or
                     max(self.eval_record[-3:]) - min(self.eval_record[-3:]) < self.convergence):
                self.stage_run_iter = 0
                stage_id += 1
                stage_trans = True
                self.record_file.writelines(
                    str(self.eval_record) + '\n' + str(self.loss_record) + '\n' + str(
                        self.true_time_record) + '\n' + str(
                        self.
                            time_record) + '\n' + str(self.
                                                      cost_record) + '\n')
                self.record_file.flush()
        del_list = os.listdir('./temp/')
        for f in del_list:
            file_path = os.path.join('./temp/', f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    print()

    def client_one_solve(self):
        client_from_queue = self.client_queue.get()

        # Local training.
        begin_time = time.time()
        client_from_queue.train()
        client_model = client_from_queue.get_upload_para()

        client_cost = self.sim_zip_for_cc(
            model=client_model,
            filename=client_from_queue.client_name,
        )
        # Upload parameters.
        self.receive_parameter(
            client_name=client_from_queue.client_name,
            client_model=client_model,
            client_aggregation_field=client_from_queue.get_field(),
            client_time=random.uniform(10, 100),
            client_true_time=time.time() - begin_time,
            client_cost=client_cost,

        )

        # Client objects out of the queue are re-queued to form a circular queue.
        self.client_queue.put(client_from_queue)

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            client_time: float,
            client_true_time: float,
            client_cost: float,
    ) -> None:

        # Store the uploaded parameters in the queue.
        print("Server %s receives uploaded parameters from Client %s" % (
            self.server_name, client_name))

        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
                'time'             : client_time,
                'true_time'        : client_true_time,
                'cost'             : client_cost,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            aggregation_parameter: Queue
    ) -> None:

        # Judge whether the conditions for starting aggregation are met.
        if self.run_aggregation(aggregation_parameter):
            # Start aggregation.
            print("Task on Server %s finishes aggregation" % self.server_name)
            eval_falt = self.run_evaluation()

            # Judge the evaluation situation.
            if eval_falt:
                # Stop task.
                print("Task on Server %s stops." % self.server_name)
                self.task_stop = True

            else:
                # Update model.

                print("Task on Server %s updates global model." % self.server_name)
                self.update_clients_model()

    def update_clients_model(self) -> None:

        for task_selected_client in self.client_selected:
            self.round_cost += self.model_para_size
            task_selected_client.download_version = self.current_round()
            task_selected_client.update_local_model(new_global=self.get_model())

    def current_round(self):
        return self.aggregation.aggregation_version

    def add_client_to_pool(
            self,
            client_to_add: StandAloneClient
    ):
        self.client_rest.append(client_to_add)

    def add_client_to_train(
            self,
            client_to_add: StandAloneClient
    ):
        client_to_add.field = self.aggregation.get_field()
        self.round_cost += self.model_para_size
        client_to_add.update_local_model(new_global=self.get_model())
        self.client_selected.append(client_to_add)

    def run_evaluation(self):
        self.eval()

        # Record time and initialize object property.
        self.time_record.append(max(self.round_time))
        self.true_time_record.append(max(self.round_true_time))
        self.round_time = []
        self.round_true_time = []

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.eval_record[
                   -1] >= self.eval_target or self.reach_convergence() or self.current_round() >= self.maxround

    def eval(self):
        if len(self.eval_client) != 0:
            acc_list = []
            loss_list = []
            for c in self.eval_client:
                print(c.client_name)
                c.download_version = self.current_round()
                c.update_local_model(new_global=self.get_model())
                loss, acc = c.trainer.test(test_data=c.train_data, test_label=c.train_label)
                loss_list.append(loss)
                acc_list.append(acc)
            acc_mean = np.mean(acc_list)
            loss_mean = np.mean(loss_list)
        else:
            self.model.eval()

            # if eval_data.shape[-1] == 1:
            #     eval_data = eval_data.reshape(eval_data.shape[0], 1, eval_data.shape[1], eval_data.shape[2])
            eval_dataset = simple_dataset(data=self.eval_data, label=self.eval_label)
            if self.eval_data.shape[0] % self.batch_size == 1:

                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, drop_last=True)
            else:
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size)

            # Client model testing.
            with torch.no_grad():
                eval_loss = 0.0
                eval_acc = 0.0
                eval_count = 0
                eval_total = 0
                for data in eval_dataloader:
                    self.model.to(self.device)
                    input = data[1].float().to(self.device)
                    label = data[0].float().to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label.long())
                    eval_loss += loss.item()
                    _, pre = torch.max(output.data, dim=1)
                    eval_acc += (pre == label).sum().item()
                    eval_count += 1
                    eval_total += label.shape[0]

                loss_mean = eval_loss / eval_count
                acc_mean = eval_acc / eval_total * 100
                print('Evaluation in Round %d:    Loss: %.4f       Accuracy: %.2f' % (
                    self.current_round(), loss_mean, acc_mean))
        self.eval_record.append(acc_mean)
        self.loss_record.append(loss_mean)

    def run_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            if self.synchronous:
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    self.round_time.append(temp['time'])
                    self.round_true_time.append(temp['true_time'])
                    self.round_cost += temp['cost']
                    aggregation_parameter.put(temp)
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )
            else:
                temp_time = []
                temp_para = []
                for i in range(aggregation_parameter.qsize()):
                    temp = aggregation_parameter.get()
                    temp_time.append(temp['time'])
                    temp_para.append(temp)
                sorted_id = sorted(range(len(temp_time)), key=lambda k: temp_time[k], reverse=False)
                choose_id = sorted_id[0:self.aggregation.min_to_start]
                no_choose_id = sorted_id[self.aggregation.min_to_start:]
                real_aggregation_parameter = Queue()
                for i in choose_id:
                    self.round_time.append(temp_para[i]['time'])
                    self.round_true_time.append(temp_para[i]['true_time'])
                    self.round_cost += temp_para[i]['cost']
                    real_aggregation_parameter.put(temp_para[i])
                for i in no_choose_id:
                    temp_para[i]['time'] = temp_para[i]['time'] - max(self.round_time)
                    temp_para[i]['true_time'] = temp_para[i]['true_time'] - max(self.round_true_time)
                    aggregation_parameter.put(temp_para[i])
                new_model = self.aggregation.aggregate(
                    {
                        'current_w': self.get_model(),
                        'parameter': real_aggregation_parameter,
                        'record'   : self.eval_record
                    }
                )

            # Update global model weight.
            self.update_model(new_model=new_model)
            for c in self.client_now:
                c.data_increment()

            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    ) -> bool:

        if self.synchronous:
            # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                    self.client_selected):
                return True

            else:
                return False

        else:
            if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= self.ration:
                return True

            else:
                return False

    def get_model(self):
        with torch.no_grad():
            weight = [p.data.clone().detach() for p in self.model.parameters()]
        return deepcopy_list(weight)

    def update_model(
            self,
            new_model: list
    ):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), new_model):
                p.data.copy_(d.data)

    def reach_convergence(self):
        # Judgment begins after ten rounds
        if len(self.eval_record) >= self.convergence_round:
            # Convergence in the range of 10 rounds.
            return max(self.eval_record[-10:]) - min(self.eval_record[-10:]) < self.convergence

        else:
            return False

    def clients_increment(self):
        for c in self.client_selected:
            c.data_increment()

    def clients_update_increment(self):
        for c in self.client_selected:
            c.update_client_increment_data()

    def sim_zip_for_cc(
            self,
            model: object = None,
            filename: str = '',
    ) -> int:
        """

        In order to simulate and calculate the communication cost during stand-alone simulation, model parameter file is saved and converted into a compressed package and then the size is calculated.

        Args:
            model (object): Model object to process.
            filename (str): Path to model parameter file.
            type (str): The type of model.

        Returns:
            sim_zip_cc (int): Communication cost value calculated by simulation.

        """

        # Judge whether to simulate the communication cost, otherwise return 0 directly.
        try:
            os.remove('./temp/' + filename + '.zip')
        except:
            print()
        torch.save(model, './temp/' + filename + '.pth')
        zip_model('./temp/' + filename + '.pth', './temp/' + filename + '.zip')
        sim_zip_cc = len(read_zip('./temp/' + filename + '.zip'))

        return sim_zip_cc
