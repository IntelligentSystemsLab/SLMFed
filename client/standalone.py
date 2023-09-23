# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 9:46
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 9:46
import random
import numpy as np
from numpy import ndarray
from trainer.base import BaseTrainer
from utils.data import calculate_IW


class StandAloneClient(object):

    def __init__(
            self,
            client_name: str,
            train_data: ndarray,
            train_label: ndarray,
            test_data: ndarray = None,
            test_label: ndarray = None,
            trainer: BaseTrainer = None,
            data_incre_min_perc: float = 0.0,
            data_incre_max_perc: float = 0.0,
            data_incre_prob: float = 0.5,
            incremental: bool = False,
            incremental_shuffle: bool = False,
            initial_rate: float = 0.5,

    ) -> None:

        # Initialize object properties.
        self.client_name = client_name
        self.total_shape = train_data.shape[0]
        if incremental:
            self.total_train_data = train_data
            self.total_train_label = train_label
            # Initial data is intercepted in part by default in the order of the overall data.
            initial_part = int(initial_rate * self.total_shape)
            self.increment_train_data = self.total_train_data[0:initial_part]
            self.train_data = self.total_train_data[0:initial_part]
            self.increment_train_label = self.total_train_label[0:initial_part]
            self.train_label = self.total_train_label[0:initial_part]
            self.new_data = []
            self.new_label = []
            self.incremental_shuffle = incremental_shuffle
        else:
            self.train_data = train_data
            self.train_label = train_label
        if test_data is None:
            self.train_test_same = True
            self.test_data = train_data
            self.test_label = train_label
        else:
            self.train_test_same = False
            self.test_data = test_data
            self.test_label = test_label
        self.data_incre_min_perc = data_incre_min_perc
        self.data_incre_max_perc = data_incre_max_perc
        self.data_incre_prob = data_incre_prob
        self.trainer = trainer
        self.field = []
        self.download_version = 0
        self.round_increment = False
        self.datasize = self.train_data.shape[0]
        self.incre_index_remark = 0
        self.incresize = 0

    def init_trainer(
            self,
            trainer: BaseTrainer
    ) -> None:

        self.trainer = trainer

    def train(self) -> None:
        if self.trainer.meta:
            if self.trainer.meta_mode == 'Reptile':
                self.trainer.train(
                    support_data=self.train_data,
                    support_label=self.train_label,
                    query_data=self.train_data,
                    query_label=self.train_label,
                )
            else:
                support_data, support_label, query_data, query_label = self.get_meta_set()
                self.trainer.train(
                    support_data=support_data,
                    support_label=support_label,
                    query_data=query_data,
                    query_label=query_label
                )

        else:

            self.trainer.train(
                train_data=self.train_data,
                train_label=self.train_label
            )

    def test(self, ) -> None or float:

        if self.trainer.meta:
            # support_data, support_label, query_data, query_label = self.get_meta_set()
            # return self.trainer.test(
            #     support_data=support_data,
            #     support_label=support_label,
            #     query_data=query_data,
            #     query_label=query_label
            # )
            return self.trainer.test(
                support_data=self.train_data,
                support_label=self.train_label,
                query_data=self.train_data,
                query_label=self.train_label
            )
        else:
            return self.trainer.test(
                test_data=self.train_data,
                test_label=self.train_label
            )

    def predict(
            self,
            data: ndarray
    ) -> ndarray:

        # Call the trainer to perform model prediction.
        return self.trainer.predict(data=data)

    def get_model(self) -> list:

        # Call the trainer to get the current local model weight.
        return self.trainer.get_model()

    def get_upload_para(self) -> list:

        # Call the trainer to get the current uploaded parameter.
        return self.trainer.get_upload_para()

    def update_local_model(
            self,
            new_global: list
    ) -> None:

        self.trainer.update_local_model(new_global=new_global)

    def get_field(self) -> dict:

        # Initialize dictionary.
        field = dict()

        # Fill data to the dictionary.
        for f in self.field:
            # Judge field name.
            if f == 'clientRound':
                # Number of local training rounds.
                field['clientRound'] = self.trainer.trained_num

            elif f == 'informationRichness':
                # Information richness of local data.
                field['informationRichness'] = calculate_IW(self.train_label)

            elif f == 'dataSize':
                # Size of local data.
                field['dataSize'] = self.train_label.shape[0]

            elif f == 'increDataSize':
                # Size of local data.
                field['increDataSize'] = self.incresize

            elif f == 'increDataProb':
                # Size of local data.
                field['increDataProb'] = self.incresize/self.train_label.shape[0]

            elif f == 'version':
                # Size of local data.
                field['version'] = self.download_version

        return field

    def data_increment(self) -> None:

        # Judge whether to perform data increment
        if random.random() < self.data_incre_prob:
            # Get a random percentage of data increment.
            percent_new = random.uniform(self.data_incre_min_perc, self.data_incre_max_perc)

            # Get incremental content and update variables.
            num_new = int(percent_new * self.total_shape)
            if self.incremental_shuffle:
                index_all = [i for i in range(self.total_shape)]
                index_new = np.random.choice(index_all, size=num_new, replace=False)
            else:
                index_new = [i + self.incre_index_remark for i in range(num_new)]
                if len(index_new) > 0:
                    for i in range(num_new):
                        if index_new[i] >= self.total_shape:
                            index_new[i] = index_new[i] - self.total_shape
                    self.incre_index_remark = index_new[-1] + 1
            data_new = self.total_train_data[index_new]
            label_new = self.total_train_label[index_new]
            if len(self.new_data) == 0:
                self.new_data = np.array(data_new)
                self.new_label = np.array(label_new)
            else:
                self.new_data = np.concatenate((self.new_data, data_new))
                self.new_label = np.concatenate((self.new_label, label_new))
            self.increment_train_data = np.concatenate((self.increment_train_data, data_new))
            self.increment_train_label = np.concatenate((self.increment_train_label, label_new))
            self.round_increment = True

    def update_client_increment_data(self) -> None:
        self.incresize = self.increment_train_label.shape[0] - self.train_label.shape[0]
        self.train_data = self.increment_train_data
        self.train_label = self.increment_train_label
        self.datasize = self.increment_train_label.shape[0]
        self.new_data = []
        self.new_label = []

    def get_meta_set(self):
        if self.train_test_same:
            split = int(self.datasize / 2)
            support_data = self.train_data[0:split]
            support_label = self.train_label[0:split]
            query_data = self.train_data[split:]
            query_label = self.train_label[split:]
            return support_data, support_label, query_data, query_label
        else:
            return self.train_data, self.train_label, self.test_data, self.test_label

    def update_server_corr(self, server_corr):
        if self.trainer.meta:
            self.trainer.server_corr = server_corr
