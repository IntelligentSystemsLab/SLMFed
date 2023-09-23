# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-05 2:04
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-05 2:04

from queue import Queue
import numpy as np
from aggregation.function import meta
from aggregation.base import BaseFed
from selection.function import softmax_prob_from_indicators


class SynMeta(BaseFed):

    def __init__(
            self,
            lr: float,
            min_to_start: int = 2,
            weight: str = False
    ) -> None:
        # Super class init.
        super().__init__(
            name='meta',
            synchronous=True,
            min_to_start=min_to_start,
            meta=True
        )
        self.lr = lr
        self.weight = weight

    def aggregate(
            self,
            datadict: {
                'current_w': list,
                'parameter': Queue,
                'record': list
            }
    ) -> list:
        # Create temporary variables
        grads = []
        client_datasize = []
        client_incredatasize = []
        client_incredataprob = []
        client_information_richness = []
        # Get the specified data.
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            grads.append(temp['model'])
            if self.weight == 'informationRichness':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
            elif self.weight == 'increDataSize':
                client_incredatasize.append(temp['aggregation_field']['increDataSize'])
            elif self.weight == 'increDataProb':
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])
            elif self.weight == 'ir_ids':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_incredatasize.append(temp['aggregation_field']['increDataSize'])
            elif self.weight == 'ir_idp':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])
            elif self.weight == 'ir_ds':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_datasize.append(temp['aggregation_field']['dataSize'])

                # Get weight of clients for aggregation.
        if self.weight == 'informationRichness':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness])
        elif self.weight == 'increDataSize':
            aggregate_percentage = softmax_prob_from_indicators([client_incredatasize])
        elif self.weight == 'increDataProb':
            aggregate_percentage = softmax_prob_from_indicators([client_incredataprob])
        elif self.weight == 'ir_ids':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_incredatasize])
        elif self.weight == 'ir_idp':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_incredataprob])
        elif self.weight == 'ir_ds':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_datasize])
        else:
            aggregate_percentage = None
        # Calling aggregation function.
        current_global_w = meta(grads=grads, current_w=datadict['current_w'], lr=self.lr, weight=aggregate_percentage)

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Data size of client
        if self.weight == 'informationRichness':
            return ['informationRichness']
        elif self.weight == 'increDataSize':
            return ['increDataSize']
        elif self.weight == 'increDataProb':
            return ['increDataProb']
        elif self.weight == 'ir_ids':
            return ['informationRichness','increDataSize']
        elif self.weight == 'ir_idp':
            return ['informationRichness','increDataProb']
        elif self.weight == 'ir_ds':
            return ['informationRichness', 'dataSize']
        else:
            return []


class AsynMeta(BaseFed):

    def __init__(
            self,
            lr: float,
            min_to_start: int,
            weight: str
    ) -> None:
        # Super class init.
        super().__init__(
            name='meta_asyn',
            synchronous=False,
            min_to_start=min_to_start,
            meta=True
        )
        self.lr = lr
        self.weight = weight

    def aggregate(
            self,
            datadict: {
                'current_w': list,
                'parameter': Queue,
                'record': list
            }
    ) -> list:
        # Create temporary variables
        grads = []
        client_version = []
        client_incredataprob = []
        client_information_richness = []
        # Get the specified data.
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            grads.append(temp['model'])
            if self.weight == 'informationRichness':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
            elif self.weight == 'increDataProb':
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])
            elif self.weight == 'version':
                client_version.append(np.exp(-(self.aggregation_version-temp['aggregation_field']['version'])))
            elif self.weight == 'ir_idp':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])
            elif self.weight == 'ir_ver':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_version.append(np.exp(-(self.aggregation_version-temp['aggregation_field']['version'])))
            elif self.weight == 'idp_ver':
                client_version.append(np.exp(-(self.aggregation_version-temp['aggregation_field']['version'])))
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])
            elif self.weight == 'ir_idp_ver':
                client_information_richness.append(temp['aggregation_field']['informationRichness'])
                client_version.append(np.exp(-(self.aggregation_version-temp['aggregation_field']['version'])))
                client_incredataprob.append(temp['aggregation_field']['increDataProb'])

                # Get weight of clients for aggregation.
        if self.weight == 'informationRichness':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness])
        elif self.weight == 'increDataProb':
            aggregate_percentage = softmax_prob_from_indicators([client_incredataprob])
        elif self.weight == 'version':
            aggregate_percentage = softmax_prob_from_indicators([client_version])
        elif self.weight == 'ir_idp':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_incredataprob])
        elif self.weight == 'ir_ver':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_version])
        elif self.weight == 'idp_ver':
            aggregate_percentage = softmax_prob_from_indicators([client_version, client_incredataprob])
        elif self.weight == 'ir_idp_ver':
            aggregate_percentage = softmax_prob_from_indicators([client_information_richness,client_version, client_incredataprob])
        else:
            aggregate_percentage = None
        # Calling aggregation function.
        current_global_w = meta(grads=grads, current_w=datadict['current_w'], lr=self.lr, weight=aggregate_percentage)

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Data size of client
        if self.weight == 'informationRichness':
            return ['informationRichness']
        elif self.weight == 'increDataProb':
            return ['increDataProb']
        elif self.weight == 'version':
            return ['version']
        elif self.weight == 'ir_idp':
            return ['informationRichness','increDataProb']
        elif self.weight == 'ir_ver':
            return ['informationRichness','version']
        elif self.weight == 'idp_ver':
            return ['version','increDataProb']
        elif self.weight == 'ir_idp_ver':
            return ['informationRichness','version','increDataProb']
        else:
            return []
