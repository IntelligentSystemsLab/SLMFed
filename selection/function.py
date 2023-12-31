# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 19:18
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 19:18

import random
import numpy as np

from utils.data import deepcopy_list, list_normalization


def random_select(
        client_list: list,
        client_selected_probability: list,
        select_percentage: float = 0
) -> list:
    """

    Random client selection without limit on quantity.

    Args:
        client_list (list): Total client list.
        client_selected_probability (list): The probability of being selected for each client.

    Returns:
        List: List of selected clients.

    """

    return_client = []
    for i in range(len(client_list)):
        p = client_selected_probability[i]
        if random.random() < p:
            return_client.append(client_list[i])
    if len(return_client) == 0:
        client_selected_probability_max = max(client_selected_probability)
        id = client_selected_probability.index(client_selected_probability_max)
        return_client.append(client_list[id])

    return return_client


def random_select_with_percentage(
        client_list: list,
        client_selected_probability: list,
        select_percentage: float
) -> list:
    """

    Random client selection based on a specified percentage.

    Args:
        client_list (list): Total client list.
        client_selected_probability (list): The probability of being selected for each client.
        select_percentage (float): Percentage of selected clients.

    Returns:
        List: List of selected clients.

    """

    return_client = []
    rest_client_list = deepcopy_list(client_list)
    rest_client_selected_probability = deepcopy_list(client_selected_probability)
    select_num = int(select_percentage * len(client_list))
    for i in range(select_num):
        item = random_pick(rest_client_list, rest_client_selected_probability)
        return_client.append(rest_client_list[item])
        rest_client_list.pop(item)
        rest_client_selected_probability.pop(item)

    return return_client


def rank_select_with_percentage(
        client_list: list,
        client_selected_probability: list,
        select_percentage: float
) -> list:
    """

    Ranked client selection based on a specified percentage.

    Args:
        client_list (list): Total client list.
        client_selected_probability (list): The probability of being selected for each client.
        select_percentage (float): Percentage of selected clients.

    Returns:
        List: List of selected clients.

    """

    return_client = []
    rank = np.argsort(client_selected_probability)
    select_num = int(select_percentage * len(client_list))
    for i in range(select_num):
        return_client.append(client_list[rank[-1 - i]])

    return return_client


def random_pick(
        obj: list,
        prob: list
) -> int:
    """

    Randomly select an object from the list and get its index number.

    Args:
        obj (list): Source list.
        prob (list): The corresponding probability of objects.

    Returns:
        Int: The selected index number.

    """

    r = sum(prob) * random.random()
    s = 0.0
    for i in range(len(obj)):
        s += prob[i]
        if r <= s:
            return i

    return -1


def softmax_prob_from_indicators(indicators: list) -> list:
    """

    Convert each index value into probability with softmax.

    Args:
        indicators (list): List of indicators.

    Returns:
        List: List of probabilities after softmax.

    """

    # Judge whether the size of each indicator is the same.
    if all(len(indicators[0]) == len(l) for l in indicators):
        length = len(indicators[0])
        indicators_norm_sum = [0 for i in range(length)]
        for ind_ in indicators:
            ind = list_normalization(ind_)
            ind_e = [np.exp(i) for i in ind]
            ind_e_sum = sum(ind_e)
            indicators_norm_sum = [indicators_norm_sum[i] + ind_e[i] / ind_e_sum for i in range(length)]
        len_ind = len(indicators)
        result = [i / len_ind for i in indicators_norm_sum]
        return result

    else:
        exit(1)
