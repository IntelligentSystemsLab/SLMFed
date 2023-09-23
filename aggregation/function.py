# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 15:32
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 15:32
from copy import deepcopy
from typing import Callable
import numpy as np
import math

import torch

from utils.data import deepcopy_list


def fedavg(
        weight: list,
        data_size: list
) -> list:
    """

    Function implementation of FedAVG, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (list): List of models to aggregate.
        data_size (list): List of data sizes of clients.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    return_result = []

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):

        # Calculate parameters for all clients.
        for i in range(client_num):

            weight_array = np.array(weight[i][l])
            content = weight_array * data_size[i] / sum(data_size)
            if first[l] == 0:
                return_result.append(content)
                first[l] = 1
            else:
                return_result[l] += content
        return_result[l] = torch.tensor(return_result[l])
    return return_result


def e3cs(
        weight: list,
        E3CS_weight: list,
        current_weight: list
) -> list:
    """

    Function implementation of FedAVG, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (list): List of models to aggregate.
        data_size (list): List of data sizes of clients.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    return_result = []

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):

        # Calculate parameters for all clients.
        for i in range(client_num):

            weight_array = np.array(weight[i][l])
            content = weight_array * E3CS_weight[i] / sum(E3CS_weight)
            if first[l] == 0:
                return_result.append(content)
                first[l] = 1
            else:
                return_result[l] += content
        return_result[l] = 0.5*return_result[l]+np.array(current_weight[l]) * 0.5
        return_result[l] = torch.tensor(return_result[l])
    return return_result


def soteriafl(
        weight: list,
        initial_reference: list,
        gamma
) -> list:
    """

    Function implementation of FedAVG, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (list): List of models to aggregate.
        data_size (list): List of data sizes of clients.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    return_result = []

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):

        # Calculate parameters for all clients.
        for i in range(client_num):

            weight_array = np.array(weight[i][l])
            content = weight_array / client_num
            if first[l] == 0:
                return_result.append(content)
                first[l] = 1
            else:
                return_result[l] += content
        temp = return_result[l]
        return_result[l] = (initial_reference[l]+temp)/2
        return_result[l] = torch.tensor(return_result[l])
        initial_reference[l] = (1-gamma)*initial_reference[l] + gamma * temp
    return return_result, initial_reference


def fedlama(
        weight: list,
        omiga,
        current_w
) -> list:
    """

    Function implementation of FedLAMA, which directly averages the corresponding values of collected model parameters.

    Args:
        weight (list): List of models to aggregate.
        data_size (list): List of data sizes of clients.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    avg_result = []
    return_result = []

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):

        # Calculate parameters for all clients.
        for i in range(client_num):

            weight_array = np.array(weight[i][l])
            content = weight_array / client_num
            if first[l] == 0:
                avg_result.append(content)
                first[l] = 1
            else:
                avg_result[l] += content

    # Update the parameters for each layer of the model separately.
    d_list = []
    dim_list = []
    for l in range(layer_num):
        temp = 0
        # Calculate parameters for all clients.
        for i in range(client_num):
            weight_array = np.array(weight[i][l])
            temp += np.linalg.norm(avg_result[l] - weight_array) ** 2
        d = temp / client_num / 10 / len(avg_result[l].shape)
        d_list.append(d)
        dim_list.append(len(avg_result[l].shape))
    d_dim = []
    for l in range(layer_num):
        d_dim.append(d_list[l] * dim_list[l])
    for l in range(layer_num):
        if l == layer_num - 1:
            delta = sum(d_dim) / sum(d_dim)
            numda = sum(d_list) / sum(d_list)
        else:
            delta = sum(d_dim[:l + 1]) / sum(d_dim)
            numda = sum(d_list[:l + 1]) / sum(d_list)
        if delta < (1 - numda):
            return_result.append(
                torch.tensor(omiga * delta * avg_result[l] + (1 - omiga * delta) * np.array(current_w[l])))
        else:
            return_result.append(torch.tensor(avg_result[l]))
    return return_result


def fedfd(
        client_id: list,
        weight: dict,
        client_round: dict,
        version_latest: int,
) -> list:
    """

    Function implementation of FedFD, which weighted averages the corresponding values of collected model parameters.

    Args:
        client_id (list): ID of clients that upload the models.
        weight (dict): Corresponding dictionary of client IDs and models to aggregate.
        client_round (dict): Corresponding dictionary of client IDs and number of training rounds for local models.
        version_latest (int): Latest model version.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Use a loop to calculate the weighted sum of each model parameter value, and then take the average.
    total = 0
    w = 0
    for c in client_id:
        total += (version_latest - client_round[c] + 1) ** (-0.5)
        try:
            weight_c = weight[c]
        except:
            continue
        theta = ((version_latest - client_round[c] + 1) ** (-0.5))
        w += theta * np.array(weight_c)
    global_model = w / total
    return_result = global_model.tolist()

    return return_result


def fedasync(
        client_id: list,
        weight: dict,
        staleness: str,
        current_weight: list,
        current_round: int,
        client_round: dict,
        alpha: float,
        beta: float
) -> list:
    """

    Args:
        client_id (list): List of uploaded client names.
        weight (dict): Dict of uploaded local model weight.
        staleness (str): Corresponds to the name of the function defined in FedAsync.
        current_weight (list): Current global model parameters.
        current_round (int): Number of current training round.
        client_round (dict): Number of global round corresponding to the model trained by each client.
        alpha (float): Corresponds to the parameter α defined in FedAsync.
        beta (float): Corresponds to the parameter β defined in FedAsync.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    alpha_clients = []
    weight_list = deepcopy_list(current_weight)
    return_result = deepcopy_list(current_weight)
    first = True
    layer_num = len(current_weight)

    # For all uploaded clients, model parameters are weightily summarized.
    for c_id in client_id:
        c_weight = deepcopy_list(weight[c_id])
        c_round = client_round[c_id]
        if staleness == 'Linear':
            s = 1 / (alpha * (current_round - c_round) + 1)
        elif staleness == 'Polynomial':
            s = math.pow(current_round - c_round + 1, (-alpha))
        elif staleness == 'Exponential':
            s = math.exp(-alpha * (current_round - c_round))
        elif staleness == 'Hinge':
            if current_round - c_round <= beta:
                s = 1
            else:
                s = 1 / (alpha * (current_round - c_round - beta))
        else:
            s = 1
        alpha_c = s * alpha
        alpha_clients.append(alpha_c)
        for l in range(layer_num):
            if first:
                weight_list[l] = c_weight[l] * alpha_c

            else:
                weight_list[l] += c_weight[l] * alpha_c
        first = False

    # The summarized model parameters are averaged to obtain the global model parameters.
    avg_alpha = sum(alpha_clients) / len(alpha_clients)
    for l in range(len(current_weight)):
        return_result[l] = (1 - avg_alpha) * current_weight[l] + avg_alpha * weight_list[l] / sum(alpha_clients)

    return return_result


def SLMFedsyn(
        weight: list,
        aggregate_percentage: list,
        current_weight: list
) -> list:
    """

    Args:
        weight (list): List of client model parameters for aggregation.
        aggregate_percentage (list): Aggregate weights for each client.
        current_weight (list): Current global model parameters.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    aggregate_percentage_array = np.array(aggregate_percentage)
    client_num = len(weight)
    layer_num = len(weight[0])
    first = [0 for i in range(layer_num)]
    return_result = deepcopy_list(current_weight)

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):
        none_client = []

        # Calculate parameters for all clients.
        for i in range(client_num):

            if weight[i][l] is None:
                none_client.append(i)
                continue
            else:
                weight_array = np.array(weight[i][l])
                content = weight_array * aggregate_percentage_array[i]
                if first[l] == 0:
                    return_result[l] = content
                    first[l] = 1
                else:
                    return_result[l] += content

        # Adjust the parameters according to the proportion of None
        if 0 < len(none_client) < client_num:
            none_prob = 0
            for i in none_client:
                none_prob += aggregate_percentage_array[i]
            return_result[l] = return_result[l] / (1 - none_prob)
        return_result[l] = torch.tensor(return_result[l])
    return return_result


def SLMFedasyn(
        client_id: list,
        weight: dict,
        aggregate_percentage: dict,
        current_weight: list,
        current_acc: float,
        target_acc: float,
        func: str
) -> list:
    """

    Args:
        client_id (list): List of client IDs for aggregation.
        weight (dict): Dictionary of client model parameters for aggregation.
        aggregate_percentage (dict): Aggregate weights for each client.
        current_weight (list): Current global model parameters.
        current_acc (float): Current accuracy corresponding to the global model.
        target_acc (float): Target accuracy of the task.
        func (str):  Function to adjust aggregation weights. Default as 'other'.

    Returns:
        List: The model generated after aggregation. And use a list to store the parameters of different layers.

    """

    # Initialize temporary variables.
    layer_num = len(weight[client_id[0]])
    return_result = deepcopy_list(current_weight)
    upload_content = deepcopy_list(current_weight)
    first = [0 for i in range(layer_num)]
    if func == 'linear':
        alpha = current_acc / target_acc
    elif func == 'concave_exp':
        alpha = 1 - math.exp(-2 * math.e * current_acc / target_acc)
    elif func == 'convex_quadratic':
        alpha = (current_acc / target_acc) ** 2
    elif func == 'concave_quadratic':
        alpha = 1 - (current_acc / target_acc - 1) ** 2
    elif func == 'convex_exp':
        alpha = (math.exp(current_acc / target_acc) - 1) / (math.e - 1)
    else:
        alpha = 0.5

    # Update the parameters for each layer of the model separately.
    for l in range(layer_num):
        p_sum = 0

        # Calculate parameters for all clients.
        for id in client_id:
            if weight[id][l] is None:
                continue
            else:
                p = aggregate_percentage[id]
                p_sum += p
                content = weight[id][l] * p
                if first[l] == 0:
                    upload_content[l] = content
                    first[l] = 1
                else:
                    upload_content[l] += content

        # Adjust parameters according to established rules.
        if p_sum != 0:
            q_new = (1 - p_sum) * alpha
            new_content = upload_content[l] / p_sum
            old_content = return_result[l]
            return_result[l] = new_content * (1 - q_new) + old_content * q_new

    return return_result


# def meta(
#         grads: list,
#         current_w: list,
#         lr: float
# ):
#     # adam
#     num = len(grads)
#     g = []
#     outer_opt = Adam(lr=lr)
#     new_w=deepcopy(current_w)
#     for i in range(len(grads[0])):
#         grad_sum = torch.zeros_like(grads[0][i])
#         for ic in range(num):
#             grad_sum += grads[ic][i]
#         grad_avg=grad_sum/num
#         g.append(grad_avg)
#     outer_opt.increase_n()
#     for i in range(len(new_w)):
#         outer_opt(new_w[i], g[i], i=i)
#     return new_w

def meta(
        grads: list,
        current_w: list,
        lr: float,
        weight: list
):
    num = len(grads)
    grad_avg_list = []
    new_w = deepcopy(current_w)
    for i in range(len(grads[0])):
        grad_sum = np.zeros_like(grads[0][i])
        for ic in range(num):
            temp = np.nan_to_num(grads[ic][i])
            if weight is None:
                grad_sum += temp
            else:
                grad_sum += temp * weight[ic]
        if weight is None:
            grad_avg = grad_sum / num
        else:
            grad_avg = grad_sum
        grad_avg_list.append(grad_avg)
    for p, g in zip(new_w, grad_avg_list):
        p.data.add_(torch.from_numpy(g).data, alpha=-lr)
    return new_w


class Adam:

    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        """

        :param lr:
        :param betas:
        :param eps:
        """
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()
        self.v = dict()
        self.n = 0
        self.creted_momtem_grad_index = set()

    def __call__(self, params, grads, i):
        # 创建对应的 id
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        self.n += 1
