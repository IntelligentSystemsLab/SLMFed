# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/7/29 15:45
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/7/29 15:45
import random
import numpy as np

from utils.data import deepcopy_list


def client_increment(client_list_now, client_list_rest, min_perc, max_perc, prob):
    shape_now = len(client_list_now)
    shape_rest = len(client_list_rest)
    percent_new = random.uniform(min_perc, max_perc)
    num_change = int(percent_new * shape_now)
    if random.random() < prob:
        flat = 1
        num_new = num_change
        client_new=[]
        if num_new <= shape_rest:
            index_all = [i for i in range(shape_rest)]
            index_new = list(np.random.choice(index_all, size=num_new, replace=False))
            index_new.sort()
            for i in range(len(index_new)):
                id=-i-1
                client_new .append(client_list_rest[index_new[id]])
                client_list_rest.pop(index_new[id])
        else:
            num_change = shape_rest
            client_new = client_list_rest
            client_list_rest = []
        for c in client_new:
            c.round_increment = True
        client_list_now = deepcopy_list(client_list_now) + deepcopy_list(client_new)

    else:
        flat = 0
        num_leave = num_change
        if num_leave <= shape_now:
            index_all = [i for i in range(shape_now)]
            index_leave = np.random.choice(index_all, size=num_leave, replace=False)
            client_leave = client_list_now[index_leave]
            client_list_now.pop(index_leave)
        else:
            num_change = 0
            client_leave = []

        client_list_rest = deepcopy_list(client_list_rest) + deepcopy_list(client_leave)

    return deepcopy_list(client_list_now), deepcopy_list(client_list_rest), num_change, flat
