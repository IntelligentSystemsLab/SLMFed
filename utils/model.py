# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-09 22:06
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-09 22:06
import random

from torch import nn
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from utils.data import deepcopy_list


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def compute_rda_alllayers(layer_num, model, data_stimulus, distance_measure,device):
    global_rdm_alllayer = []
    layer_idx=0
    for name,module in model.named_modules():
        if name[:4]=='conv' or name[:2]=='fc':
            layer_idx+=1
            layer_model = torch.nn.Sequential(*list(model.children())[:layer_idx + 1]).to(device)
            feature_all = []
            len_sti = len(data_stimulus)
            with torch.no_grad():
                for j in range(len_sti):
                    x_stimulus = torch.tensor(data_stimulus[j], dtype=torch.float32).to(device)
                    x_stimulus = x_stimulus.unsqueeze(0)  # Add batch dimension
                    try:
                        feature = layer_model(x_stimulus)
                        feature = feature.view(-1, feature.numel())
                        feature_all.append(feature.cpu().numpy())
                    except:
                        feature_all.append(np.zeros((2,2)))
            feature_all = np.array(feature_all)
            rdms_one_layer = squareform(pdist(feature_all.reshape(len_sti, -1), distance_measure))
            result = [np.nan_to_num(rdms_one_layer[0])]
        elif name[:2]=='bn':
            layer_idx += 1
            result = [0 for _ in range(4)]
        elif name=='flatten'or 'pool' in name or name[:7]=='dropout':
            layer_idx += 1
            result = [0]
        elif name=='':
            continue
        else:
            layer_idx += 1
            result = [0]

        global_rdm_alllayer += result
    return global_rdm_alllayer

def compute_rc_simp(rda1, rda2):
    pccs = pearsonr(rda1, rda2)
    return pccs


def choose_layer(w_latest, prob_list):
    return_w = deepcopy_list(w_latest)
    temp = True
    for i in range(len(return_w)):
        if prob_list[i] == 999:
            if not temp:
                return_w[i] = None
        else:
            p = random.random() - 0.5
            if p > prob_list[i]:
                return_w[i] = None
                temp = False
            else:
                temp = True

    return return_w
