# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-09 22:32
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-09 22:32
import random
import numpy as np
import torch


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False