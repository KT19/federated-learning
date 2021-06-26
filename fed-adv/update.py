#-*-coding:utf-8-*-
import numpy as np
import torch
import copy

def fed_avg(models):
    """
    args:
    models(list): models state_dict aggregated from local client

    return:
    weight_avg (state_dict)
    """
    weight_avg = copy.deepcopy(models[0])
    for k in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[k] += models[i][k]

        weight_avg[k] = torch.div(weight_avg[k], len(models))

    return weight_avg
