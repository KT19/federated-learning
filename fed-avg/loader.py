#-*-coding:utf-8-*-
import numpy as np
import torch

def prob_normalize(x):
    sum_exp = np.sum(np.exp(x))
    return np.exp(x)/sum_exp

def fed_loader(traindata, num_splits, batch_size, mode="iid"):
    if mode == "iid":
        split_info = [int(len(traindata) / num_splits) for _ in range(num_splits)]

    elif mode == "non_iid":
        split_prob = prob_normalize(np.random.normal(loc=3.0, scale=2, size=(num_splits-1))) #non i.i.d.
        split_info = []
        total_size = 0
        for p in split_prob:
            split_info.append(int(len(traindata)*p))
            total_size += int(len(traindata)*p)
        split_info.append(len(traindata)-total_size)

    data_split = torch.utils.data.random_split(traindata, split_info)
    dataloaders = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in data_split]

    return dataloaders
