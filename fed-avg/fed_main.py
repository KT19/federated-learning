#-*-coding:utf-8-*-
import numpy as np
import torch
import torchvision

from modules import Model
from central import CentralServer
from config import args, test_transform

def main():
    #set test loader
    testdata = torchvision.datasets.CIFAR10("~/dataset", train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=False)

    #set server
    server = CentralServer(Model(3, 10))
    server.build_client(args.num_clients)

    #train server
    server.train(args.num_rounds, test_loader)

if __name__ == "__main__":
    main()
