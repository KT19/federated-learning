#-*-coding:utf-8-*-
import os
import numpy as np
import random
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from update import fed_avg
from loader import fed_loader
from client import Client
from config import args, train_transform, device

class CentralServer(object):
    def __init__(self, global_model):
        """
        args:
        global_model: the model to be trained
        clients(list): client object
        """
        self.global_model = global_model.train().to(device)
        os.makedirs(args.save_dir, exist_ok=True)

    def build_client(self, num_clients):
        print("Now connecting each client..")
        self.num_clients = num_clients

        #train loder
        #Divide dataset based on client nums
        traindata = torchvision.datasets.CIFAR10("~/dataset", train=True, transform=train_transform)
        trainloaders = fed_loader(traindata, args.num_clients, args.batch_size, mode="iid") #i.i.d.

        self.clients = [Client(trainloader, i) for i,trainloader in enumerate(trainloaders)]

    @staticmethod
    def random_sampling_schduler(client_lists, idx):
        """
        args:
        client_lists: all clients
        idx: number of clients to be selected
        """
        return random.sample(client_lists, idx)

    def train(self, total_round, testloader, sampling_schduler=random_sampling_schduler.__func__):
        """
        args:
        total_round(int): total round of communications
        testloader(optional): if given, evaluate model per round
        sampling_schduler(function, optional): sampling scheduler for each client (default is random sampling)
        """
        all_clients = np.arange(self.num_clients).tolist()
        LOG_ACC = []
        LOG_ROUND = []

        for round in range(1, total_round+1, 1):
            print("Round: [{}] / [{}]".format(round, total_round))
            selected_client = sampling_schduler(all_clients, args.num_selects)

            print("selected clients: {}".format(selected_client))
            weights = [] #store trained model

            for idx in selected_client:
                print("Send model to client {}".format(idx))
                w = self.clients[idx].train(self.global_model, args.local_update_epoch)
                weights.append(w)
                print("Recieve model from client {}".format(idx))

            print("Update global model")
            aggregated_weight_state_dict = fed_avg(weights)
            self.global_model.load_state_dict(aggregated_weight_state_dict)

            acc = self.evaluate(testloader)
            LOG_ACC.append(acc)
            LOG_ROUND.append(round)
            df = pd.DataFrame({
            "ROUND": LOG_ROUND,
            "ACCURACY": LOG_ACC})
            df.to_csv(args.save_dir+"/log.csv")


        torch.save(self.global_model, args.save_dir+"/model.pth")

    def evaluate(self, testloader):
        print("Evaluation of global model")
        self.global_model.eval()
        total = 0
        acc = 0

        with torch.no_grad():
            for data,label in tqdm(testloader):
                data = data.to(device)
                label = label.to(device)
                total += data.size(0)

                output = self.global_model(data)
                pred = torch.argmax(output, 1)

                acc += (pred == label).sum().item()

        print("Accuracy: [{}]".format((acc/total)*100))
        return (acc/total)*100
