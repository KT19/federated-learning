#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_selects", type=int, default=3)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--local_update_epoch", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device",default="cuda:0")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--save_dir", default="output")

args = parser.parse_args()

device = (args.device if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
