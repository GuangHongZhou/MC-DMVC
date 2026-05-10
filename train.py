import os
import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from dataloader import load_data
from network import MCMVC
from metric import valid
from torch.utils.data import Dataset

import argparse

from loss import Loss

from res import MetricTracker

Dataname = 'HandWritten'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=200)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed = 40
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

lamba = 0.1
tracker = MetricTracker()

dataset, dims, view, data_size, class_num = load_data(args.dataset)

print(dims)
print(view)
print(data_size)
print(class_num)


data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

mm = MinMaxScaler()

def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Pre-train Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def fine_tune(epoch,lamba):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs = model(xs)
        commonz, S = model.MCMVC(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(lamba * criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Fine-tune Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


model = MCMVC(view, dims, args.low_feature_dim, args.high_feature_dim,  device)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, args.temperature_f, device).to(device)

for pre_epoch in range(1, args.rec_epochs + 1):
    pre_train(pre_epoch)


for ft_epoch in range(1, args.fine_tune_epochs + 1):
    fine_tune(ft_epoch,lamba)
    flag = valid(model, device, dataset, view, data_size, class_num, tracker)
    if flag:
        state = model.state_dict()
        file_path = './models/' + args.dataset
        os.makedirs(file_path, exist_ok=True)
        torch.save(state, file_path + '/' + args.dataset +'_' + str(view) +'_' + str(lamba) + '.pth')
        print(str(ft_epoch) + '_Saving..')

    if ft_epoch == args.fine_tune_epochs:
        Nmi, Ari, Acc, Pur = tracker.get_result()
        print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(Acc, Nmi, Pur, Ari))
