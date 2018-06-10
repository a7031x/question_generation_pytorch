from data import Dataset, TrainFeeder
from model import Discriminator
import torch
import torchvision
import torch.nn as nn
import config


def run_discriminator_epoch(itr, model, feeder, criterion, optimizer):
    feeder.prepare('train')
    nbatch = 0 
    while not feeder.eof():
        x, y, labels, _ = feeder.next()
        nbatch += 1
        similarity = model(torch.LongTensor(x), torch.LongTensor(y))
        optimizer.zero_grad()
        loss = criterion(similarity, torch.Tensor(labels))
        loss.backword()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, loss.tolist()))
        if nbatch % 10 == 0:
            print('----dev----')


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    model = Discriminator(len(dataset.ci2n), config.checkpoint_folder)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    itr = 0
    while True:
        itr += 1
        run_discriminator_epoch(itr, model, feeder, criterion, optimizer)


train(False)