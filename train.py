from data import Dataset, TrainFeeder
from model import Discriminator
import os
import config
import torch

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def run_discriminator_epoch(itr, model, feeder, criterion, optimizer):
    feeder.prepare('train')
    nbatch = 0 
    while not feeder.eof():
        x, y, labels, _ = feeder.next()
        nbatch += 1
        similarity, count = model(x, y)
        optimizer.zero_grad()
        loss = criterion(similarity, torch.tensor(labels).float())
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(itr, feeder.cursor, feeder.size, (loss/count.float()).tolist()))
        if nbatch % 10 == 0:
            torch.save(optimizer.state_dict(), ckpt_path)
            print('MODEL SAVED.')


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    model = Discriminator(len(dataset.ci2n))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if os.path.isfile(ckpt_path):
        optimizer.load_state_dict(torch.load(ckpt_path))

    itr = 0
    while True:
        itr += 1
        run_discriminator_epoch(itr, model, feeder, criterion, optimizer)


train(False)