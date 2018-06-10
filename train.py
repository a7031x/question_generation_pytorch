from data import Dataset, TrainFeeder
from model import Discriminator
import numpy as np
import os
import config
import torch

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def print_prediction(feeder, similarity, pids, qids, labels, number):
    for k in range(min(len(pids), number)):
        pid, qid, sim, lab = pids[k], qids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        questions = [feeder.ids_to_sent(q) for q in qid]
        print(passage)
        for q,s,l in zip(questions, 1/(1+np.exp(-np.array(sim))), lab):
            if q:
                print(' {} {:>.4F}: {}'.format(l, s, q))


def run_discriminator_epoch(model, feeder, criterion, optimizer, batches):
    nbatch = 0 
    while nbatch < batches:
        pids, qids, labels, _ = feeder.next()
        nbatch += 1
        x = torch.tensor(pids).cuda()
        y = torch.tensor(qids).cuda()
        similarity, count = model(x, y)
        optimizer.zero_grad()
        loss = criterion(similarity, torch.tensor(labels).cuda().float())
        loss.backward()
        optimizer.step()

        loss, similarity, count = loss.tolist(), similarity.tolist(), count.tolist()
        print_prediction(feeder, similarity, pids, qids, labels, 1)
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(feeder.iteration, feeder.cursor, feeder.size, loss/count))
        if nbatch % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'feeder': feeder.state()
                }, ckpt_path)
            print('MODEL SAVED.')


def train(auto_stop):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    model = Discriminator(len(dataset.ci2n)).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        feeder.load_state(ckpt['feeder'])
    while True:
        run_discriminator_epoch(model, feeder, criterion, optimizer, 100)


train(False)