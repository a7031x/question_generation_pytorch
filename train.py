from data import Dataset, TrainFeeder, align2d, align3d
from model import Discriminator, Generator
import numpy as np
import os
import config
import torch
import utils

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def tensor(x):
    return torch.tensor(x).cuda()


def print_prediction(feeder, similarity, pids, qids, gids, labels, number=None):
    if number is None:
        number = len(pids)
    for k in range(min(len(pids), number)):
        pid, qid, gid, sim, lab = pids[k], qids[k], gids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        print(passage)

        questions = [feeder.ids_to_sent(q) for q in qid]
        for q,s,l in zip(questions, sim, lab):
            if q:
                print(' {} {:>.4F}: {}'.format(l, s, q))
        print('generated: {}'.format(feeder.ids_to_sent(gid)))


def generator_similarities(generator, discriminator, pids, qids, labels, ctx=None, state=None, ctx_mask=None):
    x = tensor(pids)
    if ctx is None:
        ctx, state, ctx_mask = discriminator.encode_passage(x)
    question_logit = generator(ctx, state, ctx_mask)

    new_qids = []
    for questions, ls in zip(qids, labels): 
        new_qids.append([[x for x in q if x != config.NULL_ID] for q,l in zip(questions, ls) if l == 1])
    new_qids = align3d(new_qids)

    passage_similarity, question_similarity = discriminator.compute_similarity(tensor(pids), tensor(new_qids), question_logit)
    return passage_similarity, question_similarity, question_logit


def gan_loss(criterion, passage_similarity, question_similarity, labels, target):
    batch_size = passage_similarity.shape[0]
    passage_label = tensor([target]*batch_size).float()
    passage_loss = criterion(passage_similarity, passage_label).sum() / batch_size

    new_labels = [[x for x in r if x == 1] for r in labels]
    new_labels = align2d(new_labels)
    weight = tensor(new_labels if target == 1 else np.zeros_like(new_labels)).float()
    question_loss = (criterion(question_similarity, weight) * tensor(new_labels).float()).sum()
    return passage_loss, question_loss


def run_epoch(generator, discriminator, feeder, generator_optimizer, discriminator_optimizer, batches):
    nbatch = 0 
    while nbatch < batches:
        pids, qids, labels, _ = feeder.next()
        nbatch += 1
        x = tensor(pids)
        y = tensor(qids)
        similarity, count, ctx, state, ctx_mask = discriminator(x, y)
        criterion = torch.nn.BCEWithLogitsLoss(reduce=False)
        discriminator_loss = criterion(similarity, tensor(labels).float()).sum()/count.float()

        passage_similarity, question_similarity, question_logit = generator_similarities(generator, discriminator, pids, qids, labels, ctx, state, ctx_mask)
        passage_loss, question_loss = gan_loss(criterion, passage_similarity, question_similarity, labels, 0)
        generator_loss = passage_loss + question_loss
        factor = min(((discriminator_loss / generator_loss) * 0.1).tolist(), 1)
        generator_loss *= factor

        d_loss = discriminator_loss + generator_loss
        discriminator_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        passage_loss, question_loss = gan_loss(criterion, passage_similarity, question_similarity, labels, 1)
        
        g_loss = passage_loss + question_loss
        generator_optimizer.zero_grad()
        g_loss.backward()
        generator_optimizer.step()

        gids = question_logit.argmax(-1).tolist()
        d_loss, g_loss, similarity = d_loss.tolist(), g_loss.tolist(), torch.sigmoid(similarity).tolist()
        print_prediction(feeder, similarity, pids, qids, gids, labels, 1)
        print('------ITERATION {}, {}/{}, d_loss: {:>.4F}+{:>.4F}={:>.4F}, {:>.4F}+{:>.4F}={:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, discriminator_loss, generator_loss, d_loss, passage_loss, question_loss, g_loss))


def train(auto_stop, steps=50, threshold=0.2):
    dataset = Dataset()
    feeder = TrainFeeder(dataset)
    discriminator = Discriminator(len(dataset.ci2n)).cuda()
    generator = Generator(len(dataset.ci2n)).cuda()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        discriminator.load_state_dict(ckpt['discriminator'])
        generator.load_state_dict(ckpt['generator'])
        discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer'])
        generator_optimizer.load_state_dict(ckpt['generator_optimizer'])
        feeder.load_state(ckpt['feeder'])
    while True:
        #run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.2, 100)
        run_epoch(generator, discriminator, feeder, generator_optimizer, discriminator_optimizer, steps)
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'feeder': feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)