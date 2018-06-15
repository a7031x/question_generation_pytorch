from data import Dataset, TrainFeeder, align2d, align3d
from model import Discriminator, Generator
import numpy as np
import os
import config
import torch
import utils

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def print_prediction(feeder, similarity, pids, qids, labels, number=None):
    if number is None:
        number = len(pids)
    for k in range(min(len(pids), number)):
        pid, qid, sim, lab = pids[k], qids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        print(passage)
        if isinstance(lab, list):
            questions = [feeder.ids_to_sent(q) for q in qid]
            for q,s,l in zip(questions, sim, lab):
                if q:
                    print(' {} {:>.4F}: {}'.format(l, s, q))
        else:
            question = feeder.ids_to_sent(qid)
            print(' {} {:>.4F}: {}'.format(lab, sim, question))


def run_discriminator_epoch(generator, discriminator, feeder, criterion, optimizer, batches):
    nbatch = 0 
    while nbatch < batches:
        pids, qids, labels, _ = feeder.next()
        batch_size = len(pids)
        nbatch += 1
        x = torch.tensor(pids).cuda()
        y = torch.tensor(qids).cuda()
        similarity, count, ctx, state, ctx_mask = discriminator(x, y)
        discriminator_loss = criterion(similarity, torch.tensor(labels).cuda().float())/count.float()
        if generator is not None:
            question_logit = generator(ctx, state, ctx_mask)
            generated_similarity = discriminator.compute_similarity(x, question_logit)        
            generation_label = torch.tensor([0]*batch_size).cuda().float()
            generator_loss = criterion(generated_similarity, generation_label)/torch.tensor(batch_size).cuda().float()
            factor = min(((discriminator_loss / generator_loss) * 0.1).tolist(), 1)
            generator_loss *= factor
        else:
            generator_loss = 0
        loss = discriminator_loss + generator_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, similarity = loss.tolist(), torch.sigmoid(similarity).tolist()
        print_prediction(feeder, similarity, pids, qids, labels, 1)
        print('------ITERATION {}, {}/{}, loss: {:>.4F}+{:>.4F}={:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, discriminator_loss, generator_loss, loss))
    return loss


def run_generator_epoch(generator, discriminator, feeder, criterion, optimizer, threshold, batches):
    loss = 100
    ibatch = 0
    while loss >= threshold and ibatch <= batches:
        ibatch += 1
        pids, qids, labels, _ = feeder.next(align=True)
        batch_size = len(pids)

        optimizer.zero_grad()

        x = torch.tensor(pids).cuda()
        ctx, state, ctx_mask = discriminator.encode_passage(x)
        question_logit = generator(ctx, state, ctx_mask)
        gids = question_logit.argmax(-1).tolist()
        passage_similarity = discriminator.compute_similarity(x, question_logit)
        label = torch.tensor([1]*batch_size).cuda().float()
        passage_loss = criterion(passage_similarity, label) / batch_size
        passage_loss.backward(retain_graph=True)

        new_qids = []
        new_labels = []
        for questions, ls in zip(qids, labels): 
            new_qids.append([[x for x in q if x != config.NULL_ID] for q,l in zip(questions, ls) if l == 1])
            new_labels.append([1 for l in ls if l == 1])
        new_qids = align3d(new_qids)
        new_labels = align2d(new_labels)
        y = torch.tensor(new_qids).cuda()
        question_similarity = discriminator.compute_questions_similarity(y, question_logit)
        weight = torch.tensor(new_labels).cuda()
        question_loss = torch.nn.BCEWithLogitsLoss(weight=weight.float(), size_average=False)(question_similarity, weight.float())
        question_loss.backward()

        loss = passage_loss + question_loss
        optimizer.step()
        loss, passage_similarity = loss.tolist(), torch.sigmoid(passage_similarity).tolist()
        print_prediction(feeder, passage_similarity, [q[0] for q in qids], gids, label)
        print('------ITERATION {}, {}/{}, loss: {:>.4F}+{:>.4F}={:>.4F}'.format(feeder.iteration, feeder.cursor, feeder.size, passage_loss, question_loss, loss))


def train(auto_stop, steps=50, threshold=0.2):
    dataset = Dataset()
    discriminator_feeder = TrainFeeder(dataset)
    generator_feeder = TrainFeeder(dataset)
    discriminator = Discriminator(len(dataset.ci2n)).cuda()
    generator = Generator(len(dataset.ci2n)).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    discriminator_feeder.prepare('train')
    generator_feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        discriminator.load_state_dict(ckpt['discriminator'])
        generator.load_state_dict(ckpt['generator'])
        discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer'])
        generator_optimizer.load_state_dict(ckpt['generator_optimizer'])
        discriminator_feeder.load_state(ckpt['discriminator_feeder'])
        generator_feeder.load_state(ckpt['generator_feeder'])
    loss = 1
    while True:
        #run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.2, 100)
        loss = run_discriminator_epoch(generator if loss < threshold else None, discriminator, discriminator_feeder, criterion, discriminator_optimizer, steps)
        if loss < threshold:
            run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.1, 1000)
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_feeder': discriminator_feeder.state(),
            'generator_feeder': generator_feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)