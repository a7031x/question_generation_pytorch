import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import config

class Discriminator(nn.Module):
    def __init__(self, vocab_size, chpt_folder):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.passage_conv0 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size0, config.dense_vector_dim//2)
        self.passage_conv1 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.dense_vector_dim//2)
        self.question_conv0 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size0, config.dense_vector_dim//2)
        self.question_conv1 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size1, config.dense_vector_dim//2)


    def forward(self, x, y):
        state_x = self.encode(self.passage_conv0, self.passage_conv1, x)

        batch_size = y.shape[0]
        num_questions = y.shape[1]
        y = y.reshape([batch_size*num_questions, -1])
        mask = torch.sum(y != 0, -1)
        state_y = self.encode(self.question_conv0, self.question_conv1, y)
        state_x = staet_x.repeat(1, num_questions).reshape(y.shape)
        similarity = torch.sum(state_x * state_y, -1) * mask
        return similarity


    def encode(self, convs0, convs1, text):
        embed = self.embedding(text)
        state0 = self.encode_embedding(convs0, embed)
        state1 = self.encode_embedding(convs1, embed)
        state = torch.cat([state0, state1], -1)
        return state


    def encode_embedding(self, convs, embed):
        x = torch.transpose(embed, 1, 2)
        x = convs(x)
        encoding = torch.transpose(x, 1, 2)
        output = torch.sum(encoding, 1)
        return output


    def cnn_layers(self, num_layers, kernel_size, out_channels):
        modules = nn.Sequential()
        for i in range(num_layers):
            conv = nn.Conv1d(config.embedding_dim if i == 0 else out_channels, out_channels, kernel_size, padding=kernel_size//2)
            modules.add_module('conv_{}'.format(i), conv)
            modules.add_module('tanh_{}'.format(i), nn.Tanh())
        return modules


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    model = Discriminator(config.char_vocab_size, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    seq = torch.LongTensor([[1,2,3,4,5], [4,5,6,7,8]])
    target = torch.Tensor([0, 1])
    for _ in range(10):
        similarity = model(seq, seq)
        loss = criterion(similarity, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.tolist())

