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
        self.conv = nn.Conv1d(config.embedding_dim, config.dense_vector_dim, config.encoder_kernel_size0, padding=1)


    def forward(self, x):
        x = self.embedding(x)   #[batch, sequence, embedding_dim]
        x = torch.transpose(x, 1, 2)
        cx = self.conv(x)
        cx = torch.transpose(cx, 1, 2)
        print(x.size(), cx.size())


if __name__ == '__main__':
    model = Discriminator(config.char_vocab_size, None)
    seq = torch.LongTensor([[1, 2, 3, 4, 5], [4,5,6,7,8]])
    model.forward(seq)

