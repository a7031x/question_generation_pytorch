import torch
import torch.nn as nn
import numpy as np
import config
import decoder

class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.passage_conv0 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size0, config.conv_vector_dim)
        self.passage_conv1 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim)
        #self.passage_gate = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim*2)
        self.question_conv0 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size0, config.conv_vector_dim)
        self.question_conv1 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim)
        self.question_gate = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim*2)
        self.question_encoder = nn.LSTM(config.conv_vector_dim*2, config.encoder_hidden_dim, 2, bidirectional=True, batch_first=True)
        self.encoder_dim = config.encoder_hidden_dim * 2
        self.passage_dense = nn.Linear(self.encoder_dim, config.dense_match_dim)
        self.passage_encoder = nn.LSTM(config.conv_vector_dim*2, config.encoder_hidden_dim, 2, bidirectional=True, batch_first=True)


    def forward(self, x, y):
        ctx, state_x, ctx_mask = self.encode_passage(x)
        batch_size = y.shape[0]
        num_questions = y.shape[1]
        y = y.view(batch_size*num_questions, -1)
        mask = y != 0
        length = torch.sum(mask, -1)
        mask = length != 0
        state_y = self.encode_question(y)
        tiled_state_x = state_x.repeat(1, num_questions).view(batch_size*num_questions, -1)
        similarity = torch.sum(tiled_state_x * state_y, -1) * mask.float()
        similarity = similarity.view(batch_size, num_questions)
        return similarity, torch.sum(mask), ctx, state_x, ctx_mask


    def compute_similarity(self, passage, logit):
        question_embed = torch.einsum('bij,jk->bik', (logit, self.embedding.weight))
        _, state_x, _= self.encode_passage(passage)
        state_y = self.encode_question_embedding(question_embed)
        similarity = torch.sum(state_x * state_y, -1)
        return similarity


    def encode_question(self, input):
        return self.encode_question_embedding(self.embedding(input))


    def encode_passage(self, input):
        embed = self.embedding(input)
        encoding0 = self.encode_embedding(self.passage_conv0, embed)
        encoding1 = self.encode_embedding(self.passage_conv1, embed)
        encoding = torch.cat([encoding0, encoding1], -1)
        dense = self.passage_dense(encoding)
        weight_dim = dense.shape[-1]
        coref = torch.bmm(dense, dense.transpose(1, 2)) / (weight_dim**0.5)
        mask = (input != 0).float()
        coref -= (1-mask.unsqueeze(1)) * 100000
        alpha = nn.functional.softmax(coref, -1)
        context = torch.bmm(alpha, encoding)
        ctx, (state_h, state_c) = self.passage_encoder(context)
        state = torch.cat([state_h.transpose(0, 1), state_c.transpose(0, 1)], -1).view(embed.shape[0], -1)
        return ctx, state, mask


    def encode_question_embedding(self, embed):
        encoding0 = self.encode_embedding(self.question_conv0, embed)
        encoding1 = self.encode_embedding(self.question_conv1, embed)
        gate = torch.sigmoid(self.encode_embedding(self.question_gate, embed))
        encoding = torch.cat([encoding0, encoding1], -1) * gate
        _, (state_h, state_c) = self.question_encoder(encoding)
        state = torch.cat([state_h.transpose(0, 1), state_c.transpose(0, 1)], -1).view(embed.shape[0], -1)
        return state
        

    def selfmatch(self, x, dense0, dense1, mask=None):
        coref = torch.bmm(dense0(x), dense1(x).transpose(1, 2))
        if mask is not None:
            coref = coref - (1-mask)*10000
        alpha = torch.sigmoid(coref)
        return torch.bmm(alpha, x)


    def encode_embedding(self, convs, embed):
        x = torch.transpose(embed, 1, 2)
        x = convs(x)
        encoding = torch.transpose(x, 1, 2)
        return encoding


    def cnn_layers(self, num_layers, kernel_size, out_channels):
        modules = nn.Sequential()
        for i in range(num_layers):
            conv = nn.Conv1d(config.embedding_dim if i == 0 else out_channels, out_channels, kernel_size, padding=kernel_size//2)
            modules.add_module('conv_{}'.format(i), conv)
            modules.add_module('tanh_{}'.format(i), nn.Tanh())
        return modules


    def attention_layer(self):
        dense0 = nn.Linear(config.encoder_hidden_dim*2, config.attention_weight_dim, False)
        dense1 = nn.Linear(config.encoder_hidden_dim*2, config.attention_weight_dim, False)
        return dense0, dense1


class Generator(decoder.Ctx2SeqAttention):
    def __init__(self, vocab_size):
        super(Generator, self).__init__(
            ctx_dim=config.encoder_hidden_dim*2,
            num_steps=config.max_question_len,
            vocab_size=vocab_size,
            src_hidden_dim=config.dense_vector_dim,
            trg_hidden_dim=config.dense_vector_dim,
            attention_mode='dot',
            batch_size=config.batch_size,
            bidirectional=True,
            pad_token_src=config.NULL_ID,
            pad_token_trg=config.NULL_ID,
            nlayers=config.num_passage_encoder_layers,
            nlayers_trg=config.num_decoder_rnn_layers,
            dropout=1-config.keep_prob
        )


    def forward(self, ctx, state, ctx_mask):
        decoder_logit = nn.functional.softmax(super(Generator, self).forward(ctx, state, ctx_mask), -1).clone()
        return decoder_logit


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

