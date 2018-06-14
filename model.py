import torch
import torch.nn as nn
import numpy as np
import config
import seq2seq.rnn as rnn


class Discriminator(nn.Module):
    def __init__(self, vocab_size):
        super(Discriminator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.passage_conv0 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size0, config.conv_vector_dim)
        self.passage_conv1 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim)
        self.passage_gate = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim*2)
        self.question_conv0 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size0, config.conv_vector_dim)
        self.question_conv1 = self.cnn_layers(config.num_question_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim)
        self.question_gate = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim*2)
        self.encoder = nn.LSTM(config.conv_vector_dim*2, config.encoder_hidden_dim, 1, bidirectional=True, batch_first=True)
        self.passage_attention0, self.passage_attention1 = self.attention_layer()
        self.question_attention0, self.question_attention1 = self.attention_layer()


    def forward(self, x, y):
        state_x = self.encode_embedding2(self.passage_conv0, self.passage_conv1, self.passage_gate, self.embedding(x), self.passage_attention0, self.passage_attention1)
        batch_size = y.shape[0]
        num_questions = y.shape[1]
        y = y.view(batch_size*num_questions, -1)
        mask = y != 0
        length = torch.sum(mask, -1)
        mask = length != 0
        state_y = self.encode_embedding2(self.question_conv0, self.question_conv1, self.question_gate, self.embedding(y), self.question_attention0, self.question_attention1)
        state_x = state_x.repeat(1, num_questions).view(batch_size*num_questions, -1)
        similarity = torch.sum(state_x * state_y, -1) * mask.float()
        similarity = similarity.view(batch_size, num_questions)
        return similarity, torch.sum(mask)


    def compute_similarity(self, passage, logit):
        question_embed = torch.einsum('bij,jk->bik', (logit, self.embedding.weight))
        state_x = self.encode_embedding2(self.passage_conv0, self.passage_conv1, self.passage_gate, self.embedding(passage))
        state_y = self.encode_embedding2(self.question_conv0, self.question_conv1, self.question_gate, question_embed)
        similarity = torch.sum(state_x * state_y, -1)
        return similarity


    def encode_embedding2(self, convs0, convs1, convsg, embed, attention0, attention1):
        encoding0 = self.encode_embedding(convs0, embed)
        encoding1 = self.encode_embedding(convs1, embed)
        encoding = torch.cat([encoding0, encoding1], -1)
        encoding = self.selfmatch(encoding, attention0, attention1)
        gate = torch.sigmoid(self.encode_embedding(convsg, embed))
        encoding = encoding * gate
        _, (state_h, state_c) = self.encoder(encoding)
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


class Generator(rnn.Seq2SeqAttentionSharedEmbedding):
    def __init__(self, vocab_size):
        super(Generator, self).__init__(
            embedding=nn.Embedding(vocab_size, config.embedding_dim),
            src_hidden_dim=config.encoder_hidden_dim,
            trg_hidden_dim=config.decoder_hidden_dim,
            ctx_hidden_dim=config.dense_vector_dim,
            max_question_len=config.max_question_len,
            attention_mode='dot',
            batch_size=config.batch_size,
            bidirectional=True,
            pad_token_src=config.NULL_ID,
            pad_token_trg=config.NULL_ID,
            nlayers=config.num_passage_encoder_layers,
            nlayers_trg=config.num_decoder_rnn_layers,
            dropout=1-config.keep_prob
        )


    def forward(self, x):
        decoder_logit = super(Generator, self).forward(x)
        ids = torch.argmax(decoder_logit, -1)
        decoder_logit[:,:,config.EOS_ID] = 0
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

