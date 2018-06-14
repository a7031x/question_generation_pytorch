import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """
    def __init__(self, dim, ctx_dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, ctx_dim, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(ctx_dim*2, dim, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, input, context, ctx_mask):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        # Get attention
        input = self.linear_in(input)
        attn = torch.einsum('bsd,bd->bs', (context, input))
        attn -= (1-ctx_mask) * 100000
        attn = self.sm(attn).clone()#[batch, sourceL]

        weighted_context = torch.einsum('bs,bsd->bd',(attn, context))#[batch, dim]

        h_tilde = torch.cat((weighted_context, input), 1)#[batch, dim*2]
        h_tilde = self.tanh(self.linear_out(h_tilde))#[batch, dim]

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size, input_size)

    def forward(self, steps, hidden, ctx, ctx_mask):
        """Propogate input through the network."""
        def recurrence(hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, _ = self.attention_layer(hy, ctx, ctx_mask)

            return h_tilde, cy

        output = []
        for _ in range(steps):
            hidden = recurrence(hidden)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        output = torch.cat(output, 0).view(steps, *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

        
class Ctx2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""
    def __init__(
        self,
        ctx_dim,
        num_steps,
        vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        attention_mode,
        batch_size,
        pad_token_src,
        pad_token_trg,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
    ):
        """Initialize model."""
        super(Ctx2SeqAttention, self).__init__()
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        self.src_hidden_dim = src_hidden_dim//2 if self.bidirectional else src_hidden_dim
        self.decoder = LSTMAttentionDot(ctx_dim, trg_hidden_dim, batch_first=True)

        self.encoder2decoder = nn.Linear(self.src_hidden_dim * self.num_directions, trg_hidden_dim)
        self.decoder2vocab = nn.Linear(trg_hidden_dim, vocab_size)
        self.init_weights()


    def init_weights(self):
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)


    def forward(self, ctx, state, ctx_mask):
        h_t, c_t = state.chunk(2, -1)
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        trg_h, (_, _) = self.decoder(
            self.num_steps,
            (decoder_init_state, c_t),
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit


    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs