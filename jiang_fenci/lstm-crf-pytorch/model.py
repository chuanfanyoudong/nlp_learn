import torch
import torch.nn as nn

UNIT = "char" # unit for tokenization (char, word)
PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags, HIDDEN_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_DIRS, NUM_LAYERS, vectors,padding_idx, BIDIRECTIONAL = 1, DROPOUT = 0.5):
        super().__init__()
        self.lstm = lstm(vocab_size, num_tags, EMBED_SIZE, HIDDEN_SIZE, NUM_DIRS, NUM_LAYERS, vectors, padding_idx, BIDIRECTIONAL, DROPOUT)
        self.crf = crf(num_tags, BATCH_SIZE)
        self = self.cuda() if CUDA else self

    def forward(self, x, y): # for training
        mask = x.data.gt(0).float()
        h = self.lstm(x, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y, mask)
        return Z - score # NLL loss

    def decode(self, x): # for prediction
        mask = x.data.gt(0).float()
        h = self.lstm(x, mask)
        return self.crf.decode(h, mask)

class lstm(nn.Module):
    def __init__(self, vocab_size, num_tags, EMBED_SIZE, HIDDEN_SIZE, NUM_DIRS, NUM_LAYERS, vectors, padding_idx, BIDIRECTIONAL = 1, DROPOUT = 0.5):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE)
        self.embed.weight.data.copy_(torch.tensor(vectors))
        self.lstm = nn.LSTM(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # LSTM output to tag

    def init_hidden(self): # initialize hidden states
        h = zeros(
            self.NUM_LAYERS * self.NUM_DIRS, self.BATCH_SIZE, self.HIDDEN_SIZE // self.NUM_DIRS) # hidden states
        c = zeros(self.NUM_LAYERS * self.NUM_DIRS, BATCH_SIZE, self.HIDDEN_SIZE // self.NUM_DIRS) # cell states
        return (h, c)

    def forward(self, x, mask):
        self.hidden = None
        max = mask.sum(1).int().max()
        mask = mask[:,:mask.sum(1).int().max()]
        x = self.embed(x)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first = True)
        h, _ = self.lstm(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h

class crf(nn.Module):
    def __init__(self, num_tags, BATCH_SIZE):
        super().__init__()
        self.num_tags = num_tags
        self.BATCH_SIZE = BATCH_SIZE
        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000. # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000. # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        score = Tensor(self.BATCH_SIZE, self.num_tags).fill_(-10000.) # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y, mask): # calculate the score of a given sequence
        score = Tensor(self.BATCH_SIZE).fill_(0.)
        h = h.unsqueeze(3)
        y = y[:, :h.size(1) + 1]
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            list_tmp = []
            # for h, y in zip(h, y):
            #     list_tmp
            #     print(h)
            #     print(y)
            #     break
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = LongTensor()
        score = Tensor(1, self.num_tags).fill_(-10000.)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(1):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()
        return best_path

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
