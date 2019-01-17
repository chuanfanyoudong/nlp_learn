#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: model.py
@time: 2019/1/9 15:57
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class LSTMSplit(nn.Module):
    """
    LSTM类
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vectors):

        super(LSTMSplit, self).__init__()  # 继承父类
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # print(vectors)
        self.word_embeddings.weight.data.copy_(torch.tensor(vectors))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional= 1)
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)
        self.hidden = None

    def init_hidden(self):
        """
        初始化隐层
        :return: 隐层参数
        """
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(2,64,self.hidden_dim)
        # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(2,64,self.hidden_dim)

        if 1:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)


    def forward(self, sentence, x_lengths):
        scores_list =  []
        embeds = self.word_embeddings(sentence.cuda())
        max_length = max(x_lengths)
        self.hidden = None
        # embeds = embeds[:, :max_length ,:]
        # _, idx_sort = torch.sort(x_lengths, dim=0, descending=True)
        # _, idx_unsort = torch.sort(idx_sort, dim=0)
        # # input_x = embeds.index_select(0, Variable(idx_sort).cuda())
        # input_x = embeds[idx_sort]
        # length_list = list(x_lengths[idx_sort])
        # embeds = torch.nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first= 1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first= 1)
        # lstm_out = lstm_out[idx_unsort]
        tag_space = self.hidden2tag(lstm_out)
        tag_space_list = []
        # for i in range(len(x_lengths)):
        #     single_tag_space = self.hidden2tag(lstm_out[i][:x_lengths[i]])
        #     tag_space_list.append(single_tag_space)
        #     tag_scores = F.log_softmax(single_tag_space, dim=1)
        tag_space = tag_space.permute(0, 2, 1)
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores

    def forward_(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        X = self.word_embeddings(X.cuda())
        batch_size, seq_len, _ = X.size()
        # print(batch_size, seq_len)
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        _, idx_sort = torch.sort(X_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        X = X.index_select(0, Variable(idx_sort).cuda())
        X_lengths = list(X_lengths[idx_sort])

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        # X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden2tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = X.permute(0, 2, 1)
        X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, 5)
        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.view(-1)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.nb_tags)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).data[0])

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss

    def model_sort(self, array, array_list):
        array = array.detach().numpy()
        final_array = []
        index_length = np.argsort(array_list)
        # print(index_length)
        max_ = max(index_length)
        index_length = [max_ - i for i in index_length]
        # print(index_length)
        n = 0
        for i in index_length:
            index = index_length.index(n)
            n += 1
            final_array.append(list(array[index]))
        return torch.from_numpy(np.array(final_array)), sorted(array_list, reverse=1)


"""
[ 167,  290,  612, 1141, 2639,  160, 2385, 2722,  227,  684,  615,  148,
        1295, 2670,  290,   80,   13,  612, 1141,   59, 1576,  230, 1110,  618,
         113,    2,  679,  334,  461,   96, 1327,  114, 2698,  184,    1,  239,
          34,   70,  126,  518,  736,  148,    2,  968, 1489,    1,    6,  114,
        2698,  184,  114, 2097,    7,   97,  101,   64,  787,   50,  277,  999,
         618,  184,  191,  123,    3,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    

1 ,93, 167
"""


def forward(self, X, X_lengths):
    # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
    # a new batch as a continuation of a sequence
    self.hidden = self.init_hidden()

    batch_size, seq_len, _ = X.size()

    # ---------------------
    # 1. embed the input
    # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
    X = self.word_embedding(X)

    # ---------------------
    # 2. Run through RNN
    # TRICK 2 ********************************
    # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

    # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
    X = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

    # now run through LSTM
    X, self.hidden = self.lstm(X, self.hidden)

    # undo the packing operation
    X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

    # ---------------------
    # 3. Project to tag space
    # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

    # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
    X = X.contiguous()
    X = X.view(-1, X.shape[2])

    # run through actual linear layer
    X = self.hidden_to_tag(X)

    # ---------------------
    # 4. Create softmax activations bc we're doing classification
    # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
    X = F.log_softmax(X, dim=1)

    # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
    X = X.view(batch_size, seq_len, self.nb_tags)

    Y_hat = X
    return Y_hat


def loss(self, Y_hat, Y, X_lengths):
    # TRICK 3 ********************************
    # before we calculate the negative log likelihood, we need to mask out the activations
    # this means we don't want to take into account padded items in the output vector
    # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    # and calculate the loss on that.

    # flatten all the labels
    Y = Y.view(-1)

    # flatten all predictions
    Y_hat = Y_hat.view(-1, self.nb_tags)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = self.tags['<PAD>']
    mask = (Y > tag_pad_token).float()

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).data[0])

    # pick the values for the label and zero out the rest with the mask
    Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(Y_hat) / nb_tokens

    return ce_loss