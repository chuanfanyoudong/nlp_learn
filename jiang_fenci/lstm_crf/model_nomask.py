#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: model.py
@time: 2019/1/9 15:57
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from conf.config import get_config
config = get_config()


class LSTMSplit(nn.Module):
    """
    LSTM类
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vectors):
        super(LSTMSplit, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.tensor(vectors))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        """
        初始化隐层
        :return: 隐层参数
        """
        if config["segment"]["cuda"] == "1":
            return (torch.zeros(1,1,self.hidden_dim).cuda(), torch.zeros(1,1,self.hidden_dim).cuda())
        else:
            return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

    def forward__(self, sentence, length_list):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_space = tag_space.permute(0, 2, 1)
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


    def forward_(self, sentence, x_lengths):
        embeds = self.word_embeddings(sentence.cuda())
        _, idx_sort = torch.sort(x_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        input_x = embeds.index_select(0, Variable(idx_sort).cuda())
        length_list = list(x_lengths[idx_sort])
        embeds = torch.nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first= 1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first= 1)
        tag_space = self.hidden2tag(lstm_out[0])
        tag_space = tag_space.permute(0, 2, 1)
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.init_hidden())
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


