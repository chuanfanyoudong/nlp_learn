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
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        初始化隐层
        :return: 隐层参数
        """
        return (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim))


    def forward(self, sentence, x_lengths):
        embeds = self.word_embeddings(sentence.cuda())
        max_length = max(x_lengths)
        embeds = embeds[:, :max_length ,:]
        # _, idx_sort = torch.sort(x_lengths, dim=0, descending=True)
        # _, idx_unsort = torch.sort(idx_sort, dim=0)
        # input_x = embeds.index_select(0, Variable(idx_sort).cuda())
        # length_list = list(x_lengths[idx_sort])
        # embeds = torch.nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first= 1)
        lstm_out, self.hidden = self.lstm(embeds)
        # lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=1)
        tag_space = self.hidden2tag(lstm_out)
        tag_space = tag_space.permute(0, 2, 1)
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores

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