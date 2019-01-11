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

class LSTMSplit(nn.Module):
    """
    LSTM类
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMSplit, self).__init__()  # 继承父类
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        初始化隐层
        :return: 隐层参数
        """
        return (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores


