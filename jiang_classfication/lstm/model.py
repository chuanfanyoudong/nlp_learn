#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: model.py
@time: 2019/3/6 19:34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 有两个做法有待实验验证
# 1、kmax_pooling的使用，对所有RNN的输出做最大池化
# 2、分类器选用两层全连接层+BN层，还是直接使用一层全连接层
# 3、是否需要init_hidden
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from jiang_classfication.model.wordrep import WordRep


# class LSTM():
#     def __init__(self, config):
#         super(LSTM, self).__init__()
#         self.config = config
#         # self.kmax_pooling = config.kmax_pooling
#         self.bilstm_flag = config.HP_bilstm
#         self.droplstm = nn.Dropout(config.HP_dropout)
#         self.lstm_layer = config.HP_lstm_layer
#         self.embedding = nn.Embedding(config.word_alphabet.size(), config.word_emb_dim).cuda()
#         self.input_size = config.word_emb_dim
#         self.feature_num = config.feature_num
#         self.word_feature_extractor = "LSTM"
#         self.wordrep = WordRep(config)
#         if self.bilstm_flag:
#             lstm_hidden = config.HP_hidden_dim // 2
#         else:
#             lstm_hidden = config.HP_hidden_dim
#         self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
#                                 bidirectional=self.bilstm_flag).cuda()
#         self.hidden2tag = nn.Linear(config.HP_hidden_dim, config.label_alphabet_size).cuda()
#
#     def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs,
#                                                         char_seq_lengths, char_seq_recover):
#         batch_size = word_inputs.size(0)
#         embed = self.embedding(word_inputs)
#         word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
#                                       char_seq_recover)
#         packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
#         hidden = None
#         lstm_out, hidden = self.lstm(packed_words, hidden)
#         lstm_out, _ = pad_packed_sequence(lstm_out)
#         # feature_out = self.droplstm(lstm_out.transpose(1, 0))
#         outputs = self.hidden2tag(lstm_out)
#         return outputs
#
#     def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs,
#                                                         char_seq_lengths, char_seq_recover):
#         batch_size = word_inputs.size(0)
#         embed = self.embedding(word_inputs)
#         word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
#                                       char_seq_recover)
#         packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
#         hidden = None
#         lstm_out, hidden = self.lstm(packed_words, hidden)
#         lstm_out, _ = pad_packed_sequence(lstm_out)
#         # feature_out = self.droplstm(lstm_out.transpose(1, 0))
#         outputs = self.hidden2tag(lstm_out)
#         return outputs



class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        # self.kmax_pooling = config.kmax_pooling
        self.bilstm_flag = config.HP_bilstm
        self.droplstm = nn.Dropout(config.HP_dropout)
        self.lstm_layer = config.HP_lstm_layer
        self.embedding = nn.Embedding(config.word_alphabet.size(), config.word_emb_dim).cuda()
        self.input_size = config.word_emb_dim
        self.feature_num = config.feature_num
        self.word_feature_extractor = "LSTM"
        self.wordrep = WordRep(config)
        if self.bilstm_flag:
            lstm_hidden = config.HP_hidden_dim // 2
        else:
            lstm_hidden = config.HP_hidden_dim
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                                bidirectional=self.bilstm_flag).cuda()
        self.hidden2tag = nn.Linear(config.HP_hidden_dim, config.label_alphabet_size).cuda()

    # def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs,
    #                                                     char_seq_lengths, char_seq_recover):
    #     word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
    #                                   char_seq_recover)
    #     packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
    #     hidden = None
    #     lstm_out, hidden = self.lstm(packed_words, hidden)
    #     lstm_out, _ = pad_packed_sequence(lstm_out)
    #     # feature_out = self.droplstm(lstm_out.transpose(1, 0))
    #     outputs = self.hidden2tag(lstm_out)
    #     return outputs
    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs,
                                                        char_seq_lengths, char_seq_recover):
        batch_size = word_inputs.size(0)
        embed = self.embedding(word_inputs)
        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        feature_out = hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)
        outputs = self.hidden2tag(feature_out)
        return outputs


class LSTM1(nn.Module):
    def __init__(self, data):
        super(LSTM1, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num
        self.word_feature_extractor = "LSTM"
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                                bidirectional=self.bilstm_flag)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        # 得到句子的表示向量，包括word级别的和特征级别的和char级别的拼接而成
        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,char_seq_recover)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # feature_out = self.droplstm(lstm_out.transpose(1, 0))
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def sentence_representation(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover):
        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        batch_size = word_inputs.size(0)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        feature_out = hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)
        outputs = self.hidden2tag(feature_out)
        return outputs
