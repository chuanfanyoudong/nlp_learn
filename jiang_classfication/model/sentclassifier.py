# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-01-01 21:11:50
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 12:30:56

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .my_own_model import MyOwnMoedl
from jiang_classfication.lstm.model import LSTM


class SentClassifier(nn.Module):
    def __init__(self, data):
        super(SentClassifier, self).__init__()
        print("build sentence classification network...")
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        label_size = data.label_alphabet_size
        self.word_hidden = WordSequence(data)
        self.model = MyOwnMoedl(data)
        self.lstm = LSTM(data)



    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        """
        损失计算函数
        :param word_inputs:单词级别的输入词向量,  5 * 128  BATCH_SIZE * 最大句子长度
        :param feature_inputs: 特征级别的输入词向量
        :param word_seq_lengths: 句子的长度列表
        :param char_inputs: char级别的输入词向量
        :param char_seq_lengths: 句子的char级别的长度
        :param char_seq_recover:
        :param batch_label: label标签
        :param mask: 掩盖矩阵
        :return: 损失
        """
        # 核心函数，计算最终的输出，和label作比较输出维度BATCT_SIZE*
        # outs = self.word_hidden.sentence_representation(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # outs = self.model.sentence_representation(word_inputs, feature_inputs, word_seq_lengths, char_inputs,char_seq_lengths, char_seq_recover)
        outs = self.lstm.sentence_representation(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        # loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        outs = outs.view(batch_size, -1)
        # print("a",outs)
        # score = F.log_softmax(outs, 1)
        # print(score.size(), batch_label.view(batch_size).size())
        # print(score)
        # print(batch_label)
        # exit(0)
        total_loss = F.cross_entropy(outs, batch_label.view(batch_size))
        # total_loss = loss_function(score, batch_label.view(batch_size))
        
        _, tag_seq  = torch.max(outs, 1)
        # if self.average_batch:
        #     total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        # 评估函数会用到的前向传播
        # outs = self.word_hidden.sentence_representation(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # outs = self.model.sentence_representation(word_inputs, feature_inputs, word_seq_lengths, char_inputs,char_seq_lengths, char_seq_recover)
        outs = self.lstm.sentence_representation(word_inputs, feature_inputs, word_seq_lengths, char_inputs,char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        outs = outs.view(batch_size, -1)
        _, tag_seq  = torch.max(outs, 1)
        # if a == 0:
        #     print(tag_seq)
        return tag_seq


