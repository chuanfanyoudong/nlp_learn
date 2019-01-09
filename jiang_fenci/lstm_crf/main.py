#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: main.py
@time: 2019/1/9 16:19
"""
import sys

from sklearn_crfsuite import metrics

ROOT = "/data/users/zkjiang/projects/nlp_learn"
sys.path.append(ROOT)
print(sys.path)
from pre_process import *
from model import LSTMSplit
from conf.config import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
__config = get_config()
import numpy as np


def main():
    train_data, val_data, test_data, char_list = get_train_data()
    word2id, embedding_list, tag2id = get_embedding(char_list)
    print(len(word2id))
    lstm_split = LSTMSplit(int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]), len(word2id), len(tag2id))
    loss_function = nn.NLLLoss()  # 指定损失函数
    optimizer = optim.SGD(lstm_split.parameters(), lr = float(__config["segment"]["lr"]))  # 指定优化器
    if __config["segment"]["cuda"] == "1":
        print("使用cuda")
        torch.cuda.set_device(1)
        torch.cuda.manual_seed(int(__config["segment"]["seed"]))  # set random seed for gpu
        lstm_split.cuda()
    with torch.no_grad():
        inputs = prepare_sequence(train_data[0][0], word2id)
        tags_in = prepare_sequence(train_data[0][1], tag2id)
        # print(inputs)

        # inputs = inputs.cuda()
        tag_scores = lstm_split(inputs)
        print(train_data[0][0])
        print(tags_in)
        print(np.argmax(tag_scores, axis=1))

    for epoch in range(int(__config["segment"]["epoch"])):
        print("正在进行第{}次迭代".format(epoch))
        for sentence, tags in train_data:
            lstm_split.zero_grad()  # 清楚梯度
            lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            sentence_in = prepare_sequence(sentence, word2id)
            # sentence_in = sentence_in.cuda()
            # print(type(sentence_in))
            tags_in = prepare_sequence(tags, tag2id)
            tag_score = lstm_split(sentence_in)
            loss = loss_function(tag_score, tags_in)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            inputs = prepare_sequence(train_data[0][0], word2id)
            tags_in = prepare_sequence(train_data[0][1], tag2id)
            # print(inputs)

            # inputs = inputs.cuda()
            tag_scores = lstm_split(inputs)
            print(train_data[0][0])
            print(tags_in)
            print(np.argmax(tag_scores, axis=1))
        val(lstm_split, val_data, word2id, tag2id)

def val_result(model, val_data, word2id, tag2id):
    model.eval()
    with torch.no_grad():
        for sentence, tags in val_data:
            # lstm_split.zero_grad()  # 清楚梯度
            # lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            sentence_in = prepare_sequence(sentence, word2id)
            # sentence_in = sentence_in.cuda()
            # print(type(sentence_in))
            tags_in = prepare_sequence(tags, tag2id)
            tag_score = model(sentence_in)
            # loss = loss_function(tag_score, tags_in)
            # loss.backward()

def val(model, val_data, word2id, tag2id):
    """
    验证模型在验证集上的分数
    :param model: 模型
    :param val_data: 验证集
    :param word2id: 词的对应表
    :param tag2id: 标签对应表
    :return: 验证集F1
    """
    model.eval()
    all_tags = []
    all_pre_tags = []
    with torch.no_grad():
        for sentence, tags in val_data:
            # lstm_split.zero_grad()  # 清楚梯度
            # lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            sentence_in = prepare_sequence(sentence, word2id)
            # sentence_in = sentence_in.cuda()
            # print(type(sentence_in))
            tags_in = prepare_sequence(tags, tag2id)
            all_tags.append(tags_in)
            tag_score = model(sentence_in)
            all_pre_tags.append(list(np.argmax(tag_score, axis=1)))
    print(metrics.flat_classification_report(all_tags, all_pre_tags, labels=[0, 1, 2, 3], digits=3))


def tmp():
    a = np.array([[1,2,3,4,5], [1,2,3,4], [1,2,3]])
    b = np.array([[1, 2, 3, 4, 6], [1, 2, 3, 4], [1, 2, 3]])
    print(metrics.flat_f1_score(a, b, average='weighted', labels= [1,2,3,4,5,6]))
    # f1score = np.mean(metrics.f1_score(a, b, average=None))
    # print(f1score)
    print(metrics.flat_classification_report(a, b, labels=[1,2,3,4,5,6], digits=3))

if __name__ == '__main__':
    main()
    # tmp()