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
import time

from sklearn_crfsuite import metrics
from torch.autograd import Variable
from torch.utils.model_zoo import tqdm

ROOT = "/data/users/zkjiang/projects/nlp_learn"
sys.path.append(ROOT)
from pre_process_nomask import *
from model_nomask import LSTMSplit
from lstm_crf import BiLSTM_CRF
from conf.config import get_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
__config = get_config()
import numpy as np
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='no_tmp_mask_all_0113.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
logger = logging.getLogger(__name__)

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)


# train_data = [
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
# word2id = {}
# for sent, tags in train_data:
#     for word in sent:
#         if word not in word2id:
#             word2id[word] = len(word2id)
# print(word2id)
# tag2id = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6



def main_():
    train_data, val_data, test_data, char_list, train_data_length, val_data_length, test_data_length = get_train_data(data_all = 0)
    for i in train_data:
        if i == None:
            print(1)
    word2id, embedding_list, tag2id = get_embedding(char_list)
    train_dataloader = generate_iteration(train_data, word2id, tag2id, char_list, train_data_length)
    val_dataloader = generate_iteration(val_data, word2id, tag2id, char_list, val_data_length)
    test_dataloader = generate_iteration(test_data, word2id, tag2id, char_list, test_data_length)
    print(len(word2id))
    lstm_split = LSTMSplit(int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]), len(word2id), len(tag2id), embedding_list)
    loss_function = nn.NLLLoss(reduce= 0)
    optimizer = optim.SGD(lstm_split.parameters(), lr=0.1)
    if __config["segment"]["cuda"] == "1":
        print("使用cuda")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(int(__config["segment"]["seed"]))  # set random seed for gpu
        lstm_split.cuda()
    for epoch in range(300):
        print("\n正在进行第{}次迭代".format(epoch))
        total_loss = 0  # again, normally you would NOT do 300 epochs, it is toy data
        lstm_split.train()
        for batch in tqdm(train_dataloader, desc="Iteration"):
            sentence_in, tags_in, length_data = batch
            max_length = max(length_data)
            tags_in = tags_in[:,:max_length]
            tags_in = tags_in.long()
        # for sentence, tags in train_data:
            lstm_split.zero_grad()  # 清楚梯度
            lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            tag_score = lstm_split(sentence_in, length_data)
            loss = loss_function(tag_score, tags_in.cuda())
            mask_mat = Variable(torch.ones(len(list(length_data)), max_length))
            for idx_sample in range(len(list(length_data))):
                if list(length_data)[idx_sample] != max_length:
                    mask_mat[idx_sample, list(length_data)[idx_sample]:] = 0.0
            loss = (loss * mask_mat.cuda()).mean()
            tmp_loss = loss.mean()
            total_loss += tmp_loss
        print("损失为{}".format(total_loss))
        total_loss = 0

        all_tags, all_pre_tags, pre = val(lstm_split, train_dataloader, word2id, tag2id)
        print("训练集准确率是：")
        print(pre)
        # all_tags, all_pre_tags, pre = val(lstm_split, val_dataloader, word2id, tag2id)
        # print("验证集准确率是：")
        # print(pre)
        print("真实情况是：")
        print(str(all_tags[:50]))
        print("预测结果是：")
        print(str(all_pre_tags[:50]))


def main():
    train_data, val_data, test_data, char_list = get_train_data(data_all= 0)
    for i in train_data:
        if i == None:
            print(1)
    word2id, embedding_list, tag2id = get_embedding(char_list)
    print(len(word2id))
    # lstm_split = BiLSTM_CRF(int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]), len(word2id), len(tag2id), embedding_list)
    lstm_split = BiLSTM_CRF(len(word2id), tag2id, int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]),embedding_list)

    loss_function = nn.NLLLoss(reduce= 0)
    optimizer = optim.SGD(lstm_split.parameters(), lr=0.1)
    if __config["segment"]["cuda"] == "1":
        print("使用cuda")
        torch.cuda.set_device(2)
        torch.cuda.manual_seed(int(__config["segment"]["seed"]))  # set random seed for gpu
        lstm_split.cuda()
    # with torch.no_grad():
    #     inputs = prepare_sequence(train_data[0][0], word2id)
    #     tags = prepare_sequence(train_data[0][1], tag2id)
    #     tag_scores = lstm_split(inputs)
    #     if __config["segment"]["cuda"] == "1":
    #         tag_scores = np.argmax(tag_scores.cpu().numpy(), axis=1)
    #         tags = tags.cpu()
    #     else:
    #         tag_scores = np.argmax(tag_scores.numpy(), axis=1)
    #     print(tags.numpy())
    #     print(tag_scores)

    for epoch in range(300):
        print("\n正在进行第{}次迭代".format(epoch))
        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print("学习速度", param_group["lr"])
        start = time.time()

        total_loss = 0  # again, normally you would NOT do 300 epochs, it is toy data
        lstm_split.train()
        i = 0
        total_loss = 0
        all_loss = 0
        sentence_list = []
        for sentence, tags in train_data:
            i += 1
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            lstm_split.zero_grad()
            
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            # lstm_split.hidden = lstm_split.init_hidden()
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word2id)
            targets = prepare_sequence(tags, tag2id)

            # Step 3. Run our forward pass.

            # a = time.time()  # 初始化隐层
            # tag_scores = lstm_split(sentence_in)
            # b = time.time()
            # print("一批数据的时间", b - a)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # loss = loss_function(tag_scores, targets)
            loss = lstm_split.neg_log_likelihood(sentence_in, targets)
            # loss = loss.mean()
            total_loss += loss
            # print(loss)
            if i%1 == 0:
                # total_loss = total_loss/64
                total_loss.backward()
                all_loss += total_loss
                optimizer.step()
                total_loss = 0
        if total_loss != 0:
            # total_loss.backward()
            # optimizer.step()
            total_loss = 0
        all_loss += total_loss
        end = time.time()
        print("消耗时间为")
        print(start - end)
        print("损失为{}".format(all_loss))
        total_loss = 0

    # See what the scores are after training
    #     with torch.no_grad():
    #
    #         inputs = prepare_sequence(train_data[0][0], word2id)
    #         tag_scores = lstm_split(inputs)
    #         if __config["segment"]["cuda"] == "1":
    #             tag_scores = np.argmax(tag_scores.cpu().numpy(), axis=1)
    #         else:
    #             tag_scores = np.argmax(tag_scores.cpu().numpy(), axis=1)
    #         # print(tag_scores)
        all_tags, all_pre_tags, pre = val(lstm_split, train_data, word2id, tag2id)
        print("测试集准确率是：")
        print(pre)
        # all_tags, all_pre_tags, pre = val(lstm_split, val_data, word2id, tag2id)
        # print("验证集准确率是：")
        # print(pre)
        print("真实情况是：")
        print(str(all_tags[:50]))
        print("预测结果是：")
        print(str(all_pre_tags[:50]))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = float(__config["segment"]["lr"])  * (0.1 ** (epoch // 100))
    # lr = float(__config["segment"]["lr"]) * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def val_(model, val_data, word2id, tag2id):
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
        # for sentence, tags in val_data:
        for batch in tqdm(val_data, desc="Iteration"):
            sentence_in, tags_in, length_list = batch
            sentence_in, tags_in = sentence_in.cuda(), tags_in.cuda()
            tags_in = tags_in.cpu().numpy()
            tag_score = model(sentence_in, length_list)
            tag_score = np.argmax(tag_score.cpu().numpy(), axis=1)
            for i in range(len(tags_in)):
                for j in range(len(tags_in[i, :length_list[i]])):
                    if j == 5:
                        break
                    else:
                        all_tags.append(tags_in[i,j])
                        all_pre_tags.append(tag_score[i,j])
    pre = sum([1 if all_tags[i] == all_pre_tags[i] else 0 for i in range(len(all_tags))])/len(all_tags)
    return all_tags, all_pre_tags, pre

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
    right = 0
    with torch.no_grad():
        for sentence, tags in val_data:
            # lstm_split.zero_grad()  # 清楚梯度
            # lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            sentence_in = prepare_sequence(sentence, word2id)
            # sentence_in = sentence_in.cuda()
            # print(type(sentence_in))
            tags_in = prepare_sequence(tags, tag2id)
            tag_score = model(sentence_in)[1]
            if __config["segment"]["cuda"] == "1":
                # all_pre_tags += (list(np.argmax(tag_score.cpu().numpy(), axis=1)))
                all_pre_tags += tag_score
                tags_in = tags_in.cpu()
            else:
                all_pre_tags += tag_score
            all_tags += list(tags_in.numpy())
    # print(metrics.flat_classification_report(all_tags, all_pre_tags, labels=[0, 1, 2, 3], digits=3))
    pre = sum([1 if all_tags[i] == all_pre_tags[i] else 0 for i in range(len(all_tags))])/len(all_tags)
    return all_tags, all_pre_tags, pre


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