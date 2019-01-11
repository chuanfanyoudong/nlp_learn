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
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.model_zoo import tqdm

ROOT = "/data/users/zkjiang/projects/nlp_learn/"
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
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='max.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
logger = logging.getLogger(__name__)

def main():
    best_score = 0
    train_data, val_data, test_data, char_list, train_data_length, val_data_length, test_data_length = get_train_data()
    word2id, embedding_list, tag2id = get_embedding(char_list)
    train_dataloader = generate_iteration(train_data, word2id, tag2id, char_list, train_data_length)
    val_dataloader = generate_iteration(val_data, word2id, tag2id, char_list, val_data_length)
    test_dataloader = generate_iteration(test_data, word2id, tag2id, char_list, test_data_length)
    lstm_split = LSTMSplit(int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]), len(word2id), len(tag2id), embedding_list)
    loss_function = nn.NLLLoss(reduce= False) # 指定损失函数
    optimizer = optim.SGD(lstm_split.parameters(), lr = float(__config["segment"]["lr"]))  # 指定优化器
    if __config["segment"]["cuda"] == "1":
        print("使用cuda")
        torch.cuda.set_device(1)
        torch.cuda.manual_seed(int(__config["segment"]["seed"]))  # set random seed for gpu
        lstm_split.cuda()
    # for epoch in range(int(__config["segment"]["epoch"])):
    #     print("正在进行第{}次迭代".format(epoch))
    #     for batch in tqdm(train_dataloader, desc="Iteration"):
    #         sentence_in, tags_in, length_data = batch
    #         lstm_split.zero_grad()  # 清楚梯度
    #         lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
    #         tag_score = lstm_split(sentence_in, length_data)
    #         loss = loss_function(tag_score, tags_in)
    #         loss.backward()
    #         optimizer.step()
    #     val_f1 = val(lstm_split, val_dataloader, word2id, tag2id)
    #     test_f1 = test(lstm_split, test_dataloader, word2id, tag2id)
    for epoch in range(int(__config["segment"]["epoch"])):
        logging.info("正在进行第{}次迭代".format(epoch))
        total_loss = 0.0
        tmp_total_loss = 0.0
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
            tmp_loss = loss.mean()
            mask_mat = Variable(torch.ones(len(list(length_data)), max_length))
            for idx_sample in range(len(list(length_data))):

                if list(length_data)[idx_sample] != max_length:
                    mask_mat[idx_sample, list(length_data)[idx_sample]:] = 0.0
            loss = (loss * mask_mat.cuda()).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss
            tmp_total_loss += tmp_loss
        print("正在进行第{}次迭代, loss为{}, tmp_loss为{}".format(epoch, total_loss, tmp_total_loss))
        total_loss = 0
        tmp_total_loss = 0
        train_f1 = val(lstm_split, train_dataloader, word2id, tag2id)
        val_f1 = val(lstm_split, val_dataloader, word2id, tag2id)
        print("训练数据F1:{}".format(str(train_f1)))
        print("测试数据F1:{}".format(str(val_f1)))
        # test(lstm_split, test_dataloader, word2id, tag2id)
        # print("F1:{}".format(str(val_f1)))
        # logging.info("F1:{}".format(str(val_f1)))

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
        # for sentence, tags in val_data:
        for batch in tqdm(val_data, desc="Iteration"):
            sentence_in, tags_in, length_list = batch
            tags_in = tags_in.numpy()
            # if all_tags == []:
            #     all_tags = tags_in
            # else:
            #     all_tags = np.vstack((all_tags, tags_in))
            tag_score = model(sentence_in, length_list)
            tag_score = np.argmax(tag_score.cpu().numpy(), axis=1)
            for i in range(len(tags_in)):
                for j in range(len(tags_in[i, :])):
                    if j == 4:
                        break
                    else:
                        all_tags.append(tags_in[i,j])
                        all_pre_tags.append(tag_score[i,j])
        all_tags = all_tags
        all_pre_tags = all_pre_tags
        # print("真实情况是：")
        # print(all_tags)
        # print("预测结果是：")
        # print(all_pre_tags)
    # print("进行{}测试：".format(val_data.__doc__[:5]))
    # print(metrics.flat_classification_report([all_tags], [all_pre_tags], labels=[0, 1, 2, 3], digits=3))
    F1 = metrics.flat_f1_score([all_tags], [all_pre_tags], average='weighted', labels=[0, 1, 2, 3])
    return F1



def test(model, test_data, word2id, tag2id):
    """
    测试几条数据的效果
    :param model: 模型
    :param val_data: 验证集
    :param word2id: 词的对应表
    :param tag2id: 标签对应表
    :return: 验证集F1
    """
    model.eval()
    all_tags = []
    all_pre_tags = []
    # sentence = "我来到伟大的中国"
    # sentence_in = prepare_sequence(sentence, word2id)
    # tag_score = model(sentence_in, torch.tensor([8]))
    # tag_score = np.argmax(tag_score.numpy(), axis=1)
    # print(tag_score)
    with torch.no_grad():
        # for sentence, tags in val_data:
        for batch in tqdm(test_data, desc="Iteration"):
            sentence_in, tags_in, length_list = batch
            tags_in = tags_in.numpy()
            # if all_tags == []:
            #     all_tags = tags_in
            # else:
            #     all_tags = np.vstack((all_tags, tags_in))
            tag_score = model(sentence_in, length_list)
            tag_score = np.argmax(tag_score.numpy(), axis=1)
            for i in range(len(tags_in)):
                # print("tags_in:")
                # print(tags_in[i, :tag_score.shape[1]])
                # print("tags_score:")
                # print(tag_score[i, :])

                for j in range(len(tags_in[i, :])):
                    if tags_in[i, j] == 4:
                        print("tags_in:")
                        print(tags_in[i, :j])
                        print("tags_score:")
                        print(tag_score[i, :j])
                        break
    #     all_tags = all_tags
    #     all_pre_tags = all_pre_tags
    # print("进行测试集测试：")
    # print(metrics.flat_classification_report([all_tags], [all_pre_tags], labels=[0, 1, 2, 3], digits=3))
    # F1 = metrics.flat_f1_score([all_tags], [all_pre_tags], average='weighted', labels=[0, 1, 2, 3])
    # return F1

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