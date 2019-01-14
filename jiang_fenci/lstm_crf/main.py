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
                    filename='0113.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
logger = logging.getLogger(__name__)

def main():
    best_score = 0
    train_data, val_data, test_data, char_list, train_data_length, val_data_length, test_data_length = get_train_data(data_all = 1)
    word2id, embedding_list, tag2id = get_embedding(char_list)
    train_dataloader = generate_iteration(train_data, word2id, tag2id, char_list, train_data_length)
    val_dataloader = generate_iteration(val_data, word2id, tag2id, char_list, val_data_length)
    test_dataloader = generate_iteration(test_data, word2id, tag2id, char_list, test_data_length)
    lstm_split = LSTMSplit(int(__config["segment"]["embedding_dim"]), int(__config["segment"]["hidden_dim"]), len(word2id), len(tag2id), embedding_list)
    loss_function = nn.NLLLoss(reduce= False) # 指定损失函数
    optimizer = optim.Adam(lstm_split.parameters(), lr = float(__config["segment"]["lr"]))  # 指定优化器
    if __config["segment"]["cuda"] == "1":
        logger.info("使用cuda")
        torch.cuda.set_device(1)
        torch.cuda.manual_seed(int(__config["segment"]["seed"]))  # set random seed for gpu
        lstm_split.cuda()
    for epoch in range(int(__config["segment"]["epoch"])):
        # logger.info("正在进行第{}次迭代".format(epoch))
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
            mask_mat = Variable(torch.ones(len(list(length_data)), max_length))
            for idx_sample in range(len(list(length_data))):
                if list(length_data)[idx_sample] != max_length:
                    mask_mat[idx_sample, list(length_data)[idx_sample]:] = 0.0
            loss = (loss * mask_mat.cuda()).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss
        logger.info("正在进行第{}次迭代, loss为{}, tmp_loss为{}".format(epoch, total_loss, 1))
        # logger.info("正在进行第{}次迭代, loss为{}, tmp_loss为{}".format(epoch, total_loss, tmp_total_loss))
        total_loss = 0
        tmp_total_loss = 0
        train_f1 = val(lstm_split, train_dataloader, word2id, tag2id)
        val_f1 = val(lstm_split, val_dataloader, word2id, tag2id)
        logger.info("训练数据F1:{}".format(str(train_f1)))
        logger.info("测试数据F1:{}".format(str(val_f1)))
        if val_f1 > best_score:
            best_score = val_f1
            checkpoint = {
                'state_dict': lstm_split.state_dict()
            }
            torch.save(checkpoint, __config["segment"]["lstm_model_path"] + str(val_f1))
            print('Best tmp model f1score: {}'.format(best_score))
        # if val_f1 < best_score:
        #     model.load_state_dict(torch.load(save_path)['state_dict'])
        #     lr1 *= args.lr_decay
        #     lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
        #     optimizer = model.get_optimizer(lr1, lr2, 0)
        #     print('* load previous best model: {}'.format(best_score))
        #     print('* model lr:{}  emb lr:{}'.format(lr1, lr2))
        #     if lr1 < args.min_lr:
        #         print('* training over, best f1 score: {}'.format(best_score))
        #         break


def val_result(model, val_data, word2id, tag2id):
    model.eval()
    with torch.no_grad():
        for sentence, tags in val_data:
            # lstm_split.zero_grad()  # 清楚梯度
            # lstm_split.hidden = lstm_split.init_hidden()  # 初始化隐层
            sentence_in = prepare_sequence(sentence, word2id)
            # sentence_in = sentence_in.cuda()
            # logger.info(type(sentence_in))
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
            sentence_in, tags_in = sentence_in.cuda(), tags_in.cuda()
            max_length = max(length_list)
            tags_in = tags_in.cpu().numpy()
            # if all_tags == []:
            #     all_tags = tags_in
            # else:
            #     all_tags = np.vstack((all_tags, tags_in))
            tag_score = model(sentence_in, length_list)
            tag_score = np.argmax(tag_score.cpu().numpy(), axis=1)
            for i in range(len(tags_in)):
                for j in range(len(tags_in[i, :max_length])):
                    if tags_in[i, j] == 4:
                        break
                    else:
                        all_tags.append(tags_in[i,j])
                        all_pre_tags.append(tag_score[i,j])
        logger.info("真实情况是：")
        logger.info(str(all_tags[:100]))
        logger.info("预测结果是：")
        logger.info(str(all_pre_tags[:100]))
    # logger.info("进行{}测试：".format(val_data.__doc__[:5]))
    # logger.info(metrics.flat_classification_report([all_tags], [all_pre_tags], labels=[0, 1, 2, 3], digits=3))
    pre = sum([1 if all_tags[i] == all_pre_tags[i] else 0 for i in range(len(all_tags))])/len(all_tags)
    return pre



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
    # logger.info(tag_score)
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
                # logger.info("tags_in:")
                # logger.info(tags_in[i, :tag_score.shape[1]])
                # logger.info("tags_score:")
                # logger.info(tag_score[i, :])

                for j in range(len(tags_in[i, :])):
                    if tags_in[i, j] == 4:
                        logger.info("tags_in:")
                        logger.info(tags_in[i, :j])
                        logger.info("tags_score:")
                        logger.info(tag_score[i, :j])
                        break
    #     all_tags = all_tags
    #     all_pre_tags = all_pre_tags
    # logger.info("进行测试集测试：")
    # logger.info(metrics.flat_classification_report([all_tags], [all_pre_tags], labels=[0, 1, 2, 3], digits=3))
    # F1 = metrics.flat_f1_score([all_tags], [all_pre_tags], average='weighted', labels=[0, 1, 2, 3])
    # return F1

def tmp():
    a = np.array([[1,2,3,4,5], [1,2,3,4], [1,2,3]])
    b = np.array([[1, 2, 3, 4, 6], [1, 2, 3, 4], [1, 2, 3]])
    logger.info(metrics.flat_f1_score(a, b, average='weighted', labels= [1,2,3,4,5,6]))
    # f1score = np.mean(metrics.f1_score(a, b, average=None))
    # logger.info(f1score)
    logger.info(metrics.flat_classification_report(a, b, labels=[1,2,3,4,5,6], digits=3))

if __name__ == '__main__':
    main()
    # tmp()