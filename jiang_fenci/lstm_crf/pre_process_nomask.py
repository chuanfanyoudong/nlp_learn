#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: pre_process.py
@time: 2019/1/9 11:13
"""
import logging
import os
import pickle
import traceback

import torch
from pymongo import MongoClient

"""
词向量的准备
句子向量的生成
"""

from conf.config import get_config
__config = get_config()
ROOT_DATA = __config["path"]["root"]
EMBEDDING_ROOT = __config["segment"]["embedding"]
TRAIN_DATA_PATH = __config["segment"]["lstm_train_data"]
TEST_DATA_PATH = __config["segment"]["lstm_test_data"]
VAL_DATA_PATH = __config["segment"]["lstm_val_data"]

def get_embedding(char_list):
    tag2id = {"B":0, "E":1, "O":2, "S":3}
    path = ROOT_DATA + EMBEDDING_ROOT
    if  os.path.exists(path):
        embedding_file = open(path, 'rb')
        embedding_dict =  pickle.load(embedding_file)
    else:
        embedding_dict = {}
        client = MongoClient('mongodb://192.168.10.27:27017/')
        admin = client.admin
        admin.authenticate("root", "nlp_dbroot1234")
        embedding_data = client.embeddings.tenlent_vector_200
        embedding_file = open(ROOT_DATA + EMBEDDING_ROOT, "wb")
        for vector_info in embedding_data.find({"$where":"this.word.length <2"}):
            word = vector_info["word"]
            vector = vector_info["vector"]
            embedding_dict[word] = vector
        pickle.dump(embedding_dict, embedding_file)
    word2id = {}
    n = 2
    for idx, word in enumerate(embedding_dict, 2):
        if word in char_list:
            word2id[word] = n
            n += 1
    # word2id = {word : idx if word in char_list else "<unk>" for idx, word in enumerate(embedding_dict, 2)}
    embedding_list = [embedding_dict[word] for word in word2id]
    word2id["<unk>"] = 0
    word2id["<pad>"] = 0
    print(len(word2id))
    return word2id, embedding_list, tag2id

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w]  if w in to_ix else to_ix["<unk>"] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_train_data():
    """
    获取训练数据
    :return:
    """
    train_data = []
    val_data = []
    test_data = []
    char_set = set()
    val_data_file = open(ROOT_DATA + VAL_DATA_PATH, "r", encoding= "utf-8")
    train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH, "r", encoding="utf-8")
    # new_train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH + "_", "w", encoding="utf-8")
    test_data_file = open(ROOT_DATA + TEST_DATA_PATH, "r", encoding="utf-8")
    n = 0
    for line in test_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        test_data.append(process_line(line))
    for line in val_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        val_data.append(process_line(line))
    for line in train_data_file:
        n += 1
        for char in line:
            if char != " ":
                char_set.add(char)
            # train_data_file.write(line)
        train_data.append(process_line(line))
    print(len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data, list(char_set)


def process_line(line = ""):
    # line = "“  征  而  未  用  的  耕地  和  有  收益  的  土地  ，  不准  荒芜  。"
    if line.strip() == "":
        return None
    line_list = line.strip().split("  ")
    while "" in line_list:
        line_list.remove("")
    sentence = "".join(line_list)
    tag_list = []
    for word in line_list:
        if line_list == []:
            continue
        if len(word) == 1:
            tag_list.append("S")
        elif len(word) == 2:
            tag_list.append("B")
            tag_list.append("E")
        else:
            tmp_tag_list = ["O" for char in word]
            tmp_tag_list[0] = "B"
            tmp_tag_list[-1] = "E"
            tag_list = tag_list + tmp_tag_list
    if len(list(sentence)) != len(tag_list):
        raise Exception("数据处理出错")
    return (list(sentence), tag_list)



def get_feature(method = "train"):
    if method not in ["train", "val", "test"]:
        logging.info("特征获取方法错误")
        raise Exception("特征获取方法错误")



if __name__ == '__main__':
    # print(get_embedding())
    prepare_sequence(1,2)
