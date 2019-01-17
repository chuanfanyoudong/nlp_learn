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
from torch.utils.data import TensorDataset, DataLoader

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
    tag2id = {"B":0, "E":1, "O":2, "S":3, "<START>": 4, "<STOP>": 5}
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
    word2id["<unk>"] = 1
    word2id["<pad>"] = 0
    # word2id = {word : idx if word in char_list else "<unk>" for idx, word in enumerate(embedding_dict, 2)}
    default_vector = [0. for _ in range(200)]
    embedding_list = []
    reverse_word2id = {}
    default_vector = [0. for _ in range(200)]
    for key, value in word2id.items():
        reverse_word2id[value] = key
    for i in range(len(word2id)):
        word = reverse_word2id[i]
        if word in embedding_dict:
            embedding_list.append(embedding_dict[word])
        else:
            embedding_list.append(default_vector)
    print(len(word2id))
    return word2id, embedding_list, tag2id

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w]  if w in to_ix else to_ix["<unk>"] for w in seq]
    if __config["segment"]["cuda"] == "1":
        return torch.tensor(idxs, dtype=torch.long).cuda()
    else:
        return torch.tensor(idxs, dtype=torch.long)

def generate_iteration(data, word2id, tag2id, char_list, data_length):
    x = torch.tensor(
        [[word2id[word] if word in word2id else word2id["<unk>"] for word in single_data[0]] for single_data in
         data])
    y = torch.tensor(
        [[tag2id[tag] if tag in tag2id else tag2id["<unk>"] for tag in single_data[1]] for single_data in data])
    length = torch.tensor(data_length)
    tensor_data = TensorDataset(x, y, length)
    dataloader = DataLoader(tensor_data, batch_size=int(__config["segment"]["lstm_batch_size"]))
    return dataloader

def get_train_data_(data_all = 0):
    """
    获取训练数据
    :return:
    """
    train_data = []
    val_data = []
    test_data = []
    train_data_length = []
    val_data_length = []
    test_data_length = []
    char_set = set()
    if data_all:
        val_data_file = open(ROOT_DATA + VAL_DATA_PATH, "r", encoding= "utf-8")
        train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH, "r", encoding="utf-8")
        # new_train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH + "_", "w", encoding="utf-8")
        test_data_file = open(ROOT_DATA + TEST_DATA_PATH, "r", encoding="utf-8")
    else:
        val_data_file = open(ROOT_DATA + VAL_DATA_PATH + "_min", "r", encoding="utf-8")
        train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH + "_min", "r", encoding="utf-8")
        # new_train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH + "_", "w", encoding="utf-8")
        test_data_file = open(ROOT_DATA + TEST_DATA_PATH + "_min", "r", encoding="utf-8")
    n = 0
    for line in test_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        processd_line = process_line(line)
        if processd_line:
            test_data.append(processd_line[0])
            test_data_length.append(processd_line[1])
    for line in val_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        processd_line = process_line(line)
        if processd_line:
            val_data.append(processd_line[0])
            val_data_length.append(processd_line[1])
    for line in train_data_file:
        n += 1
        for char in line:
            if char != " ":
                char_set.add(char)
            # train_data_file.write(line)
        processd_line = process_line(line)
        if processd_line:
            train_data.append(processd_line[0])
            train_data_length.append(processd_line[1])
    print(len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data, list(char_set), train_data_length, val_data_length, test_data_length



def get_train_data(data_all=0):
    """
    获取训练数据
    :return:
    """
    train_data = []
    val_data = []
    test_data = []
    char_set = set()
    if data_all:
        val_data_file = open(ROOT_DATA + VAL_DATA_PATH, "r", encoding="utf-8")
        train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH, "r", encoding="utf-8")
        test_data_file = open(ROOT_DATA + TEST_DATA_PATH, "r", encoding="utf-8")
    else:
        val_data_file = open(ROOT_DATA + VAL_DATA_PATH + "_min", "r", encoding="utf-8")
        train_data_file = open(ROOT_DATA + TRAIN_DATA_PATH + "_min", "r", encoding="utf-8")
        test_data_file = open(ROOT_DATA + TEST_DATA_PATH + "_min", "r", encoding="utf-8")
    n = 0
    for line in test_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        if line.strip("\n").strip(" ") != "":
            processed_line = process_line(line)
            if processed_line != None:
                test_data.append(processed_line)
    for line in val_data_file:
        for char in line:
            if char != " ":
                char_set.add(char)
        if line.strip("\n") != "":
            processed_line = process_line(line)
            if processed_line != None:
                val_data.append(processed_line)
    for line in train_data_file:
        n += 1
        for char in line:
            if char != " ":
                char_set.add(char)
            # train_data_file.write(line)
        if line.strip("\n") != "":
            processed_line = process_line(line)
            if processed_line != None:
                train_data.append(processed_line)
    # train_data = train_data[:int(len(train_data) * data_percentage)]
    # test_data = test_data[:int(len(test_data) * data_percentage)]
    # val_data = val_data[:int(len(val_data) * data_percentage)]
    print(len(train_data), len(val_data), len(test_data))

    return train_data, val_data, test_data, list(char_set)

def process_line_(line = "", max_length = 209):
    # line = "“  征  而  未  用  的  耕地  和  有  收益  的  土地  ，  不准  荒芜  。"
    if line.strip() == "":
        return None
    line_list = line.strip("\n").split("  ")
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
    list_sentence= list(sentence)
    real_length = len(list_sentence)
    while len(list_sentence) < max_length:
        list_sentence.append("<pad>")
        tag_list.append("T")
    if len(list_sentence) != len(tag_list) or len(list_sentence) > max_length:
        print(list_sentence)
        raise Exception("数据处理出错")
    return (list_sentence, tag_list), real_length

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
