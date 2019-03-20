#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: split_data.py
@time: 2019/3/11 20:18
"""
import sys

import os

import random
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
ROOT = cur_dir_path + "/../"
print(os.listdir(ROOT))
sys.path.append(ROOT)
from conf.config import get_config
from jiang_classfication.model.sentclassifier import SentClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

__config = get_config()
DATA_PATH = __config["deep_classfication"]["concept_data"]
ROOT = __config["path"]["root"]
sys.path.append(ROOT)
DATA_PATH = ROOT + DATA_PATH



"""
此函数的作用是按照比例拆分出训练集、验证集、测试集
"""

def split_data(file_path, train_part, val_part, test_part):
    """
    返回三个文件
    :param file_path:总数据路径
    :param train_part: 训练集比例
    :param val_part: 验证集比例
    :param test_part: 测试集比例
    :return:
    """
    train_data_path = file_path + "_train"
    train_data_file = open(train_data_path, "w", encoding= "utf-8")
    val_data_path = file_path + "_val"
    val_data_file = open(val_data_path, "w", encoding="utf-8")
    test_data_path = file_path + "_test"
    test_data_file = open(test_data_path, "w", encoding="utf-8")
    f = open(file_path, "r", encoding= "utf-8")
    all_data = []
    for line in f:
        all_data.append(line)
    random.shuffle(all_data)
    length_data = len(all_data)
    first_dot = int(train_part/(train_part + val_part + test_part) * length_data)
    second_dot = int((train_part + val_part)/(train_part + val_part + test_part)* length_data)
    train_part_data = all_data[:first_dot]
    val_part_data= all_data[first_dot:second_dot]
    test_part_data = all_data[second_dot:]
    for line in train_part_data:
        train_data_file.write(line)
    for line in val_part_data:
        val_data_file.write(line)
    for line in test_part_data:
        test_data_file.write(line)


if __name__ == '__main__':
    split_data(DATA_PATH + "/entity_type", 7,2,1)
