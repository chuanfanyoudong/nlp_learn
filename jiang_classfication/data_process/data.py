#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: DATA_PROCESS.py
@time: 2019/3/5 9:42
"""

import time

import jieba
import os
from fastText import train_supervised
import sys
from conf.config import get_config
__config = get_config()
DATA_PATH = __config["classfication"]["data_path"]
ROOT = __config["path"]["root"]
sys.path.append(ROOT)
DATA_PATH = ROOT + DATA_PATH
#
#
def get_data(file_name):
    f = open(DATA_PATH + file_name)
    write_file = open(DATA_PATH + "bert_text.val", "w", encoding= "utf-8")
    num = 0
    label_dict = {}
    label_num = 0
    corpus_list = []
    label_list = []
    for line in f:
        # print(line)
        num += 1
        line = line.strip()
        label = line[:2]
        if label not in label_dict:
            label_dict[label] = label_num
            label_num += 1
        content = line[3:]
        # if type in data_dict:
        list = jieba.lcut(content)
        # print(list)
        if list != []:
            write_file.write(str(label_dict[label]))
            write_file.write(" ".join(list) + "\n")
            label_list.append(label)
            corpus_list.append(" ".join(list))
    # print(corpus_list, label_list)
    return corpus_list, label_list

get_data("cnews.val.txt")


