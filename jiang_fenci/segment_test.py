#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: segment_test.py
@time: 2019/1/4 17:12
"""
sentence = "我来到家里"
import logging
from conf.config import get_config
__config = get_config()
import jieba
from jiang_fenci.max_fre_segment.max_fre import TokenGet
from jiang_fenci.hmm_segment.segment.model import Segment
from jiang_fenci.lstm_crf.simple_model_test import ModelMain
mm = ModelMain()
tg = TokenGet()
print(mm.main(sentence))
hmm_sg = Segment()
# print(hmm_sg.cut())
# hmm_sg = TokenGet()
# tg = TokenGet()

# print(tg.main("我来到家里"))
ROOT = __config["path"]["root"]
SPLIT_DATA = __config["path"]["split_data"]
# print(ROOT)

def segment_test(file = "pku", train = False, train_file = "pku", test_function = jieba.lcut):
    writer= open("result.txt", "w", encoding= "utf-8")

    origin_file = file + "_origin.utf8"
    segment_file = file + "_segment.utf8"
    # print(test_function("我是好人"))
    if file not in ["pku", "msr"]:
        logging.info("输入格式有误")
        return Exception("输入格式错误")
    # print(SPLIT_DATA)
    for data in ["pku", "msr"]:
        TP = 0
        REAL = 0
        POSITIVE = 0
        origin_file = data + "_origin.utf8"
        segment_file = data + "_segment.utf8"
        origin_file_list = open(ROOT + SPLIT_DATA + origin_file, "r", encoding = "utf-8").readlines()
        segment_file_list = open(ROOT + SPLIT_DATA + segment_file, "r", encoding="utf-8").readlines()
        for i in range(len(origin_file_list)):
            test_segment = test_function(origin_file_list[i].strip())
            writer.write("  ".join(test_segment) + "\n")
            if test_segment == []:
                break
            real_segment = segment_file_list[i].strip().split("  ")
            # print(test_segment)
            # print(real_segment)
            REAL += len(real_segment)
            POSITIVE += len(test_segment)
            test_segment_dict = trans_list(test_segment)
            real_segment_dict = trans_list(real_segment)
            for i in real_segment_dict:
                if i in test_segment_dict and test_segment_dict[i] == real_segment_dict[i]:
                    TP += 1
        RECAL = TP/REAL
        PRECITION = TP/POSITIVE
        F1 = 2 * TP /(REAL + POSITIVE)
        print("数据集{}\n准确率:{}\n召回率:{}\nF1:{}\n".format(data, PRECITION, RECAL, F1))

        # print(test_line_dict)

def trans_list(segment_list):
    segment_dict = {}
    index = 0
    for i in segment_list:
        tail = index + len(i)
        segment_dict[index] = tail
        index = tail
    return segment_dict


segment_test(test_function = tg.main, file= "msr")




