#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: generate_fre_vocb.py
@time: 2019/1/5 15:23
"""

from conf.config import get_config

__config = get_config()
ROOT_PATH = __config["path"]["root"]
SEGMENT_PATH = __config["segment"]["split_data"]
FRE_DICT_PATH = __config["segment"]["dict_path"]


"""
生成词语的频率统计表
"""

def generate_fre_vocb():

    for data in ["pku", "msr"]:
        fre_dict = {}
        writer = open(ROOT_PATH + FRE_DICT_PATH + data + "_dict.txt", "w", encoding= "utf-8")
        origin_file = open(ROOT_PATH + SEGMENT_PATH + data + "_training.utf8", "r", encoding = "utf-8")
        for line in origin_file.readlines():
            line_list = line.strip().split("  ")
            for word in line_list:
                if word in fre_dict:
                    fre_dict[word] += 1
                else:
                    fre_dict[word] = 1
        for key in fre_dict:
            writer.write(key + "\t" + "tmp" + "\t" + str(fre_dict[key]) + "\n")


if __name__ == '__main__':
    generate_fre_vocb()