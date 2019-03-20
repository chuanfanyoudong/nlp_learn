#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: DATA_PROCESS.py
@time: 2019/3/5 9:42
"""

import jieba
import sys
from conf.config import get_config
__config = get_config()
DATA_PATH = __config["classfication"]["data_path"]
ROOT = __config["path"]["root"]
sys.path.append(ROOT)
DATA_PATH = ROOT + DATA_PATH
from jiang_classfication.data_process.alphabet import Alphabet
from jiang_classfication.data_process.functions import read_instance
#
#
def get_data(file_name):
    f = open(DATA_PATH + file_name)
    write_file = open(DATA_PATH + "lstm_text.train", "w", encoding= "utf-8")
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
            write_file.write(" ".join(list) + "|||")
            write_file.write(str(label_dict[label]) + "\n")
            label_list.append(label)
            corpus_list.append(" ".join(list))
    # print(corpus_list, label_list)
    return corpus_list, label_list

# get_data("cnews.train.txt")

class Data:

    """
    数据类：其基本操作流程有
    》》》1 读取配置
    》》》2 构建唯一id
    》》》3 生成数据集
    》》》4 配置预训练向量
    """

    def __init__(self):
        """
        初始化参数
        """
        # 文件路径
        self.train_dir = None
        self.test_dir = None
        self.val_dir = None
        self.word_emb_dir = None
        self.model_dir = "./"
        # 句子和标记的分隔符
        self.split_token = "|||"
        # 唯一标记类
        self.word_alphabet = Alphabet("word")
        self.label_alphabet = Alphabet("label", label= True)
        self.word_alphabet_size = 0
        self.label_alphabet_size = 0
        # 迭代默认次数配置
        self.HP_iteration = 5
        # 模型配置
        self.word_feature_extractor = "LSTM"
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.use_char = False
        self.HP_batch_size = 5
        self.sentence_classification = True
        self.word_emb_dim = 20
        self.pretrain_word_embedding = None
        self.feature_num = 0
        self.feature_emb_dims = 10
        self.HP_hidden_dim = 8
        self.optimizer = "sgd"
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8
        self.average_batch_loss = False
        self.tagScheme = "Not sequence labeling task"
        self.seg = False

    def show_data_summary(self):
        """
        打印Data类的描述信息包括各参数等
        :return:
        """
        print("标签数量为：{}".format(self.label_alphabet_size))
        print("token数量为：{}".format(self.word_alphabet_size))


    def build_alphabet(self, input_file):
        """
        用一个类来表示word，char的唯一标记，这里为填充这个类
        :param inputfile: 要构建唯一id的文件
        :return:
        """
        # 生成文件迭代器
        lines = open(input_file, "r", encoding= "utf-8").readlines()
        # 读取文件内容
        for line in lines:
            # 分出句子和标签
            pairs = line.strip().split(self.split_token)
            # 句子
            sentence = pairs[0]
            # 标签
            label = pairs[1]
            # 对句子分词
            words = sentence.strip().split(" ")
            for word in words:
                # 遍历句子分词后的记过，加到唯一标记类中
                self.word_alphabet.add(word)
            # 将标记也加入到标记类中
            self.label_alphabet.add(label)
        # 记录word，label的长度
        self.word_alphabet_size = self.word_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()


        pass

    def build_pretrained_emb(self):
        """
        构建词向量
        :return:
        """
        if self.word_emb_dir:
            print("正在构建训练集词向量，向量文件路径是：{}".format(self.word_emb_dir))

    def generate_instance(self, name):
        """
        生成训练、测试、验证集合
        :param name:
        :return:
        """
        # 生成验证集
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet, self.label_alphabet, self.split_token)
        # 生成验证集
        elif name == "val":
            self.dev_texts, self.dev_Ids = read_instance(self.val_dir, self.word_alphabet,self.label_alphabet, self.split_token)
        # 生成测试集
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet, self.label_alphabet, self.split_token)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))




    def read_config(self, config_dict):
        """
        读取配置文件，输入为字典样式
        :param config_dict: 配置参数的字典
        :return:
        """
        # 配置训练集路径
        the_item = "train_dir"
        if the_item in config_dict:
            self.train_dir = ROOT + config_dict[the_item]
        # 配置验证集路径
        the_item = "val_dir"
        if the_item in config_dict:
            self.val_dir =  ROOT + config_dict[the_item]
        # 配置测试集路径
        the_item = "test_dir"
        if the_item in config_dict:
            self.test_dir =  ROOT + config_dict[the_item]
        # 配置句子和标记的分隔符
        the_item = "split_token"
        if the_item in config_dict:
            self.split_token = config_dict[the_item]
        # 模型的存储路径
        the_item = "model_dir"
        if the_item in config_dict:
            self.model_dir =  ROOT + config_dict[the_item]
        # 是否使用GPU
        the_item = "HP_gpu"
        if the_item in config_dict:
            self.HP_gpu = bool(config_dict[the_item])
        # 迭代次数
        the_item = "HP_iteration"
        if the_item in config_dict:
            self.HP_iteration = int(config_dict[the_item])
        the_item = 'seg'
        if the_item in config_dict:
            self.seg = bool(int(config_dict[the_item]))
        # 是否使用字符级别特征
        the_item = 'use_char'
        if the_item in config_dict:
            self.use_char = int(config_dict[the_item])
        # 优化器选择
        the_item = 'optimizer'
        if the_item in config_dict:
            self.optimizer = config_dict[the_item]

if __name__ == '__main__':
    data = Data()
    data.read_config(__config["deep_classfication"])
