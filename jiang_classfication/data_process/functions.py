# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-14 11:08:45
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, label_alphabet, split_token='|||'):
    lines = open(input_file, "r", encoding= "utf-8")
    words = []
    word_Ids = []
    instence_texts = []
    instance_Ids = []
    for line in lines:
        # 分出句子和标签
        pairs = line.strip().split(split_token)
        # 句子
        sentence = pairs[0]
        # 标签
        label = pairs[1]
        # 得到label的Id
        label_Id = label_alphabet.get_index(label)
        # 对句子分词
        words_list = sentence.strip().split(" ")
        # 对每一个词加入到输出中，同时标签也加入进去
        for word in words_list:
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
        instence_texts.append([words, [], [], label])
        instance_Ids.append([word_Ids,[],[],label_Id])
        words = []
        word_Ids = []
    return instence_texts, instance_Ids

# def rad_instance(input_file, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length, sentence_classification=False, split_token='|||', char_padding_size=-1, char_padding_symbol = '</pad>'):
#     feature_num = len(feature_alphabets)
#     in_lines = open(input_file,'r', encoding="utf8").readlines()
#     instence_texts = []
#     instence_Ids = []
#     words = []
#     features = []
#     chars = []
#     labels = []
#     word_Ids = []
#     feature_Ids = []
#     char_Ids = []
#     label_Ids = []
#
#     ## if sentence classification data format, splited by \t
#     for line in in_lines:
#         if len(line) > 2:
#             pairs = line.strip().split(split_token)
#             sent = pairs[0]
#             if sys.version_info[0] < 3:
#                 sent = sent.decode('utf-8')
#             original_words = sent.split()
#             for word in original_words:
#                 words.append(word)
#                 if number_normalized:
#                     word = normalize_word(word)
#                 word_Ids.append(word_alphabet.get_index(word))
#                 ## get char
#                 char_list = []
#                 char_Id = []
#                 for char in word:
#                     char_list.append(char)
#                 if char_padding_size > 0:
#                     char_number = len(char_list)
#                     if char_number < char_padding_size:
#                         char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
#                     assert(len(char_list) == char_padding_size)
#                 for char in char_list:
#                     char_Id.append(char_alphabet.get_index(char))
#                 chars.append(char_list)
#                 char_Ids.append(char_Id)
#
#             label = pairs[-1]
#             label_Id = label_alphabet.get_index(label)
#             ## get features
#             feat_list = []
#             feat_Id = []
#             for idx in range(feature_num):
#                 feat_idx = pairs[idx+1].split(']',1)[-1]
#                 feat_list.append(feat_idx)
#                 feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
#             ## combine together and return, notice the feature/label as different format with sequence labeling task
#             if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
#                 instence_texts.append([words, feat_list, chars, label])
#                 instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
#             words = []
#             features = []
#             chars = []
#             char_Ids = []
#             word_Ids = []
#             feature_Ids = []
#             label_Ids = []
#     if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
#         instence_texts.append([words, feat_list, chars, label])
#         instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
#         words = []
#         features = []
#         chars = []
#         char_Ids = []
#         word_Ids = []
#         feature_Ids = []
#         label_Ids = []
#     return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    """
    构建词向量
    :param embedding_path:
    :param word_alphabet:
    :param embedd_dim:
    :param norm:
    :return:
    """
    embedd_dict = dict()
    if embedding_path != None:
        # 加载词向量
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])  # 没有这个词就会随机初始化
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    """
    加载词向量文件
    :param embedding_path:
    :return:
    """
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
