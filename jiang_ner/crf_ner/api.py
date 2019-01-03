# -*- coding: utf-8 -*-
"""
API
----
封装对外提供接口
"""
from jiang_ner.crf_ner.corpus import get_corpus
from jiang_ner.crf_ner.model import get_model

__all__ = ["pre_process", "train", "recognize"]


def pre_process():
    """
    抽取语料特征
    """
    corpus = get_corpus()
    corpus.pre_process()


def train():
    """
    训练模型
    """
    model = get_model() # 初始化标签序列、词序列、词性序列
    model.train()


def recognize(sentence):
    """
    命名实体识别
    """
    model = get_model()
    return model.predict(sentence)
