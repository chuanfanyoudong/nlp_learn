#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: lda.py
@time: 2019/2/21 17:37
"""

# LDA model with Gibbs Sampling
# the implementation is based on
# Darling W M. A theoretical and practical implementation tutorial on topic modeling and gibbs sampling, 2011.
# Gregor Heinrich, Parameter estimation for text analysis, 2004
# Many variables may have sparse property which may help optimize
# computation.
import numpy as np
import random
from collections import OrderedDict
import os


class Documents:
    def __init__(self, data=None, dir=None):
        if data is not None:
            self.documents = []
            self.ndoc = len(data)
            self.dict = {}
            self.reverse_dict = {}
            self.nword = 0
            for document in data:
                self.documents.append(np.zeros(len(document)))
                for i in range(len(document)):
                    word = document[i]
                    if not word in self.dict:
                        self.dict[word] = self.nword
                        self.reverse_dict[self.nword] = word
                        self.nword += 1
                    self.documents[-1][i] = self.dict[word]
        else:  # input a directory
            pass


class LDA:
    def __init__(self, K=2, alpha=0.1, beta=0.1):
        self.K = K  # 主题数目
        self.alpha = 0.1
        self.beta = 0.1
        # the following variable is correspond to the link provided
        self.n_d_k = None
        self.n_k_word = None
        self.n_k = None
        self.phi = None
        self.theta = None

    def fit(self, data, iter_times=100, detailed=False):
        # initial variables
        self.n_d_k = np.zeros((data.ndoc, self.K))  # 每个文档的主题频率
        self.n_k_word = np.zeros((self.K, data.nword))  # 每个主题下每个词的频率
        self.n_k = np.zeros(self.K)
        self.n_d = np.zeros(data.ndoc)
        self.p = np.zeros(self.K)  # is not normalized
        z = [np.zeros(len(document)) for document in data.documents]  # Here we will only use the shape
        for d in range(data.ndoc):
            document = data.documents[d]
            for w in range(len(document)):
                word = document[w]
                k = np.random.randint(self.K)
                z[d][w] = k
                self.n_d_k[d, k] += 1
                self.n_k_word[k, int(word)] += 1
                self.n_k[k] += 1
                self.n_d[d] += 1
        # Gibbs Sampling
        for epoch in range(iter_times):
            if detailed:
                print("Epoch:", epoch)
            for d in range(data.ndoc):
                document = data.documents[d]
                for w in range(len(document)):
                    word = document[w]  # 吉布斯采样，对每篇文档的每一个词
                    k = int(z[d][w])
                    self.n_d_k[d, k] -= 1
                    self.n_k_word[k, int(word)] -= 1
                    self.n_k[k] -= 1
                    self.n_d[d] -= 1
                    self.p = (self.n_d_k[d, :] + self.alpha) * (self.n_k_word[:, int(word)] + self.beta) / (
                                self.n_k + self.beta * data.nword)
                    p = np.random.uniform(0, np.sum(self.p))
                    # print("p:", p)
                    # print("self.p:", self.p)
                    for k in range(self.K):  # 逐项累计求得，目的是重新分配p的概率分布，按照各主题概率的大小分布
                        if p <= self.p[k]:
                            z[d][w] = k
                            self.n_d_k[d, int(k)] += 1
                            self.n_k_word[k, int(word)] += 1
                            self.n_k[k] += 1
                            self.n_d[d] += 1
                            break
                        else:
                            p -= self.p[k]
        self.phi = np.zeros((self.K, data.nword))
        for k in range(self.K):
            self.phi[k, :] = (self.n_k_word[k, :] + self.beta) / (self.n_k[k] + self.beta)
        self.theta = np.zeros((data.ndoc, self.K))
        for d in range(data.ndoc):
            self.theta[d, :] = (self.n_d_k[d, :] + self.alpha) / (self.n_d[d] + self.alpha)
