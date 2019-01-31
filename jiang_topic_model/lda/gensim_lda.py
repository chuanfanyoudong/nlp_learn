#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: gensim_lda.py
@time: 2019/1/24 16:57
"""
from gensim.models import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
lda_model = LdaModel(common_corpus, num_topics=10)
other_texts = [
     ['computer', 'time', 'graph'],
     ['survey', 'response', 'eps'],
     ['human', 'system', 'computer']
]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
# lda_model.update(other_corpus)
# vector = lda_model[unseen_doc]
# print(common_corpus[0])
vector = lda_model[common_corpus[0]]
print(vector)
