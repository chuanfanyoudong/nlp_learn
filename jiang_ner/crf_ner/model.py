# -*- coding: utf-8 -*-
"""
NER
----
封装条件随机场命名实体识别
"""
import os

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
from jiang_ner.crf_ner.util import q_to_b
from jiang_ner.crf_ner.corpus import get_corpus
from flast_practice.config import get_config
#
#
#
# DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "nlp_learn/data/ner_data/"))
__model = None


class NER:

    def __init__(self):
        self.corpus = get_corpus()
        self.corpus.initialize()
        self.config = get_config()
        self.root = self.config.get('path', 'root')
        self.model = None
        # self.load_model()


    def initialize_model(self):
        """
        初始化
        """
        algorithm = self.config.get('model', 'algorithm')
        c1 = float(self.config.get('model', 'c1'))
        c2 = float(self.config.get('model', 'c2'))
        max_iterations = int(self.config.get('model', 'max_iterations'))
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        """
        训练
        """
        self.initialize_model()
        x, y = self.corpus.generator()
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        # self.load_model()
        labels = list(self.model.classes_)
        # labels.remove('O')
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        self.save_model()

    def predict(self, sentence):
        """
        预测
        """
        self.load_model()
        u_sent = q_to_b(sentence)
        word_lists = [[u'<BOS>']+[c for c in u_sent]+[u'<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        entity_list = []
        entity = ''
        for index in range(len(y_predict[0])):
            if y_predict[0][index][0] == u'B':

                if entity != "":
                    entity_list.append(entity)
                entity = ''
                entity += u_sent[index]
            if y_predict[0][index][0] == u'I':
                # entity = ''
                entity += u_sent[index]
            # elif y_predict[0][index - 1] == u'O':

            # elif entity[-1] != u' ':
            #     entity += u' '
        if entity != "":
            entity_list.append(entity)
        return entity_list, y_predict

    def load_model(self, name='model'):
        """
        加载模型 
        """
        model_path = self.config.get('model', 'model_path').format(name)
        # print(model_path)
        self.model = joblib.load(self.root + model_path)

    def save_model(self, name='model'):
        """
        保存模型
        """
        model_path = self.config.get('model', 'model_path').format(name)
        # root = self.config.get('path', 'root')
        joblib.dump(self.model, self.root + model_path)


def get_model():
    """
    单例模型获取
    """
    global __model
    if not __model:
        __model = NER()
    return __model
