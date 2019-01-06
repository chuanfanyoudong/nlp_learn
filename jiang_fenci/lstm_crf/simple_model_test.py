#!/usr/bin/env python
# coding: utf-8


import torch
from jiang_fenci.lstm_crf.lstm_crf import BiLSTM_CRF, prepare_sequence
import marshal
import os
now_path = os.path.dirname((os.path.abspath(__file__)))
# print("crf", now_path)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 12
HIDDEN_DIM = 12

class ModelMain(object):

    def __init__(self):
        self.load_label()
        self.tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    # def prepare_sequence(self, seq, to_ix):
    #     idxs = [to_ix[w] if w in to_ix else 25 for w in seq]
    #     return torch.tensor(idxs, dtype=torch.long)

    def load_label(self):
        word_label = open(now_path + "/word_label.cache", "rb")
        self.word_to_ix = marshal.load(word_label)

    def main(self, sentence):
        # print(self.word_to_ix)
        # print(self.tag_to_ix)
        model = BiLSTM_CRF(len(self.word_to_ix), self.tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        model.load_state_dict(torch.load(now_path + "/lstm_crf_model", map_location='cpu'))
        model.eval()
        # sentence = "来到2"
        if sentence == "":
            return []
        split_sentence = list(sentence)
        inputs = prepare_sequence(split_sentence, self.word_to_ix)
        tag_scores = model(inputs)[1]
        # print(tag_scores)
        # tag_scores[1]
        result = []
        i = 0
        j = 0
        while i < len(sentence):
            if tag_scores[i] != 1:
                if j != i:
                    result.append(sentence[j:i])
                j = i

            i += 1
        if j != i:
            result.append(sentence[j:i])
        return result


if __name__ == '__main__':
    mm = ModelMain()
    sentence = "来 "
    mm.main(sentence)




