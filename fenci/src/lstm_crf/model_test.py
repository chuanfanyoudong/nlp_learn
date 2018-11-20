import re

import torch
from simple_lstm_crf import BiLSTM_CRF

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


def similary(list1, list2):
    score = sum([1 if list1[i] == list2[i] else 0for i in range(len(list1))])
    return score/len(list1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

reverse_dict = {}
for key in tag_to_ix:
    value = tag_to_ix[key]
    reverse_dict[value] = key

def double(matched):
    line= matched.group(0)[1:-1]
    line_list = line.strip().split("  ")
    final_lien = "".join([word.strip().split("/")[0] for word in line_list])
    # print(final_lien)
    return final_lien + "/"

def pre_process(line):
    pattern1 = re.compile(r'\[.*\]')
    num = re.sub(r'\[.*?\]', double, line)
    return num


def get_word_id():
    with open("/data/users/zkjiang/fenci/data/renmin.txt", "r", encoding= "utf-8") as file:
        training_data_origin = []
        for line in file:
            if line != "":
    #         print(line)
    #         line.replace("  "," ")
    #         line.replace(" ", "  ")
                line_final = pre_process(line)
    #             print(line_final)
                b_i_o_list = []
                line_list = line_final.strip().split(" ")[1:]
                word_list = [word_part.strip().split("/")[0] for word_part in line_list]
                sentence_list = []
                for word in word_list:
                    sentence_list += list(word)
                    if len(word) == 1:
                        b_i_o_list.append("O")
                    else:
                        b_i_o_list.append("B")
                        for i in word[:-1]:
                            b_i_o_list.append("I")
                # part_list = [word_part.strip().split("/")[1] for word_part in line_list]
                training_data_origin.append((sentence_list, b_i_o_list))
    training_data = training_data_origin[1:10]
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix
# reverse_dict = {}
import numpy as np
word_to_ix = get_word_id()
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("./lstm_crf_model"))
model.eval()

score_list = []
print(reverse_dict)
while 1:
    input = input("请输入：")
    input= list(input)
    try:
        inputs = prepare_sequence(input, word_to_ix)
        #         print(training_data[2])

        tag_scores = model(inputs)
        print(tag_scores)
    except:
        print(1)
# with torch.no_grad():
#     for i in training_data:
#         if len(i[0]) == 0:
#             continue
#         inputs = prepare_sequence(["我","来","到","中","国"], word_to_ix)
# #         print(training_data[2])
#         tag_scores = model(inputs)
#         print(tag_scores)
#         print(i[1])

#         # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#         # for word i. The predicted tag is the maximum scoring tag.
#         # Here, we can see the predicted sequence below is 0 1 2 0 1
#         # since 0 is index of the maximum value of row 1,
#         # 1 is the index of maximum value of row 2, etc.
#         # Which is DET NOUN VERB DET NOUN, the correct sequence!
# #         tag_scores.argmax(dim=1)
#     #     print(tag_scores.argmax(dim=1))
#         final_result = [reverse_dict[i] for i in tag_scores[1]]
#         score = similary(i[1],final_result )
#         print(i)
#         print(final_result)
#         score_list.append(score)
# print(np.mean(score_list))