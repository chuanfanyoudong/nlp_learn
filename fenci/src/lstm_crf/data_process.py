# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/28 16:15
# @Ide     : PyCharm
"""
import re


import pandas as pd

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
            if sentence_list == []:
                continue
            # part_list = [word_part.strip().split("/")[1] for word_part in line_list]
            training_data_origin.append(("".join(sentence_list), "".join(b_i_o_list)))
            # training_data_origin.append((sentence_list, b_i_o_list))
df = pd.DataFrame(training_data_origin, columns= ["word", "tag"])
df.sample(frac= 1)
df_train = df[:17500]
df_val = df[17500:]
print(df)
df_train.to_csv("../../data/train.csv")
df_val.to_csv("../../data/val.csv")

