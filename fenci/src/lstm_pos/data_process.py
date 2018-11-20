# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/8/28 16:15
# @Ide     : PyCharm
"""
import re


import pandas as pd
import traceback


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
        try:
            line_final = pre_process(line).strip().split(" ")[1:]
            if line_final != [''] and line_final != []:
                print(line_final)
                word_list = [word_part.strip().split("/")[0] for word_part in line_final]
                part_list = [word_part.strip().split("/")[1] for word_part in line_final]
                training_data_origin.append(("#".join(word_list), "#".join(part_list)))
        except:
            traceback.print_exc()
df = pd.DataFrame(training_data_origin, columns= ["word", "pos"])
df.sample(frac= 1)
df_train = df[:17500]
df_val = df[17500:]
print(df)
df_train.to_csv("../../data/pos_train.csv")
df_val.to_csv("../../data/pos_val.csv")

