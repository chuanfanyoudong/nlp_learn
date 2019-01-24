#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: sequence_label.py
@time: 2019/1/21 13:38
"""

def decode(sentence, tag, tag_num_dict):
    if tag == []:
        return sentence
    if len(tag) == 1:
        return sentence
    final_result = []
    origin_tag = [tag_num_dict[i] for i in tag]
    i = 0
    j = 0
    entity = ""
    for i in range(len(origin_tag)):
        if origin_tag[i] == "S":
            final_result.append(sentence[i])
            j = i + 1
        elif origin_tag[i] == "B":
            j = i
            entity += sentence[i]
            # i += 1
        elif origin_tag[i] == "E":
            entity += sentence[i]
            final_result.append(entity)
            entity = ""
            j = 0
            j = i + 1
        else:
            entity += sentence[i]
        # else:
        #     i += 1

    return final_result

if __name__ == '__main__':
    sentence = "我是中国好人"
    tag = [6,6,3,5,6,4]
    tag_num_dict = {3: 'B', 4: 'E', 5: 'O', 6: 'S', 1: '<SOS>', 0: '<PAD>', 2: '<EOS>'}
    result = decode(sentence, tag, tag_num_dict)
    print(result)