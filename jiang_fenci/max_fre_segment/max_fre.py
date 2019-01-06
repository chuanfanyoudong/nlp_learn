import os
from math import log
# import conf.config.get_config as get_config
from conf.config import get_config
__config = get_config()
dict_path = __config["path"]["root"] + __config["segment"]["dict_path"] + __config["segment"]["dict_name"]

class TokenGet():

    def __init__(self):
        self.FRDC = {}
        self.total = 0
        self.get_dict()
        self.sentence = ""

    def get_dict(self):
        """
        生成前缀字典
        :return: 前缀字典
        """
        with open(dict_path, "r", encoding= "utf-8") as file:
            changed = False
            fre_index = 1
            for line in file:
                line_list = line.strip().split("\t")
                if len(line_list) == 3:
                    word = line_list[0]
                    if not changed:
                        try:
                            fre = int(line_list[1])
                            fre_index = 1
                        except:
                            fre = int(line_list[2])
                            fre_index = 2
                        changed = True
                    else:
                        fre = int(line_list[fre_index])
                    self.total += fre
                    self.FRDC[word] = fre
                    for i in range(len(word) - 1):
                        if word[:i + 1] not in self.FRDC:
                            self.FRDC[word[:i + 1]] = 0

    def get_dag(self, reverse = True):
        '''
        :param sentence: 输入的问句
        :return: 返回词图
        '''
        dag_dict = {}
        length_sentence = len(self.sentence)
        if length_sentence == 0:
            return {}
        else:
            if reverse:
                for i in range(length_sentence):
                    dag_dict[i] = [i]
                    k = i
                    for z in range(k + 1, length_sentence):
                        if self.sentence[k:z + 1] in self.FRDC:
                            if self.FRDC[self.sentence[k:z]] != 0:
                                dag_dict[i].append(z)
                        else:
                            break
            else:
                for i in range(length_sentence):
                    dag_dict[i + 1] = [i]
                for i in range(length_sentence):
                    k = i
                    for z in range(k + 1, length_sentence):
                        if self.sentence[k:z + 1] in self.FRDC:
                            if self.FRDC[self.sentence[k:z]] != 0:
                                dag_dict[z + 1].append(i)
                        else:
                            break
            return dag_dict, reverse

    def dynamic(self, dag_dict, reverse = True):
        '''
        :param dag_dict: 输入词图
        :return: 输出基于动态规划的求解最大路径分词
        '''

        logtotal = log(self.total)
        route = {}
        N = len(self.sentence)
        if reverse:
            route[N] = (0, 0)
            for idx in range(N - 1, -1, -1):
                route[idx] = max((log(self.FRDC.get(self.sentence[idx:x + 1]) or 1) -
                                  logtotal + route[x + 1][0], x) for x in dag_dict[idx])
        else:
            route[0] = (0, 0)
            for idx in range(1, N + 1, 1):
                route[idx] = max((log(self.FRDC.get(self.sentence[x : idx]) or 1) -
                                  logtotal + route[x][0], x) for x in dag_dict[idx])
        return route

    def get_split_sentence(self, route, reverse = True):
        '''
        得到最终分词结果
        :param route:
        :return: 分词结果，list
        '''
        split_result = []
        # print(split_result)
        if route == 0:
            return []
        else:
            if reverse:
                i = 0
                while i < len(self.sentence):
                    target_idx = route[i][1] + 1
                    split_result.append(self.sentence[i:target_idx])
                    i = target_idx
            else:
                i = len(route) - 1
                while i > 0:
                    target_idx = route[i][1]
                    split_result.append(self.sentence[target_idx : i])
                    i = target_idx
                split_result = split_result[::-1]
            return split_result


    def main(self, sentence, reverse = True):
        if sentence.strip() != "" and isinstance(sentence, str):
            self.sentence = sentence
            dag, reverse = self.get_dag(reverse = reverse)
            # print(dag)
            route = self.dynamic(dag, reverse)
            # print(route)
            split_result = self.get_split_sentence(route, reverse = reverse)
            return split_result
        else:
            return []


if __name__ == '__main__':
    sentence = "杭州是浙江的省会"
    tg = TokenGet()
    split_result= tg.main(sentence, reverse= False)
    print(split_result)

"""
{0: [0, 1], 1: [1], 2: [2], 3: [3, 4], 4: [4], 5: [5], 6: [6, 7], 7: [7]}
{8: (0, 0), 7: (-6.481020630156426, 7), 6: (-10.58838240981175, 7), 5: (-15.827537895671869, 5), 4: (-25.025837749106383, 4), 3: (-25.725416880594402, 4), 2: (-30.048371342987327, 2), 1: (-38.2826476685701, 1), 0: (-40.02040921008014, 1)}
['杭州', '是', '浙江', '的', '省会']

{1: [0], 2: [1, 0], 3: [2], 4: [3], 5: [4, 3], 6: [5], 7: [6], 8: [7, 6]}
{0: (0, 0), 1: (-10.757719326176376, 0), 2: (-9.972037867092812, 0), 3: (-14.294992329485737, 2), 4: (-25.620373802386283, 3), 5: (-24.192871314408272, 3), 6: (-29.43202680026839, 5), 7: (-37.03626193605578, 6), 8: (-40.02040921008014, 6)}
['杭州', '是', '浙江', '的', '省会']
"""

