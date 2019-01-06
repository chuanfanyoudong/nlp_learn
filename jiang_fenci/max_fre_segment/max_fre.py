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

    def get_dag(self):
        '''
        :param sentence: 输入的问句
        :return: 返回词图
        '''
        dag_dict = {}
        length_sentence = len(self.sentence)
        if length_sentence == 0:
            return {}
        else:
            for i in range(length_sentence):
                dag_dict[i] = [i]
                k = i
                for z in range(k + 1, length_sentence):
                    if self.sentence[k:z + 1] in self.FRDC:
                        if self.FRDC[self.sentence[k:z]] != 0:
                            dag_dict[i].append(z)
                    else:
                        break
            return dag_dict

    def dynamic(self, dag_dict):
        '''
        :param dag_dict: 输入词图
        :return: 输出基于动态规划的求解最大路径分词
        '''

        route = {}
        N = len(self.sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.FRDC.get(self.sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in dag_dict[idx])
        return route

    def get_split_sentence(self, route):
        '''
        得到最终分词结果
        :param route:
        :return: 分词结果，list
        '''
        split_relult = []
        if route == 0:
            return []
        else:
            i = 0
            while i < len(self.sentence):
                target_idx = route[i][1] + 1
                split_relult.append(self.sentence[i:target_idx])
                i = target_idx
            return split_relult

    def main(self, sentence):
        self.sentence = sentence
        dag = self.get_dag()
        route = self.dynamic(dag)
        split_result = self.get_split_sentence(route)
        return split_result


if __name__ == '__main__':
    sentence = "杭州是浙江的省会"
    tg = TokenGet()
    split_result= tg.main(sentence)
    print(split_result)
