#!/usr/bin/env python3
import os, sys
import json
import timeit
from threading import Thread
import datetime
import traceback
from uuid import uuid4
from bson.objectid import ObjectId

current_dir, filename = os.path.split(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '../modules'))
from modules.common_lib.db.client import get_cli
from modules.kgqa.kgqa_api import KGQA
from modules.qa_retrieval.qa_retrieval_api import QaRetrieval
#from modules.gossip.gossip_api import Gossip
from modules.query_classifier.query_classifier import QueryClassification


# 计时函数，返回了包装了的结果字典，多了name（函数名）与elapsed（函数耗时）属性
def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        return {'name': name, 'result': result, 'elapsed': elapsed}

    return clocked


class QAModule(Thread):
    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


class QueryManage():
    def __init__(self):
        try:
            self.query_classifier = QueryClassification()
            self.kgqa = KGQA()
            self.qa_retri = QaRetrieval()
            #self.gossip = Gossip()
        except:
            traceback.print_exc()

    @clock
    def module_qa(self, module_object, query: str):
        try:
            return module_object.qa(query)
        except:
            traceback.print_exc()
            return {'Answer': 'NULL'}

    def multi_qa(self, query):
        threads = []
        for m in [self.kgqa, self.qa_retri]:#[object, object]
            thread = QAModule(self.module_qa, m, query)
            threads.append(thread)
            thread.start()
        result = []
        for thread in threads:
            thread.join()
            tresult = thread.get_result()
            result.append(tresult['result'])
        return result

    def qa(self, query):
        t0 = timeit.default_timer()
        #['kg', 'retrieve', 'why', 'generate']
        label = self.query_classifier.classification(query)
        ret = {'type':0, 'info':{'Answer':'NULL'}}
        if not label:#包含敏感词，拒绝回答
            pass
        elif ['kg'] == label:#只查KGQA
            answer = self.kgqa.qa(query)
            ret['info'] = answer
            if 'Template' in answer:
                ret['type'] = 1
        elif 'kg' in label:#两个都查
            answer = {'Answer':'NULL'}
            for x in self.multi_qa(query):
                if x['Answer'] != 'NULL':
                    answer = x
                    break
            ret['info'] = answer
            if 'Template' in answer:
                ret['type'] = 1
            elif 'Sim' in answer:
                ret['type'] = 2
        else:#只查检索
            answer = self.qa_retri.qa(query)
            ret['info'] = answer
            if 'Sim' in answer:
                ret['type'] = 2
        elapsed = timeit.default_timer() - t0
        return {'err_code':0, 'data':{'res':[ret]}, 'elapsed':elapsed}


if __name__ == "__main__":
    qm = QueryManage()
    result = qm.qa('苏州的人口')
    print('Result:\n', result)
    print('-'*50)

    while True:
        query = input('query:')
        result = qm.qa(query.strip())
        print('Result:\n', result)
        print('*'*50)
