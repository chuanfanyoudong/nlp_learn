#!/usr/bin/env python3
import os, sys
import json
import timeit
from threading import Thread
import datetime
import traceback
from uuid import uuid4
from bson.objectid import ObjectId


sys.path.append('..')
sys.path.append('../modules')
from modules.common_lib.db.client import get_cli
try:
    from modules.kgqa.kgqa_api import KGQA
except:
    print(traceback.format_exc())
    KGQA = None
try:
    from modules.gossip.gossip_api import Gossip
except:
    Gossip = None


def add_row_id_to_one_result(one_result):
    for i, v in enumerate(one_result['result']):
        # one_result['result'][i]['row_id'] = '_'.join([str(uuid4()),str(log_id)])
        one_result['result'][i]['row_id'] = str(uuid4())
    return one_result


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
            self.client = get_cli()
            self.db = self.client['QAdemo']
            self.collection_log = self.db['chat_log']
        except:
            traceback.print_exc()
        try:
            self.kgqa = KGQA()
        except:
            traceback.print_exc()
            self.kgqa = None
        try:
            self.gossip = Gossip()
        except:
            traceback.print_exc()
            self.gossip = None

    @clock
    def kg_qa(self, kgqa_object, query: str):
        try:
            return [kgqa_object.qa(query)]
        except:
            traceback.print_exc()
            return [{'Alarm': 'ERROR'}]

    @clock
    def gossip_qa(self, gossip_object, query: str):
        try:
            return gossip_object.qa(query)
        except:
            traceback.print_exc()
            return [{'Alarm': 'ERROR'}]

    @clock
    def doc_qa(self, doc_object, query: str, tresult):
        try:
            return list(doc_object.do_doc_query(tresult, query))
        except:
            traceback.print_exc()
            return [{'Alarm': 'ERROR'}]

    def update_chat_log(self, result):
        try:
            print(result['log_id'])
            rate_result = result['rate_data']
            one = self.collection_log.find_one({'_id': ObjectId(result['log_id'])})
            one['rate_result'] = rate_result
            self.collection_log.save(one)
            return "rate succeed"
        except:
            traceback.print_exc()
            return "rate failed"

    def insert_chat_log(self, query, result):
        try:
            date = datetime.datetime.now()
            data = {'query': query, 'date': date}
            for r in result:
                data[r['name']] = r
            id = self.collection_log.insert(data)
            return str(id)
        except:
            traceback.print_exc()
            return "insert_log_error"

    def qa(self, query):
        threads = []
        for m in [(self.kg_qa, self.kgqa),
                  ]:  # (func, object)
            thread = QAModule(m[0], m[1], query)
            threads.append(thread)
            thread.start()
        result = []
        for thread in threads:
            thread.join()
            tresult = thread.get_result()
            result.append(tresult)
        for i, one_result in enumerate(result):
            result[i] = add_row_id_to_one_result(one_result)
        log_id = self.insert_chat_log(query, result)
        return {'query_result': result, 'log_id': log_id}


if __name__ == "__main__":
    qm = QueryManage()
    result = qm.qa('苏州的人口')
    print('result:', result)
