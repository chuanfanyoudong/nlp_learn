import logging
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class CommandsSearcher():
    def __init__(self):
        host = '0.0.0.0'
        port = '9200'
        self.es = Elasticsearch([{"host":host, "port":port}])
        self.index_name = 'commands'
        self.type_name = 'default'
        self.timeout = 1


    def exact_search(self, question):
        '''
        精确查找
        '''
        res = {}
        q = {
            "query": {"match":{"question_kw":question}},
            '_source':['question_kw', 'command'],
            'size':1
        }
        try:
            exact_res = self.es.search(index=self.index_name,doc_type=self.type_name, body=q, request_timeout=self.timeout)
        except Exception as e:
            logging.error("[search_err] %s" % e)
            return res
        
        exact_hits = exact_res["hits"]["total"]
        if exact_hits == 0:
                return res
        else:
            for exact_q in exact_res['hits']['hits']:
                return exact_q["_source"]


    def fuzzy_search(self, question):
        '''
        模糊查找
        '''
        res = []
        q = {
            "query": {"match":{"question_tx":question}},
            '_source':['question_tx', 'command'],
            'size':5
        }
        try:
            fuzzy_res = self.es.search(index=self.index_name,doc_type=self.type_name, body=q, request_timeout=self.timeout)
        except Exception as e:
            logging.error("[search_err] %s" % e)
            return res

        fuzzy_hits = fuzzy_res["hits"]["total"]
        if fuzzy_hits == 0:
            return res
        else:
            for fuzzy_q in fuzzy_res['hits']['hits']:
                res.append(fuzzy_q["_source"])
            return res


if __name__ == '__main__':
    cs = CommandsSearcher()
    while 1:
        question = input('\n需要查询的指令: ')

        exact_res = cs.exact_search(question)
        if exact_res:
            print(exact_res)

        else:
            fuzzy_res = cs.fuzzy_search(question)
            if fuzzy_res:
                for r in fuzzy_res:
                    print(r)
            else:
                print('索引结果为空，要么没查到，要么就没有')
