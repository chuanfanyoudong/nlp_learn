import codecs
import json
import logging
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class DataManage():
    def __init__(self):
        host = '0.0.0.0'
        port = '9201'
        self.es = Elasticsearch([{"host":host, "port":port}])
        self.index_name = 'commands'
        self.type_name = 'default'


    def create_index(self):
        '''
        创建index
        '''
        index_mappings = {
            'mappings':{
                self.type_name:{
                    'properties':{
                        'question_tx':{'type':'text'},
                        'question_kw':{'type':'keyword'},
                        'command':{'type':'keyword', 'index':'false'},
                    }
                }
            }
        }

        self.es.indices.delete(index=self.index_name)
        print('delete success')
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=index_mappings)
            print('(create_index) created index: '+ self.index_name)
            return True
        else:
            print('(create_index) index already exist: '+ self.index_name)
            return False


    def insert_data(self):
        '''
        插入数据
        '''
        question = []
        with codecs.open('data.txt', 'r', 'utf-8') as infs:
            for inf in infs:
                inf = json.loads(inf.strip())
                question.append({
                    'question_tx': inf['question'],
                    'question_kw': inf['question'],
                    'command': inf['command'],
                })

        try:
            def gendata():
                for q in question:
                    q['_index'] = self.index_name
                    q['_type'] = self.type_name
                    yield q
            helpers.bulk(self.es, gendata())
        except Exception as e:
            logging.error("[insert_err] %s" % e)


if __name__ == '__main__':
    dm = DataManage()
    dm.create_index()
    dm.insert_data()
