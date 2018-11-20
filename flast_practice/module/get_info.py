from pymongo import MongoClient


class GetInfo():

    def __init__(self):
        self.client = MongoClient("123.56.223.26:27017")
        self.db = self.client.raw_data
        self.thing_date = self.db.thing_date

    def insert_data(self, date_dict):
        self.thing_date.insert_one(date_dict)


    def get_thing(self, date):
        data_list = []
        data = self.thing_date.find({"date":date})
        for i in data:
            data_list.append(i)
        if data:
            return data_list
        else:
            return []

    def get_all_info(self):
        all_thing_list = []
        for i in self.thing_date.find():
            all_thing_list.append(i)
        return all_thing_list
