# 基于mongodb的文件存储
import pymongo
from gridfs import GridFS

class MFileOdb(object):
    def __init__(self):
        self.name = 'apps.model.MFileOdb'
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db = mongo_client['tcvdb']
        self.tbl = self.db['file_odb']

    def insert(self, file_path, query):
        print('文件对象数据库')
        fs = GridFS(self.db, self.tbl)

