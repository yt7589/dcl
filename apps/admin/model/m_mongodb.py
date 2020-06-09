# MongoDB的公共类

class MMongoDb(object):
    def __init__(self):
        self.name = 'apps.admin.model.MMongoDb'

    @staticmethod
    def convert_recs(recs):
        '''
        将具有多条记录的cursor转为list[dict]
        '''
        rows = []
        for rec in recs:
            row = {}
            for k, v in rec.items():
                row[k] = v
            rows.append(row)
        return rows

    @staticmethod
    def convert_rec(rec):
        '''
        将只有一条的结果转为dict
        '''
        row = {}
        for k, v in rec.items():
            row[k] = v
        return row