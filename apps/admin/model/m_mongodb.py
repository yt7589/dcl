# MongoDB的公共类

class MMongoDb(object):
    def __init__(self):
        self.name = 'apps.admin.model.MMongoDb'

    @staticmethod
    def convert_recs(recs):
        '''
        将具有多条记录的cursor转为list[dict]
        '''
        print('recs: {0};'.format(recs))
        rows = []
        for rec in recs:
            row = {}
            for k, v in rec.items():
                if k != '_id':
                    row[k] = v
            rows.append(row)
        return rows

    @staticmethod
    def convert_rec(rec):
        '''
        将只有一条的结果转为dict
        '''
        print('rec: {0};'.format(rec))
        row = {}
        if rec is None:
            return row
        for k, v in rec.items():
            row[k] = v
        return row