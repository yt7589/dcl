#
import random
from apps.vbg.model.m_vehicle_brand import MVehicleBrand

class CVbg(object):
    @staticmethod
    def get_survey_data(question_num):
        model = MVehicleBrand()
        total = model.get_total_recs()
        vehicle_brand_ids = set()
        survey = []
        question = CVbg.create_question(model, total, vehicle_brand_ids)
        print(question)
        survey.append(question)

    @staticmethod
    def create_question(model, total, vehicle_brand_ids):
        question = {}
        rec = model.get_random_rec(total)
        while rec['vehicle_brand_id'] in vehicle_brand_ids:
            rec = model.get_random_rec(total)
        vehicle_brand_ids.add(rec['vehicle_brand_id'])
        question['vbicon'] = 'displayVbicon/vbicon_{0}.jpg'.format(rec['vehicle_brand_id'])
        question['answer'] = rec['vehicle_brand_id']
        question['choose'] = -1
        pos = random.randint(1, 4)
        options = []
        for idx in range(1, pos):
            opt = model.get_random_rec(total)
            while opt['vehicle_brand_id'] == rec['vehicle_brand_id']:
                opt = model.get_random_rec(total)
            options.append(opt)
        options.append(rec)
        for idx in range(pos+1, 4+1):
            opt = model.get_random_rec(total)
            while opt['vehicle_brand_id'] == rec['vehicle_brand_id']:
                opt = model.get_random_rec(total)
            options.append(opt)
        question['options'] = options
        return question
