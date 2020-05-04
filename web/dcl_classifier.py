#

class DclClassifier(object):
    def __init__(self):
        self.name = 'web.DclClassifier'

    def predict(self, img_name):
        print('预测图像：{0};'.format(img_name))
        return 10