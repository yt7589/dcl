#

class AppConfig(object):
    '''
    # 人脸识别数据集
    img_w = 100
    img_h = 100
    img_channel = 1
    training_dir = "./datasets/siamese/faces/training/"
    testing_dir = "./datasets/siamese/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 5 # 100
    '''
    # 车辆识别数据集
    img_w = 224
    img_h = 224
    img_channel = 3
    training_dir = "./datasets/vehicles/training/"
    testing_dir = "./datasets/vehicles/testing/"
    train_batch_size = 8
    train_number_epochs = 10000 # 100