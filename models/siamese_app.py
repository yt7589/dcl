#
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SiameseApp(object):
    def __init__(self):
        self.name = 'models.SiameseApp'

    def startup(self, args):
        print('Siamese Network App v0.0.1')

    def imshow(self, img,text=None,should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()    

    def show_plot(self, iteration,loss):
        plt.plot(iteration,loss)
        plt.show()