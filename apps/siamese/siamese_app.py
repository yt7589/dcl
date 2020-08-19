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
#
from apps.siamese.app_config import AppConfig
from apps.siamese.atnt_face_ds import AtntFaceDs

class SiameseApp(object):
    def __init__(self):
        self.name = 'models.SiameseApp'

    def startup(self, args):
        print('Siamese Network App v0.0.2')
        folder_dataset = dset.ImageFolder(root=AppConfig.training_dir)
        siamese_dataset = AtntFaceDs(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
        vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
        dataiter = iter(vis_dataloader)


        example_batch = next(dataiter)
        print('example_batch: {0};'.format(example_batch.shape))
        concatenated = torch.cat((example_batch[0],example_batch[1]),0)
        self.imshow(torchvision.utils.make_grid(concatenated))
        print(example_batch[2].numpy())




















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