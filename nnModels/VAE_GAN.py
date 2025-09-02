import torch.nn as nn
import torch.nn.functional as F
import torch
from CVAE2D_ORIGINAL import CVAE2D_ORIGINAL

import nnModels.nnModels_utils as nnModels_utils
from torch.autograd import Variable


class Discriminator():
    
    def __init__(self):
        super(Discriminator, self).__init__()
        nn.Module.__init__(self)

        ## TODO: UNFINISHED, this is a copy paste from Sauty's github
        #  This model is used for the 3D MRI. Need to change layer/size for starmen dataset ! 
        
        # Discriminator
        self.d_conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)               # 32 x 40 x 48 x 40
        self.d_conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)              # 64 x 20 x 24 x 20
        self.d_conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)             # 128 x 10 x 12 x 10
        self.d_conv4 = nn.Conv3d(128, 256, 3, stride=2, padding=1)            # 256 x 5 x 6 x 5
        self.d_conv5 = nn.Conv3d(256, 1, 3, stride=1, padding=1)              # 1 x 5 x 6 x 5
        self.d_bn1 = nn.BatchNorm3d(32)
        self.d_bn2 = nn.BatchNorm3d(64)
        self.d_bn3 = nn.BatchNorm3d(128)
        self.d_bn4 = nn.BatchNorm3d(256)
        self.d_bn5 = nn.BatchNorm3d(1)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.relu3 = nn.LeakyReLU(0.02, inplace=True)
        self.relu4 = nn.LeakyReLU(0.02, inplace=True)
        self.relu5 = nn.LeakyReLU(0.02, inplace=True)
        #self.d_fc1 = nn.Linear(38400, 500)
        self.d_fc = nn.Linear(150, 1)
        
    def forward(self, image):
        image = image #+ torch.normal(torch.zeros(image.shape), 0.1, generator=None, out=None).to(device).detach()
        d1 = self.relu1(self.d_conv1(image))
        #d1_n = d1 + torch.normal(torch.zeros(d1.shape), 0.1, generator=None, out=None).to(device).detach()
        d2 = self.relu2(self.d_conv2(d1))
        #d2_n = d2 + torch.normal(torch.zeros(d2.shape), 0.1, generator=None, out=None).to(device).detach()
        d3 = self.relu3(self.d_conv3(d2))
        #d3_n = d3 + torch.normal(torch.zeros(d3.shape), 0.1, generator=None, out=None).to(device).detach()
        d4 = self.relu4(self.d_conv4(d3))
        #d4_n = d4 + torch.normal(torch.zeros(d4.shape), 0.1, generator=None, out=None).to(device).detach()
        d5 = self.relu5(self.d_conv5(d4))
        d6 = torch.sigmoid(self.d_fc(d5.flatten(start_dim=1)))
        return d6


class VAE_GAN(nn.Module):
    
    def __init__(self):
        super(VAE_GAN, self).__init__()
        nn.Module.__init__(self)
        
        self.VAE = CVAE2D_ORIGINAL()
        self.discriminator = Discriminator()

    def train(self):
        # TODO: Train code
        return
        
    def forward(self, x):
        # TODO: forward, x is image input

        return 

    

