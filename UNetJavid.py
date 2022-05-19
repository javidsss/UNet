import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.model_selection import train_test_split

def DoubleConvJavid(in_ch, out_ch):
    conv_fun = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
        nn.ReLU(inplace=True)
    )
    return conv_fun

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels), #Commented by Javid to test whether the network works without this
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels), #Commented by Javid to test whether the network works without this
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



def CropFunc(input_image, target_image):

    h = int(input_image.shape[2])
    delta_h = int(input_image.shape[2] - target_image.shape[2])
    if delta_h < 0:
        delta_h = 0

    z = int(input_image.shape[3])
    delta_z = int(input_image.shape[3] - target_image.shape[3])
    if delta_z < 0:
        delta_z = 0

    output_image = input_image[:, :, int(delta_h/2):int(h-delta_h/2), int(delta_z/2):int(z-delta_z/2)]

    return output_image

class UNet(nn.Module):
    def __init__(self, inputchannels, outputchannels):
        super(UNet, self).__init__()
        self.inputchannels = inputchannels
        self.outputchannels = outputchannels
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = DoubleConv(self.inputchannels, 64) #Changed the number of input channels to 3, bc a specific data had RGB!
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)


        self.convtran1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.convtran2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.convtran3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.convtran4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, self.outputchannels, kernel_size=1, stride=1)


    def forward(self, input_image):

        x1 = self.conv1(input_image) #Skip
        x2 = self.maxpool(x1)

        x3 = self.conv2(x2)  #Skip
        x4 = self.maxpool(x3)

        x5 = self.conv3(x4)  #Skip
        x6 = self.maxpool(x5)

        x7 = self.conv4(x6)  #Skip
        x8 = self.maxpool(x7)

        x9 = self.conv5(x8)

        x = self.convtran1(x9)
        y7 = CropFunc(x7, x)
        x = self.conv6(torch.cat([x, y7], dim=1))

        x = self.convtran2(x)
        y5 = CropFunc(x5, x)
        x = self.conv7(torch.cat([x, y5], dim=1))

        x = self.convtran3(x)
        y3 = CropFunc(x3, x)
        x = self.conv8(torch.cat([x, y3], dim=1))

        x = self.convtran4(x)
        y1 = CropFunc(x1, x)
        x = self.conv9(torch.cat([x, y1], dim=1))

        F = nn.Upsample(size=(input_image.shape[2], input_image.shape[3])) #Added extra by Javid, will correct the minor size differences if downsampling is making it incorrect
        x = F(x) #Added extra by Javid
        x = self.out(x)

        return x





# if __name__ == '__main__':
#     model = UNet()
#     randInput = torch.rand((1, 1, 572, 572))
#     print(model(randInput).shape)


