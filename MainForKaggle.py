import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
from torch.utils.data import Dataset, DataLoader
import math
from UNetJavid import UNet
from DataLoaderForKaggle import KaggleData
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
# from ModelFromYoutube import UNET
from torchvision import models #Added by Javid for checking the model summary
from torchsummary import summary #Added by Javid for checking the model summary

#Hyper parameters
num_epochs = 3
learning_rate = 1e-4
batch_size = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 1
Load = False


#Model Save Location
ModelLocation = '/home/user/Documents/Segmentation_Model_Javid/ModelV1Worked.pth'

#Data Location
TrainDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/train'
TrainMaskDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/train_masks'

TestDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/test'
TestMaskDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/test_masks'


# IMAGE_HEIGHT = 240 #Commented since right now, somewhere else in the code the images are scaled down!
# IMAGE_WIDTH = 160 #Commented since right now, somewhere else in the code the images are scaled down!
train_transform = A.Compose(
    [
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            # max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

dataset_training = KaggleData(TrainDataLoc, TrainMaskDataLoc, scale_factor=8, transform=train_transform)
dataloader_training = DataLoader(dataset=dataset_training, num_workers=num_workers, batch_size=batch_size, shuffle=True)


if 'model' not in globals():
    model = UNet(InputChannels=3, OutputChannels=1).to(DEVICE)
if 'model' in globals():
    model = model.to(DEVICE)
summary(model, (3, 160, 240)) #Added by Javid to check the model summary

loss_Func = nn.BCEWithLogitsLoss()
Optimization = torch.optim.Adam(model.parameters(), lr=learning_rate)
SavingLoss = [0]

if Load:
    Loaded_Checkpoint = torch.load(ModelLocation)
    model.load_state_dict(Loaded_Checkpoint['State_dict'])
    Optimization.load_state_dict(Loaded_Checkpoint['Optimizer'])

n_total_steps = len(dataloader_training)
for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(dataloader_training):

        images = images.to(device=DEVICE)
        masks = masks.to(device=DEVICE)
        PredictedMask = model(images)
        masksFinal = masks[:, 0, :, :].float().unsqueeze(1)
        loss = loss_Func(PredictedMask, masksFinal)

        # Backward and optimize
        Optimization.zero_grad()
        loss.backward()
        Optimization.step()

        # if (i+1) % 2000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        SavingLoss.append(loss.item())


checkpoint = {'epoch': epoch, 'State_dict': model.state_dict(), 'Optimizer': Optimization.state_dict()}
torch.save(checkpoint, ModelLocation)

iteration = iter(dataloader_training)
images, labels = next(iteration)
x = 2

for images, labels in iter(dataloader_training):
    for j in range(1):
        plt.figure()
        plt.imshow(images[j, 0, :, :], cmap='gray')
        plt.imshow(labels[j, 0, :, :], cmap='gray', alpha=0.7)

        testing_output_lable = model(images.to(device=DEVICE))
        testing_output_lable = testing_output_lable.cpu().detach().numpy()

        plt.figure()
        plt.imshow(images[j, 0, :, :], cmap='gray')
        plt.imshow(testing_output_lable[j, 0, :, :], cmap='gray', alpha=0.7)


# images, masks = dataset_training[0]

# dataset_testing = DataLoader(TestDataLoc, TestMaskDataLoc)
# dataloader_training = DataLoader(dataset=dataset_testing, batch_size=batch_size, shuffle=True)

# if __name__ == '__main__':
#     x = 2
