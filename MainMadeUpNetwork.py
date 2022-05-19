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
from UNetJavid import UNet

#Hyper-parameters
num_epochs = 20
batch_size = 4
leanrning_rate = 0.001


ChiariFoldLocation = '/Volumes/Kurtlab/Chiari Morphology/AutomaticSegmentationData/HighResolution/Chiari/'
HealthyFoldLocation = '/Volumes/Kurtlab/Chiari Morphology/AutomaticSegmentationData/HighResolution/Healthy/'


def LoadingData(FolderLocation, Filename, NumberOfFolders):
    Counter = 1
    for SubjNum in os.listdir(FolderLocation):
        print(SubjNum)
        if SubjNum == 'Subject1':
            ImageSize = nib.load(FolderLocation + SubjNum + Filename).shape
            ImageSizeFinal = np.append(ImageSize, NumberOfFolders)
            StackedData = np.zeros(ImageSizeFinal)
            StackedData[:, :, :, 0] = nib.load(FolderLocation + SubjNum + Filename).get_fdata()

        if SubjNum.startswith('Subject') and SubjNum != 'Subject1':
            print(FolderLocation + SubjNum)
            StackedData[:, :, :, Counter] = nib.load(FolderLocation + SubjNum + Filename).get_fdata()
            Counter = Counter + 1

    MidSlice = np.int(StackedData.shape[0] / 2)
    [z, h, w, nP] = np.shape(StackedData)
    StackedData2D = np.zeros([1, h, w, nP])
    StackedData2D[0, :, :, :] = np.squeeze(StackedData[MidSlice, :, :, :])
    StackedData2D = StackedData2D.astype('float32')
    StackedData2D = StackedData2D / np.max(StackedData2D)
    return torch.from_numpy(StackedData2D)


NumberOfFolders = 12
FilenameT1 = '/T1.nii'
Features = LoadingData(ChiariFoldLocation, FilenameT1, NumberOfFolders)

NumberOfFolders = 12
FilenameTonsilMask = '/CerebralTonsilMask.nii'
Labels = LoadingData(ChiariFoldLocation, FilenameTonsilMask, NumberOfFolders)
n_sampels = np.int(Labels.shape[2])


class Morphological_Data(Dataset):
    def __init__(self, featuresT1, masks):
        self.x = (featuresT1)
        self.y = (masks)
        self.n_samples = featuresT1.shape[0]

    def __getitem__(self, index):
        return self.x[index, :, :], self.y[index, :, :]

    def __len__(self):
        return self.n_samples



X_Training, X_Testing, Y_Training, Y_Testing = train_test_split(torch.permute(Features, (3, 0, 1, 2)), torch.permute(Labels, (3, 0, 1, 2)), test_size=0.2, random_state=0)

dataset_training = Morphological_Data(X_Training, Y_Training)
dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, shuffle=True)

dataset_testing = Morphological_Data(X_Testing, Y_Testing)
dataloader_testing = DataLoader(dataset=dataset_testing, batch_size=batch_size, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   #1 is the input size (1 color channel), 6 is the number of output channels, 5 is the kernel size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))   #6 is the input size, 16 is the output size, 5 is the kernell size
        self.conv3 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))

        self.conv6 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.up_sample = nn.Upsample(scale_factor=2)
        self.conv7 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(6, 6, (3, 3), (1, 1), (1, 1))
        self.conv11 = nn.Conv2d(6, 1, (3, 3), (1, 1), (1, 1))
        self.up_sample2 = nn.Upsample(size=(256, 256))


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.up_sample(F.relu(self.conv7(x)))
        x = self.up_sample(F.relu(self.conv8(x)))
        x = self.up_sample(F.relu(self.conv9(x)))
        x = self.up_sample2(F.relu(self.conv10(x)))
        x = self.conv11(x)  #No sigmoid function bc in the BCEWithLogitsLoss it is already applied
        return x

# model = ConvNet()
model = UNet()
# loss_fn = nn.BCELoss()
# loss_fn = nn.L1Loss()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=leanrning_rate)

n_total_steps = len(dataloader_training)
SavingLoss = [0]
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader_training):

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 2000 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        SavingLoss.append(loss.item())

plt.plot(SavingLoss[1:])
x = 2


for i, (images, labels) in enumerate(dataloader_testing):

    for j in range(3):
        plt.figure()
        plt.imshow(images[j, 0, :, :], cmap='gray')
        plt.imshow(labels[j, 0, :, :], cmap='gray', alpha=0.7)

        testing_output_lable = model(images)
        testing_output_lable = testing_output_lable.detach().numpy()

        plt.figure()
        plt.imshow(images[j, 0, :, :], cmap='gray')
        plt.imshow(testing_output_lable[j, 0, :, :], cmap='gray', alpha=0.7)


### Testing if dataset is working correctly!Å
# dataset = Morphological_Data(Features, Labels)
# FirstData = dataset[0]
# Features, Labels = FirstData
# plt.figure()
# plt.imshow(Features[:, :], cmap='gray')
# plt.imshow(Labels[:, :], cmap='gray', alpha=0.7)

## Testing if dataset is working correctly!
# batchsize = 4
# dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)
# dataiter = iter(dataloader)
# data = dataiter.next()
# Features, Labels = data
# plt.figure()
# plt.imshow(Features[0, :, :], cmap='gray')
# plt.imshow(Labels[0, :, :], cmap='gray', alpha=0.7)

# batchsize = 4
# dataloader = DataLoader(dataset=dataset[0:NumberofDataToTrain], batch_size=batchsize, shuffle=True)


###Testing the developed layers
dataloader_trainingItered = iter(dataloader_training)
featuresss, labelssss = dataloader_trainingItered.next()

conv1_padding = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3,3), stride=(1,1), padding=(1,1))
xpadded = conv1_padding(featuresss)
print(xpadded.shape)
#
# conv1 = nn.Conv2d(1, 6, 5)
# pool = nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(6, 16, 5)
# conv3 = nn.Conv2d(16, 1, 5)
#
# print(featuresss.shape)
# x1 = conv1(featuresss)
# print(x1.shape)
# x2 = pool(x1)
# print(x2.shape)
# x3 = conv2(x2)
# print(x3.shape)
# x4 = pool(x3)
# print(x4.shape)
#
# up_sample = nn.Upsample(size=(256, 256))
# x5 = up_sample(x4)
# print(x5.shape)


# def forwardd(x):
#     conv1 = nn.Conv2d(1, 6, 5)  # 1 is the input size (1 color channel), 6 is the number of output channels, 5 is the kernel size
#     pool = nn.MaxPool2d(2, 2)
#     conv2 = nn.Conv2d(6, 6, 5)  # 6 is the input size, 16 is the output size, 5 is the kernell size
#     conv3 = nn.Conv2d(6, 6, 5)
#     conv4 = nn.Conv2d(6, 6, 5)
#
#     up_sample = nn.Upsample(scale_factor=2)
#     conv7 = nn.Conv2d(6, 6, 5)
#     conv8 = nn.Conv2d(6, 6, 5)
#     conv9 = nn.Conv2d(6, 6, 5)
#     conv10 = nn.Conv2d(6, 6, 5)
#     up_sample2 = nn.Upsample(size=(256, 256))
#
#     conv11 = nn.Conv2d(6, 1, 5)
#
#     x = pool(F.relu(conv1(x)))
#     x = pool(F.relu(conv2(x)))
#     x = pool(F.relu(conv3(x)))
#     x = pool(F.relu(conv4(x)))
#
#     x = up_sample(F.relu(conv7(x)))
#     x = up_sample(F.relu(conv8(x)))
#     x = up_sample(F.relu(conv9(x)))
#     x = up_sample2(F.relu(conv10(x)))
#
#     x = conv11(x)  # It is already included in the loss that's being used below
#
#     return x
#
#
# x = pool(F.relu(conv1(images)))
# x = pool(F.relu(conv2(x)))
# x = pool(F.relu(conv3(x)))
# x = pool(F.relu(conv4(x)))
#
# x = up_sample(F.relu(conv7(x)))
# x = up_sample(F.relu(conv8(x)))
# x = up_sample(F.relu(conv9(x)))
# x = up_sample2(F.relu(conv10(x)))
#
# x = conv11(x)  # It is already included in the loss that's being used below
