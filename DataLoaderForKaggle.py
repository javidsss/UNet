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
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import nibabel as nib

class Data_Loader_Javid(Dataset):
    def __init__(self, train_dir, train_mask_dir, transform=None):
        self.train_dir = train_dir
        self.train_mask_dir = train_mask_dir
        # self.transform = transform

        self.train_data = os.listdir(train_dir) #The names of all files in the folder. It can later be combined with the folder directory
        self.train_mask_data = os.listdir(train_mask_dir)
        # self.n_samples = train_dir.shape[0]

    def __getitem__(self, index):
        train_finalpath = os.path.join(self.train_dir, self.train_data[index])
        train_mask_finalpath = os.path.join(self.train_mask_dir, self.train_mask_data[index].replace(".jpg", "gif"))

        train = np.array(Image.open(train_finalpath).convert("RGB"))
        train_mask = np.array(Image.open(train_mask_finalpath).convert("RGB"))
        train_mask[train_mask == 255.0] = 1.0

        # if self.transform is not None:
        #     augmentations = self.transform(image=train, mask=train_mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return np.transpose(train, [2, 0, 1]).astype('float32'), np.transpose(train_mask, [2, 0, 1]).astype('float32')

    def __len__(self):
        return len(self.train_data)


## Testig whether the code works!!
# TrainDataLoc = '/Volumes/Kurtlab/Chiari Morphology/LearningUNetSampleData/train'
# TrainMaskDataLoc = '/Volumes/Kurtlab/Chiari Morphology/LearningUNetSampleData/train_masks'
# batch_size = 4
#
# dataset_training = Data_Loader_Javid(TrainDataLoc, TrainMaskDataLoc)
# dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, shuffle=True)
# images, masks = dataset_training[0]



class KaggleData(Dataset):
    def __init__(self, train_loc, train_mask_loc, scale_factor=None, transform=None):
        super(KaggleData, self).__init__()

        self.train_loc = train_loc
        self.train_mask_loc = train_mask_loc
        self.train_foldernames = os.listdir(self.train_loc)
        self.train_mask_foldernames = os.listdir(self.train_mask_loc)
        self.scale_factor = scale_factor
        self.transform = transform

    def __getitem__(self, index):
        self.train_loc_final = os.path.join(self.train_loc, self.train_foldernames[index])
        train_data_numpy = np.array(Image.open(self.train_loc_final).convert("RGB"))
        [h, w, NChannels] = train_data_numpy.shape
        train_data_numpy_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor)), NChannels], dtype='float32')
        if self.scale_factor is not None:
            for i in range(NChannels):
                train_data_numpy_final[:, :, i] = cv2.resize(train_data_numpy[:, :, i], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
        # train_data = torch.from_numpy(train_data_numpy_final)
        train_data = train_data_numpy_final
        self.train_data_final = train_data / train_data.max()

        self.train_mask_loc_final = os.path.join(self.train_mask_loc, self.train_mask_foldernames[index])
        train_mask_numpy = np.array(Image.open(self.train_mask_loc_final).convert("RGB"), dtype='float32')
        train_mask_numpy_final = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor)), NChannels], dtype='float32')
        if self.scale_factor is not None:
            for i in range(NChannels):
                train_mask_numpy_final[:, :, i] = cv2.resize(train_mask_numpy[:, :, i], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor))) #open cvs size input automatically transposes the sizess!!!
        # train_mask = torch.from_numpy(train_mask_numpy_final)
        train_mask = train_mask_numpy_final
        self.train_mask_final = train_mask / train_mask.max()

        if self.transform is not None:
            augmentations = self.transform(image=self.train_data_final, mask=self.train_mask_final) #image and mask are dict names, I can use whatever name I want. Then I have to call them as I named them!
            image = augmentations["image"] #For some reason, image is permuted extra!!
            mask = augmentations["mask"]
        return image, torch.permute(mask, [2, 0, 1]) #For some reason in the albumentation step the image is permuted on its own!

    def __len__(self):
        return len(self.train_foldernames)

# TrainDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/train'
# TrainMaskDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/train_masks'
# batch_size = 4
#
# DataLoadPractice = KaggleData(TrainDataLoc, TrainMaskDataLoc)
# IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)
#
# for i, (image, mask) in enumerate(IterationOfTheData):
#     for i in range(image.shape[0]):
#         plt.figure()
#         plt.imshow(image[i,:,:,:])
#         plt.figure()
#         plt.imshow(mask[i,:,:,:])
#         plt.close('all')
#
# if __name__ == '__main__':
#     x = 2

