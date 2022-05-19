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
# import nibabel.freesurfer.mghformat as mgh

class CerebellumData(Dataset):
    def __init__(self, train_loc, train_mask_loc, scale_factor=None, transform=None):
        super(CerebellumData, self).__init__()

        self.train_loc = train_loc
        self.train_mask_loc = train_mask_loc
        self.train_foldernames = os.listdir(self.train_loc)
        self.train_foldernames_NoExtra = []
        for FolderNames in self.train_foldernames:
            if FolderNames.startswith('Subj'):
                self.train_foldernames_NoExtra.append(FolderNames)

        self.train_mask_foldernames = os.listdir(self.train_mask_loc)
        self.train_mask_foldernames_NoExtra = []
        for FolderNames in self.train_mask_foldernames:
            if FolderNames.startswith('Subj'):
                self.train_mask_foldernames_NoExtra.append(FolderNames)

        self.scale_factor = scale_factor
        self.transform = transform

    def __getitem__(self, index):
        self.train_loc_final = os.path.join(self.train_loc, self.train_foldernames_NoExtra[index], 'T1.nii')
        train_data_numpy3D = np.array(nib.load(self.train_loc_final).get_fdata())
        train_data_numpy = np.squeeze(train_data_numpy3D[math.ceil(train_data_numpy3D.shape[0]/2), :, : ])
        [h, w] = train_data_numpy.shape
        if self.scale_factor is not None:
            train_data = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
            train_data[:, :] = cv2.resize(train_data_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor)))  # open cvs size input automatically transposes the sizess!!!
        else:
            train_data = train_data_numpy
        self.train_data_final = train_data / train_data.max()


        self.train_mask_loc_final = os.path.join(self.train_mask_loc, self.train_mask_foldernames_NoExtra[index], 'CerebralTonsilMask.nii')
        train_mask_numpy3D = np.array(nib.load(self.train_mask_loc_final).get_fdata())
        train_mask_numpy = np.squeeze(train_mask_numpy3D[math.ceil(train_mask_numpy3D.shape[0]/2), :, :])
        if self.scale_factor is not None:
            train_mask = np.zeros([int(math.ceil(h / self.scale_factor)), int(math.ceil(w / self.scale_factor))], dtype='float32')
            train_mask[:, :] = cv2.resize(train_mask_numpy[:, :], dsize=(int(math.ceil(w / self.scale_factor)), int(h / self.scale_factor))) #open cvs size input automatically transposes the sizess!!!
        else:
            train_mask = train_mask_numpy
        self.train_mask_final = train_mask / train_mask.max()

        if self.transform is not None:
            augmentations = self.transform(image=self.train_data_final, mask=self.train_mask_final) #image and mask are dict names, I can use whatever name I want. Then I have to call them as I named them!
            image = np.squeeze(augmentations["image"]) #For some reason, image is permuted extra!!
            mask = np.squeeze(augmentations["mask"])
        else:
            image = self.train_data_final
            mask = self.train_mask_final
        return image, mask

    def __len__(self):
        return len(self.train_mask_foldernames_NoExtra)


# TrainDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/HighResolution/Chiari/'
# TrainMaskDataLoc = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/HighResolution/Chiari'
# batch_size = 4
#
# IMAGE_HEIGHT = 240
# IMAGE_WIDTH = 180
# train_transform = A.Compose(
#     [
#         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#         A.Rotate(limit=35, p=1.0),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.Normalize(
#             mean=[0.0],
#             std=[1.0],
#             max_pixel_value=255.0,
#         ),
#         # ToTensorV2(),
#     ],
# )

# DataLoadPractice = CerebellumData(TrainDataLoc, TrainMaskDataLoc, transform=train_transform)
# IterationOfTheData = DataLoader(DataLoadPractice, batch_size=batch_size, shuffle=False)
#
# for i, (image, mask) in enumerate(IterationOfTheData):
#     for i in range(image.shape[0]):
#         # plt.figure()
#         plt.imshow(image[i,:,:], cmap='gray')
#         plt.imshow(mask[i,:,:], cmap='gray', alpha=0.7)
#         plt.close('all')
