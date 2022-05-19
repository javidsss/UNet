import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
import numpy as np
import os
import nibabel as nib

DirectoryDir = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/LowResolution/Chiari/'
SaveFolder = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/Combined/Chiari/'

AllTheFolders = os.listdir(DirectoryDir)
AllTheFolders_NoExtra = []
for FolderNames in AllTheFolders:
    if FolderNames.startswith('Subj') and not FolderNames.endswith('trial'):
        AllTheFolders_NoExtra.append(FolderNames)

for FolderNames in AllTheFolders_NoExtra:
    print(FolderNames)
    Brain = np.array(nib.load(os.path.join(DirectoryDir, FolderNames, 'T1_LPI.nii')).get_fdata())
    BrainTranspose = (np.transpose(Brain, [0, 2, 1]))
    BrainTransposeLR = np.flip(BrainTranspose, axis=2)
    BrainNifti = nib.Nifti1Image(BrainTransposeLR, affine=np.eye(4))

    AllCerebellum = np.array(nib.load(os.path.join(DirectoryDir, FolderNames, 'TonsilEdit.nii')).get_fdata())
    Onlytonsil1 = AllCerebellum == 25
    Onlytonsil2 = AllCerebellum == 23
    TonsilMask = (Onlytonsil1 + Onlytonsil2).astype(int)
    TonsilMaskTranspose = (np.transpose(TonsilMask, [0, 2, 1]))
    TonsilMaskTransposeLR = np.flip(TonsilMaskTranspose, axis=2)
    TonsilMaskNifti = nib.Nifti1Image(TonsilMaskTransposeLR, affine=np.eye(4))

    FinalFolder = os.path.join(SaveFolder, FolderNames)
    os.makedirs(FinalFolder, exist_ok=True)

    nib.save(BrainNifti, os.path.join(FinalFolder, 'T1.nii'))
    nib.save(TonsilMaskNifti, os.path.join(FinalFolder, 'CerebralTonsilMask.nii'))


# Brain = np.array(nib.load('/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/LowResolution/Chiari/Subj51/T1_LPI.nii').get_fdata())
# BrainTranspose = (np.transpose(Brain, [0, 2, 1]))
# BrainTransposeLR = np.flip(BrainTranspose, axis=2)
# Mask = np.array(nib.load('/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/AutomaticSegmentationData/LowResolution/Chiari/Subj51/TonsilEdit.nii').get_fdata())
# Mask1 = Mask == 25
# Mask2 = Mask == 23
# MaskFinal = Mask1+Mask2
# MaskFinalTranspose = (np.transpose(MaskFinal, [0, 2, 1]))
# MaskFinalTransposeLR = np.flip(MaskFinalTranspose, axis=2)
# plt.imshow(BrainTransposeLR[128, :, :], cmap='gray')
# plt.imshow(MaskFinalTransposeLR[128, :, :], cmap='gray', alpha=0.7)