import albumentations as A
from PIL import Image
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
import numpy as np


Location = '/run/user/1000/gvfs/smb-share:server=mesyno01.me.stevens-tech.edu,share=kurtlab/Chiari Morphology/LearningUNetSampleData/train/0cdf5b5d0ce1_01.jpg'

ImageFinal = Image.open(Location)
ImageFinal2 = cv2.imread(Location)
# ImageFinal2 = ImageFinal2[:,:,0]
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 180
train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        # ToTensorV2(),
    ],
)

Image_list = [ImageFinal2]
ImageFinal = np.array(ImageFinal2)

# plt.subplot(1, 11)
for i in range(10):
    transformations = train_transform(image=ImageFinal, mask=ImageFinal)
    transformedImages = transformations["image"]
    plt.subplot(1, 10, i+1)
    plt.imshow(transformedImages)

