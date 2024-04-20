import os

os.environ["TQDM_INTERVAL"] = '1'

import albumentations

from PIL import Image as PILImage
from Utils.Dataset_utils import CategoryId2Caption
import numpy as np


#语义分割的数据集transform
def segment_dataset_transform( examples: dict, aug=False,num_labels=None):
    # 创建图像数据增强的函数
    transforms = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        #albumentations.RandomCrop(332,332),
        albumentations.GaussNoise(p=0, var_limit=(10.0, 20.0)),
    ]) if aug else None

    images, masks = [], []
    for image, mask in zip(examples['image'], examples['mask']):
        if isinstance(image, PILImage.Image):
            image = np.array(image.convert("RGB"))  # 转换为np
        if isinstance(mask, PILImage.Image):
            mask = np.array(mask)

        # 进行数据增强
        if transforms is not None:
            out = transforms(image=image, mask=mask)
        else:
            out = {"image": image, "mask": mask}

        pixel_values = np.moveaxis(out["image"], -1, 0)  # 将channel放在第一维
        semantic_seg_transformed = out["mask"]

        #如果num_labels不为None，则将mask中的大于num_labels的值置为0
        if num_labels is not None:
            semantic_seg_transformed[semantic_seg_transformed >= num_labels] = 0


        images.append(pixel_values)
        masks.append(semantic_seg_transformed)

    target_size = examples['bbox_subfig_save_size'] if 'bbox_subfig_save_size' in examples \
        else [image.shape[1:] for image in images]

    return {'image': images, 'segmentation_maps': masks,'target_size': target_size}


#图像转文字的数据集transform
def image2text_dataset_transform(examples: dict, aug=False):
    # 创建图像数据增强的函数
    if aug:
        transforms = albumentations.Compose([
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.GaussNoise(p=0, var_limit=(10.0, 20.0)),
        ])
    else:
        transforms = None

    images = []
    for image, caption in zip(examples['image'], examples['caption']):
        image = np.array(image)
        # 进行数据增强
        if transforms is not None:
            out = transforms(image=image)
        else:
            out = {"image": image}

        images.append(PILImage.fromarray(out['image']))
        # captions.append(caption[0])

    return {'image': images, 'caption': examples['caption']}

#多标签二分类的数据集transform
def multi_label_dataset_transform(examples: dict,num_labels,aug=False,reduce_labels=True,label_column_name='labels'):
    # 创建图像数据增强的函数
    if aug:
        transforms = albumentations.Compose([
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.GaussNoise(p=0, var_limit=(10.0, 20.0)),
        ])
    else:
        transforms = None

    images = []
    for image in examples['image']:
        image = np.array(image)
        # 进行数据增强
        if transforms is not None:
            out = transforms(image=image)
        else:
            out = {"image": image}

        images.append(PILImage.fromarray(out['image']))
        # captions.append(caption[0])

    labels=[]
    #将多标签转换为多标签二分类，例如[0,3]转为[1,0,0,1],[1,2]转为[0,1,1,0]
    for label in examples[label_column_name]:
        label_new=CategoryId2Caption.label_to_binary(label,num_labels=num_labels,reduce_labels=reduce_labels)
        labels.append(label_new)

    return {'image': images, 'labels': labels}



