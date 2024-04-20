import os

os.environ["TQDM_INTERVAL"] = '1'
import cv2

from typing import List, Dict, Tuple
import copy

from transformers import YolosImageProcessor, OneFormerProcessor, GitProcessor, BlipProcessor, Blip2Processor, \
    CLIPSegProcessor
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
import albumentations
import skimage
from skimage import io, draw
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, remove_small_holes, remove_small_objects
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image as PILImage
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, Image, load_dataset, concatenate_datasets
from transformers import ConvNextImageProcessor
from Utils.Dataset_utils import MaskTransform


# 图片分类的collator
class ImageClassificationCollator:
    def __init__(self, processor: ConvNextImageProcessor, return_tensor='pt', need_labels=True, **kwargs):
        self.processor = processor
        self.need_labels = need_labels
        self.return_tensor = return_tensor
        self.kwargs = kwargs

    def __call__(self, batch):
        # 用GitProcessor处理数据
        batch_new = self.processor(images=[item['image'] for item in batch],
                                   return_tensors=self.return_tensor,
                                   **self.kwargs)
        # 将batch的labels整合为一个tensor
        if self.need_labels:
            batch_new['labels'] = torch.tensor(np.array([item['labels'] for item in batch])).float()

        return batch_new


# 语义分割的collator
class SemanticSegmentationCollator:
    def __init__(self, processor: ConvNextImageProcessor, return_tensor='pt', need_labels=True,
                 return_target_sizes=False, **kwargs):
        self.processor = processor
        self.need_labels = need_labels
        self.return_tensor = return_tensor
        self.return_target_sizes = return_target_sizes  # 是否返回原始图像大小，用于分割结果后处理
        self.kwargs = kwargs

    def __call__(self, batch):
        batch_new = self.processor([item['image'] for item in batch],
                                   task_inputs=["panoptic"] * len(batch),
                                   segmentation_maps=[item['segmentation_maps'].astype(np.uint8) for item in batch],
                                   return_tensors="pt",
                                   ignore_index=None)

        # 返回原始图像大小，每个'target_size'是元组，包含了图片的宽和高、通道数
        if self.return_target_sizes:
            batch_new['target_sizes'] = torch.tensor([item['target_size'][:2] for item in batch])

        return batch_new

# 文本驱动的语义分割的collator
class TextDrivenSemanticSegmentationCollator:
    def __init__(self, processor: CLIPSegProcessor, label2id,return_tensor='pt', need_labels=True,
                 return_target_sizes=False, **kwargs):
        self.processor = processor
        self.label2id = label2id
        self.need_labels = need_labels
        self.return_tensor = return_tensor
        self.return_target_sizes = return_target_sizes  # 是否返回原始图像大小，用于分割结果后处理
        self.kwargs = kwargs

    def __call__(self, batch):
        #将每个segmentation_maps转换为多个二元mask以及对应的文本
        for item in batch:
            item['binary_maps'],item['text'] = MaskTransform.multi_mask_to_binary_masks(item['segmentation_maps'],label2id=self.label2id)
            #image需要相对应复制n份
            item['image'] = [item['image']] * len(item['text'])

        # 如果0代表背景且id2label里没有背景，那么do_reduce_labels=True
        batch_new = self.processor(text=[text for item in batch for text in item['text']],
                              images=[image for item in batch for image in item['image']],
                              padding=True,
                              return_tensors="pt")
        #将每个binary_mask放缩到[352,352]大小
        binary_masks=[cv2.resize(binary_mask, tuple(self.processor.image_processor.size.values())) for item in batch for binary_mask in item['binary_maps']]
        batch_new['labels']=torch.tensor(np.array(binary_masks)).float()

        # 返回原始图像大小，每个'target_size'是元组，包含了图片的宽和高、通道数
        if self.return_target_sizes:
            batch_new['target_sizes'] = torch.tensor([item['target_size'][1:] for item in batch])
        return batch_new


# 图片转文字的collator
class ImageToTextCollator:
    def __init__(self, processor: Union[BlipProcessor, Blip2Processor, GitProcessor],
                 return_tensor='pt', padding=True, max_length=128, need_labels=True, add_eos_token=False,
                 **kwargs):
        self.processor = processor
        self.return_tensor = return_tensor
        self.padding = padding
        self.max_length = max_length
        self.need_labels = need_labels
        self.add_eos_token = add_eos_token
        self.kwargs = kwargs

    def __call__(self, batch):
        #如果不需要labels，就不需要输入text
        batch_new = self.processor(
            text=[item['caption'] +
                  (self.processor.tokenizer.eos_token if self.add_eos_token else '') for item in
                  batch] if self.need_labels else None,
            images=[item['image'] for item in batch],
            return_tensors=self.return_tensor,
            padding=self.padding,
            max_length=self.max_length,
            **self.kwargs)

        if self.need_labels:
            batch_new['labels'] = batch_new['input_ids'].clone()

        return batch_new

# 图片分类的collator
class ImageClassificationCollator:
    def __init__(self, processor: ConvNextImageProcessor, return_tensor='pt', need_labels=True, **kwargs):
        self.processor = processor
        self.need_labels = need_labels
        self.return_tensor = return_tensor
        self.kwargs = kwargs

    def __call__(self, batch):
        # 用GitProcessor处理数据
        batch_new = self.processor(images=[item['image'] for item in batch],
                                   return_tensors=self.return_tensor,
                                   **self.kwargs)
        # 将batch的labels整合为一个tensor
        if self.need_labels:
            batch_new['labels'] = torch.tensor(np.array([item['labels'] for item in batch])).float()

        return batch_new



# 定义groupvit的data_collator，只用于训练，text和image的数量必须一致
class GroupvitCollator:
    def __init__(self, processor: ConvNextImageProcessor, return_tensor='pt', **kwargs):
        self.processor = processor
        self.return_tensor = return_tensor
        self.kwargs = kwargs

    def __call__(self,batch: List[Dict]):
        # 用GitProcessor处理数据
        batch_new = self.processor(text=[item['caption'] for item in batch],
                                       images=[item['image'] for item in batch],
                                       padding=True,
                                       return_tensors='pt')

        # 设置return_loss=True，返回loss
        batch_new['return_loss'] = True

        return batch_new



