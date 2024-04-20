import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from Utils.Cv_Preprocess import ClipSeg_utils
from Utils.used_class import get_cache_dir
from Utils.Dataset_utils import CategoryId2Caption, MaskTransform
from torchvision.transforms import Resize
from typing import List, Tuple
from PIL import Image as PILImage
import torch
from transformers import SegformerForSemanticSegmentation,MobileNetV2ForSemanticSegmentation
from 传统检测 import TraditionalDetection

cache_dir = get_cache_dir()


# 对一张图片做分类
class ClassificationProcess:
    def __init__(self, model, processor, problem_type='multilabel_classification'):
        self.model = model
        self.processor = processor
        self.problem_type = problem_type

    def predict_one_image(self, image: PILImage.Image,return_logits=False) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(**inputs)
        logits = outputs.logits
        # logits的第一维如果是1，就去掉
        if logits.shape[0] == 1:
            logits = logits[0].cpu()
        # 获取分类结果
        if self.problem_type == 'multilabel_classification':
            # 多标签二分类
            preds = logits.float().sigmoid().round().long().cpu().numpy()
        elif self.problem_type == 'multiclass_classification':
            # 单标签多分类
            preds = logits.argmax(-1).cpu().numpy()
        else:
            raise ValueError('problem_type must be multilabel_classification or multiclass_classification')
        if return_logits:
            return preds,logits
        else:
            return preds

    # 将多标签二分类结果转化为list，list中出现的元素代表为1的标签
    def preds2list(self, preds) -> Tuple[List[int], List[str]]:
        label_id_list = []
        label_name_list = []
        for i in range(len(preds)):
            if preds[i] == 1:
                label_id_list.append(i + 1)
                label_name_list.append(self.model.config.id2label[i])
        return label_id_list, label_name_list


# 对一张图片做分割
class SegmentationProcess(MaskTransform):
    def __init__(self, model:SegformerForSemanticSegmentation, processor, min_area_threshold=0.01):
        self.model = model
        self.processor = processor
        self.min_area_threshold = min_area_threshold

    def predict_one_image(self, image: PILImage.Image,before_logits:torch.tensor=None,logits_alpha=1) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(**inputs)
        if before_logits is not None:
            before_logits=np.array(before_logits.float())
            logits = outputs.logits.float().cpu().numpy()
            logits=self.change_logits(logits,before_logits,logits_alpha=logits_alpha)
            outputs.logits=torch.tensor(logits)
        # 获取分割结果
        # img.size为（宽，高），而此处需要的是（高，宽）
        mask = \
        self.processor.post_process_semantic_segmentation(outputs, target_sizes=[[image.size[1], image.size[0]]])[0]
        # 膨胀，删除小连通域
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = self.mask_postprocess_multi(mask, min_area_threshold=self.min_area_threshold)
        return mask

    #修改语义分割的logits
    def change_logits(self,logits:np.ndarray,before_logits:np.ndarray,logits_threshold=0.5,logits_alpha=1):
        # 取before_logits的前len(lable2id)-1个值
        before_logits = before_logits[:len(self.model.config.label2id) - 1]
        # 如果before_logits[i]取sigmoid后大于阈值，则设为0
        for i in range(before_logits.shape[0]):
            if 1/(1+np.exp(-before_logits[i]))>=logits_threshold:
                before_logits[i]=0

        #logits大小为(batch_size, num_labels, height/4, width/4)
        #logits中代表第n类的logits，需要与before_logits中的第n-1个值相加。第0类需要加上before_logits的平均值
        for i in range(logits.shape[1]):
            if i==0:
                logits[:,i,:,:]+=0 #np.mean(before_logits,axis=0)
            else:
                logits[:,i,:,:]+=before_logits[i-1] * logits_alpha

        return logits


    # 只保留mask中的某些类别
    def filter_mask(self, mask: np.ndarray, keep_classes: list) -> np.ndarray:
        # 若mask中某个值不在keep_classes中，则将其置为0。mask为10的值不会被置为0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] not in keep_classes and mask[i, j] != 10:
                    mask[i, j] = 0
        return mask


# 对一张图片先做分类，再根据分类结果做分割
class ClassificationAndSegmentationProcess:
    def __init__(self, classification_model, segmentation_model, classification_processor, segmentation_processor,
                 problem_type='multilabel_classification', min_area_threshold=0.001,label2id=None,use_logits_process=False,logits_alpha=1):
        self.classification_process = ClassificationProcess(classification_model, classification_processor,
                                                            problem_type)
        self.segmentation_process = SegmentationProcess(segmentation_model, segmentation_processor,
                                                        min_area_threshold=min_area_threshold)
        self.traditional_detection = TraditionalDetection(missing_pixel_label_id=label2id['像素缺失'],
                                                          stripe_noise_label_id=label2id['条状噪声'])
        self.label2id = label2id
        self.use_logits_process=use_logits_process
        self.logits_alpha=logits_alpha

    def predict_one_image(self, image: PILImage.Image, tradition_detect=True) -> np.ndarray:
        if not self.use_logits_process:
            # 先做分类
            preds = self.classification_process.predict_one_image(image)
            preds_ids, labelnames = self.classification_process.preds2list(preds)
            # 若preds_ids为空，则返回全0的mask
            if len(preds_ids) == 0:
                return np.zeros((image.size[1], image.size[0]), dtype=np.uint8)  # img.size为（宽，高）
            # 再根据分类结果做分割
            mask = self.segmentation_process.predict_one_image(image)

        else:
            #做分类，返回logits是一列向量
            preds,logits = self.classification_process.predict_one_image(image,return_logits=True)
            preds_ids, labelnames = self.classification_process.preds2list(preds)
            # 再根据分类结果做分割
            mask = self.segmentation_process.predict_one_image(image,before_logits=logits,logits_alpha=self.logits_alpha)

        if tradition_detect:
            # 做传统检测
            mask_tra = self.traditional_detection.predict_one_image(image)
            # 将传统检测的结果加入mask中，取最大值
            mask = np.maximum(mask, mask_tra)

        #如果labelnames中有”拼接错误“
        if "拼接错误" in labelnames:
            #将mask中为0的部分置为“拼接错误”对应的id
            mask[mask==0] = self.label2id["拼接错误"]
        elif "拼接痕迹" in labelnames:
            mask[mask==0] = self.label2id["拼接痕迹"]

        if not self.use_logits_process:
            # 只保留mask中的某些类别,但“像素缺失”一定保留
            mask = self.segmentation_process.filter_mask(mask, preds_ids+[self.label2id["像素缺失"]])
        else:
            # 只保留mask中的某些类别,但“像素缺失”一定保留，且pred_ids中需要增加0到len(label2id)
            preds_ids=preds_ids+[self.label2id["像素缺失"]]+list(range(len(self.label2id)))
            mask = self.segmentation_process.filter_mask(mask, preds_ids)
        return mask


# 对一张图片做图像转文字
class Image2TextProcess(CategoryId2Caption):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict_one_image(self, image: PILImage.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                generated_ids = self.model.generate(**inputs, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption

    # 将多标签二分类结果转化为list，list中出现的元素代表为1的标签
    def caption2list(self, caption, label2id) -> Tuple[List[int], List[str]]:
        label_id_list, label_name_list = self.caption_to_category_id(caption, label2id, return_label_names=True)
        return label_id_list, label_name_list


# 对一张图片做基于文本的分割
class TextDrivenSemanticSegmentationProcess(ClipSeg_utils):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict_one_image(self, image: PILImage.Image, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to(
            self.model.device)
        with torch.no_grad():
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(**inputs)
        logits = outputs.logits

        # 获取分类结果
        mask = self.logits_to_mask(logits, texts, label2id=self.model.config.label2id)

        # 将mask的大小Resize为原图的大小
        torch_resize = Resize(image.size[0:2], interpolation="nearest")
        mask = torch_resize(mask)
        mask = mask.numpy()
        return mask

    # 只保留mask中的某些类别
    def filter_mask(self, mask: np.ndarray, keep_classes: list) -> np.ndarray:
        # 若mask中某个值不在keep_classes中，则将其置为0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] not in keep_classes:
                    mask[i, j] = 0
        return mask
