import os

os.environ["TQDM_INTERVAL"] = '1'
import cv2

from typing import List, Dict, Tuple
import copy

from transformers import YolosImageProcessor, OneFormerProcessor, GitProcessor, BlipProcessor,Blip2Processor,CLIPSegProcessor,GroupViTModel
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

import numpy as np
import torch
from datasets import Dataset, Image, load_dataset, concatenate_datasets
from torch import nn


# 目标检测预处理类
class Obj_detect_prerpocess:
    # transforming a batch
    @classmethod
    def transform_aug_ann(cls, examples, aug=False):
        # 创建图像数据增强的函数
        transforms = albumentations.Compose([
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0),
            albumentations.GaussNoise(p=0.2),
            albumentations.RandomCropFromBorders(p=0),
        ], bbox_params=albumentations.BboxParams(format='coco', label_fields=['category']))

        # 对每个样本进行数据增强
        image_ids = examples['image_id']
        images, bboxes, area, categories = [], [], [], []
        for image, area_, bboxes_, category_id in zip(examples['image'], examples['area'], examples['bbox'],
                                                      examples['category_id']):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            if aug:
                out = transforms(image=image, bboxes=bboxes_, category=category_id)
            else:
                out = {"image": image, "bboxes": bboxes_, "category": category_id}

            area.append(area_)
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": cls.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]
        return {'image': images, 'image_annotations': targets}

    # 转换为coco标注
    @staticmethod
    def formatted_anns(image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations

    # 显示示例
    @staticmethod
    def show_example_object_detection(dataset: Dataset, max_index=10, show_time=30,
                                      pred_results: List[Dict] = None, id2label: Dict = None, no_label=False):
        # 打乱数据并选取n个
        dataset = dataset.shuffle()
        dataset = dataset.select(range(min(len(dataset), max_index)))
        # plt显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 如果no_label，则选取result的boxes不为空的n个样本
        if no_label:
            if pred_results is None:
                raise Exception('no_label=True,必须提供pred_results')
            # 选取result的boxes不为空的n个样本
            # dataset=dataset.select([i for i in range(len(dataset)) if len(pred_results[i]['boxes'])>0])

        # 显示每个样本
        for index in range(len(dataset)):
            if index > max_index:
                break
            # 转换为numpy，并转换为RGB
            one_sample = dataset[index]
            image_show = one_sample['image']
            # 转换为pil,转换为RGB
            if isinstance(image_show, np.ndarray):
                image_show = PILImage.fromarray(image_show[:, :, ::-1])

            plt.figure(index)
            plt.imshow(image_show)
            print(image_show.size)

            if not no_label:
                annotations = one_sample['image_annotations']['annotations']

                # 打上框和标签
                for i in range(len(annotations)):
                    bboxes_show = annotations[i]['bbox']
                    labels_show = str(annotations[i]['category_id']) if id2label is None else id2label[
                        annotations[i]['category_id']]
                    rect = plt.Rectangle((bboxes_show[0], bboxes_show[1]), bboxes_show[2],
                                         bboxes_show[3], linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    plt.text(bboxes_show[0], bboxes_show[1], labels_show, color='red', fontsize=8)

            # 如果pred_results不为空，就显示预测结果
            if pred_results is not None:
                result = pred_results[index]

                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    label_show_pred = str(label.item()) if id2label is None else id2label[label.item()]

                    print(f"Detected {label_show_pred} with confidence "
                          f"{round(score.item(), 3)} at location {box}")

                    # draw bounding box
                    plt.gca().add_patch(
                        plt.Rectangle(box[:2], *np.subtract(box[2:], box[:2]), fill=False, color='blue', linewidth=1))

                    # draw label
                    plt.gca().text(box[0], box[1] - 2, f'{label_show_pred}: {round(score.item(), 3)}',
                                   fontsize=8, color='white')

        plt.show(block=False)
        plt.pause(show_time)  # 显示10s
        plt.close()

    @staticmethod
    # 定义目标检测的data_collator
    def data_collator_detection(batch, processor: YolosImageProcessor, return_target_sizes=False):
        # 用YolosImageProcessor处理数据
        batch_new = processor.preprocess(images=[item['image'] for item in batch],
                                         annotations=[item['image_annotations'] for item in
                                                      batch] if 'image_annotations' in batch[0] else None,
                                         return_tensors="pt", padding=True, format='coco_detection')

        # 返回原始图像大小，每个'target_size'是元组，包含了图片的宽和高、通道数
        if return_target_sizes:
            batch_new['target_sizes'] = torch.tensor([item['target_size'][:2] for item in batch])
        return batch_new


# 全景分割预处理类
class Panorama_Segmentation_preprocess:
    # 数据增强并转换为image_processor的输入
    @classmethod
    def transform_aug_ann(cls, examples: dict, aug=False):
        # 创建图像数据增强的函数
        if aug:
            transforms = albumentations.Compose([
                albumentations.Resize(512, 512),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.GaussNoise(p=0.2, var_limit=(10.0, 20.0)),
            ])
        else:
            transforms = None

        images, masks, category_maps = [], [], []
        for image, points_, category_id in zip(examples['image'], examples['points'], examples['category_id']):
            if isinstance(image, PILImage.Image):
                image = np.array(image.convert("RGB"))  # 转换为np

            # points转换为mask
            instance_seg, inst2class = cls.points_to_mask2(points_, image.shape[:2], category_id)

            # 进行数据增强
            if transforms is not None:
                out = transforms(image=image, mask=instance_seg)
            else:
                out = {"image": image, "mask": instance_seg}

            pixel_values = np.moveaxis(out["image"], -1, 0)  # 将channel放在第一维
            instance_seg_transformed = out["mask"]

            images.append(pixel_values)
            masks.append(instance_seg_transformed)
            category_maps.append(inst2class)

        return {'image': images, 'segmentation_maps': masks,
                'instance_id_to_semantic_id': category_maps, 'target_size': examples['bbox_subfig_save_size']}

    # points转换为mask和inst2class
    # 定义函数，将points勾出的多边形，转换为两层mask，第一层表示每个像素的类别，第二层表示每个像素所属的instance
    @staticmethod
    def points_to_mask2(points: List[List[int]], shape: tuple, category_ids: List, background_as_class=True) -> Tuple[
        np.ndarray, dict]:
        """
        将points转换为mask
        :param points: 表示多边形的列表，例如[[1,1,1,2,2,2],[3,3,3,4,4,4]]
        :param shape: 图片的大小，即mask的大小
        :param category_ids: 表示每个多边形类别的列表，例如[0,1]
        :param background_as_class: 是否把背景作为一个类，若为否，则背景属于ignore，处理后为255,同时id2label里面不包含背景，
                                    0代表第一个物体，；若为是，背景属于类别0，处理后为0，同时id2label里面包含背景，0代表背景，1代表第一个物体
        :return: 一个mask，和一个字典inst2class

        """
        shape = shape[:2]
        # points为一个list，每个元素是一个list，表示一个多边形的点
        # shape为图片的大小
        mask2_list = []
        for i, point in enumerate(points):
            point = np.array(point).reshape(-1, 2)
            point = np.flip(point, axis=1)
            # 画出多边形范围
            mask = skimage.draw.polygon2mask(shape, point)
            # 画mask2，代表每个像素所属的instance。0代表没有多边形，1代表第一个instance，2代表第二个instance
            mask2 = mask * (i + 1)
            mask2_list.append(mask2)

        mask2 = np.stack(mask2_list, axis=0) if len(mask2_list) > 1 else np.expand_dims(mask2_list[0], axis=0)
        # 从多个二元mask转换为1个多元mask
        instance_seg = np.max(mask2, axis=0)

        del mask2_list, mask2
        # 计算instance到类别的映射
        # create mapping between instance IDs and semantic category IDs
        # 如果背景不作为一个类别，则没有key为0，反之则有
        # 需要把category_ids+1，因为category_ids里面0代表云，但label2id里面1代表云，0代表背景
        inst2class = {0: 0} if background_as_class else {}
        inst2class.update({k: (v + 1) for k, v in enumerate(category_ids, 1)})

        return instance_seg, inst2class

    @staticmethod
    def mask_to_points(mask: np.ndarray, inst2class: dict) -> dict:
        mask = np.array(mask)
        # 如果mask全为-1，说明没有mask
        if mask.max() == -1 or mask.max() == 0:
            return {'points': [], 'category_id': []}
        # 将mask转换为points
        # mask为一个二维的np数组，每个元素代表一个像素的类别
        # 返回一个list，每个元素是一个list，表示一个多边形的点
        points = []
        for i in range(1, mask.max() + 1):
            # 画出每个instance的mask
            mask_i = mask == i
            # 膨胀
            mask_i = skimage.morphology.binary_dilation(mask_i, footprint=skimage.morphology.disk(2))
            # 填充内部孔洞
            mask_i = remove_small_holes(mask_i, area_threshold=10)
            # 画出每个instance的轮廓
            contours = skimage.measure.find_contours(mask_i, fully_connected='high')
            # 如果contours长度为0，则把整个图作为轮廓
            if len(contours) == 0:
                contours = [
                    np.array([[0, 0], [0, mask_i.shape[1]], [mask_i.shape[0], mask_i.shape[1]], [mask_i.shape[0], 0]])]
            # 只保留最长的轮廓
            contour = sorted(contours, key=lambda x: len(x), reverse=True)[0]

            # 将轮廓转换为points
            contour = np.flip(contour, axis=1)  # 将轮廓的坐标转换为x,y
            contour = np.round(contour).astype(np.int32)
            contour = contour.tolist()
            points.append(contour)

        # 实例的序号
        inst_ids = np.unique(mask).tolist()
        # 实例对应的类别
        class_ids = [inst2class[inst_id] - 1 for inst_id in inst_ids if inst_id != 0]

        return {'points': points, 'category_ids': class_ids}

    @staticmethod
    # 将segments_info 转换为inst2class
    def segments_info_to_inst2class(segments_info: List[Dict]) -> dict:
        inst2class = {}
        for segment in segments_info:
            inst2class[segment['id']] = segment['label_id']
        return inst2class

    @staticmethod
    def data_collator_segmentation(batch: List[Dict], processor: OneFormerProcessor, return_target_sizes=False):
        # 用YolosImageProcessor处理数据
        # 如果0代表背景且id2label里没有背景，那么do_reduce_labels=True
        batch_new = processor([item['image'] for item in batch],
                              task_inputs=["panoptic"] * len(batch),
                              segmentation_maps=[item['segmentation_maps'].astype(np.uint8) for item in batch],
                              return_tensors="pt",
                              instance_id_to_semantic_id=[item['instance_id_to_semantic_id'] for item in batch],
                              # do_reduce_labels=False,
                              ignore_index=None)

        # 返回原始图像大小，每个'target_size'是元组，包含了图片的宽和高、通道数
        if return_target_sizes:
            batch_new['target_sizes'] = torch.tensor([item['target_size'][:2] for item in batch])
        return batch_new

    @classmethod
    def data_collator_clipseg(cls,batch: List[Dict], processor: CLIPSegProcessor, label2id,return_target_sizes=False):
        #将每个segmentation_maps转换为多个二元mask以及对应的文本
        for item in batch:
            item['binary_maps'],item['text'] = cls.multi_mask_to_binary_masks(item['segmentation_maps'],label2id=label2id)
            #image需要相对应复制n份
            item['image'] = [item['image']] * len(item['text'])
            #
        # 如果0代表背景且id2label里没有背景，那么do_reduce_labels=True
        batch_new = processor(text=[text for item in batch for text in item['text']],
                              images=[image for item in batch for image in item['image']],
                              padding=True,
                              return_tensors="pt")
        #将每个binary_mask放缩到[352,352]大小
        binary_masks=[cv2.resize(binary_mask,(352,352)) for item in batch for binary_mask in item['binary_maps']]
        batch_new['labels']=torch.tensor(np.array(binary_masks)).float()

        # 返回原始图像大小，每个'target_size'是元组，包含了图片的宽和高、通道数
        if return_target_sizes:
            batch_new['target_sizes'] = torch.tensor([item['target_size'][:2] for item in batch])
        return batch_new

    @staticmethod
    #将一个多元mask转换为多个二元mask
    def multi_mask_to_binary_masks(multi_mask: np.ndarray,label2id:dict) -> tuple:
        id2label={v:k for k,v in label2id.items()}
        ids_contained=np.unique(multi_mask).tolist()
        #ids_contained.remove(0)
        binary_masks=[]
        label_names=[]
        for id in ids_contained:
            binary_mask=(multi_mask==id)
            #binary_mask转换为包含0和1的float32类型
            binary_mask=binary_mask.astype(np.float32)
            binary_masks.append(binary_mask)
            label_names.append(id2label[id])
        return binary_masks,label_names


    @staticmethod
    def draw_mask(mask_in):
        mask = copy.deepcopy(mask_in)
        mask[mask == -1] = 255
        # 选取10种颜色
        cmap = matplotlib.colormaps.get_cmap('rainbow')
        unique_values = np.unique(mask).size
        cmap = ListedColormap(cmap(np.linspace(0, 255, unique_values).astype(np.uint8)))
        # 创建边界规范
        bounds = np.array(list((np.unique(mask))) + [np.max(mask) + 1])  # 需要+1，否则少一种颜色
        norm = BoundaryNorm(bounds, cmap.N)
        # 绘制颜色条
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ticks=bounds + 0.5,
            boundaries=bounds,
            orientation='vertical',
            format='%d'
        )
        plt.imshow(mask, cmap=cmap, norm=norm)

    # 显示示例
    @classmethod
    def show_example_segmentation(cls, dataset: Dataset, max_index=10, show_time=30,
                                  pred_results: List[Dict] = None, id2label: Dict = None, no_label=False):
        # plt显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 如果no_label，则必须提供pred_results
        if no_label:
            if pred_results is None:
                raise Exception('no_label=True,必须提供pred_results')

        # 打乱数据
        dataset = dataset.shuffle()

        # 显示每个样本
        for index in range(len(dataset)):
            if index > max_index:
                break
            # 转换为numpy，并转换为RGB
            one_sample = dataset[index]
            image_show = one_sample['image']
            # 转换为pil,转换为RGB
            if isinstance(image_show, np.ndarray):
                # 将channel放在最后。本身已经是RGB，不需要转换
                image_show = PILImage.fromarray(image_show.transpose(1, 2, 0))

            plt.figure(index, figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(image_show)
            print(image_show.size)

            # 打上标签
            if not no_label:
                segmentation_maps = one_sample['segmentation_maps']
                inst2class = one_sample['instance_id_to_semantic_id']

                plt.subplot(1, 2, 2)
                cls.draw_mask(segmentation_maps)
                # 如果id2label不为空，就显示类别名
                if id2label is not None:
                    inst2classname = {key: id2label[value] for key, value in inst2class.items()}
                    plt.title(inst2classname)

            # 如果pred_results不为空，就显示预测结果
            if pred_results is not None:
                result = pred_results[index]
                segmentation_maps = result['segmentation']  # 有可能为全-1矩阵
                inst2class = cls.segments_info_to_inst2class(result['segments_info'])

                plt.subplot(1, 2, 2)
                cls.draw_mask(segmentation_maps)

                # 如果id2label不为空，就显示类别名
                if id2label is not None:
                    inst2classname = {key: id2label[value] for key, value in inst2class.items()}
                    # 0代表未知
                    inst2classname[0] = '未知'
                    plt.title(inst2classname)

        plt.show(block=True)
        # plt.pause(show_time)  # 显示10s
        # plt.close()


# 语义分割预处理类
class Semantic_Segmentation_preprocess(Panorama_Segmentation_preprocess):
    # 数据增强并转换为image_processor的输入.examples是一个字典，包含了image和mask
    @classmethod
    def transform_aug_cloud(cls, examples: dict, aug=False):
        # 创建图像数据增强的函数
        transforms = albumentations.Compose([
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.GaussNoise(p=0.2, var_limit=(10.0, 20.0)),
        ]) if aug else None

        images, masks = [], []
        for image, mask in zip(examples['image'], examples['mask']):
            if isinstance(image, PILImage.Image):
                image = np.array(image.convert("RGB"))  # 转换为np
            if isinstance(mask, PILImage.Image):
                mask = np.array(mask)

            # # 将mask中所有不为0的数值都设置为1
            # mask[mask != 0] = 1

            # 进行数据增强
            if transforms is not None:
                out = transforms(image=image, mask=mask)
            else:
                out = {"image": image, "mask": mask}

            pixel_values = np.moveaxis(out["image"], -1, 0)  # 将channel放在第一维
            instance_seg_transformed = out["mask"]

            images.append(pixel_values)
            masks.append(instance_seg_transformed)

        target_size = examples['bbox_subfig_save_size'] if 'bbox_subfig_save_size' in examples \
            else [image.shape for image in images]
        # 0代表背景，1代表云
        category_maps = [{0: 0, 1: 1} for _ in range(len(images))]

        return {'image': images, 'segmentation_maps': masks,
                'instance_id_to_semantic_id': category_maps, 'target_size': target_size}

    # points转换为mask和inst2class
    # 定义函数，将points勾出的多边形，转换为mask
    @staticmethod
    def points_to_semantic_mask(points: List[List[int]], shape: tuple, category_ids: List[int],
                                background_as_class0=True) -> np.ndarray:
        #将points和category_ids按照category_ids分成字典
        category_ids = np.array(category_ids)
        points_dict = {}
        for category_id in category_ids:
            points_dict[category_id] = []
        for point, category_id in zip(points, category_ids):
            points_dict[category_id].append(point)


        print('根据points生成语义分割mask')
        shape = shape[:2]
        semantic_seg = np.zeros(shape, dtype=np.uint8)
        # points为一个list，每个元素是一个list，表示一个多边形的点
        # shape为图片的大小
        for category_id, points in tqdm(points_dict.items(),desc='将每个points画出多边形',total=len(points_dict)):

        #for i, point in tqdm(enumerate(points),desc='将每个points画出多边形',total=len(points)):
            #将points中每个point的顺序反转，因为draw.polygon2mask要求的顺序是y,x
            #points=[np.flip(np.array(point).reshape(-1, 2),axis=1) for point in points]
            #如果用cv2.fillPoly，就不需要反转
            points = [np.array(point).reshape(-1, 2) for point in points]

            # 画出多边形范围,mask是bool类型
            # mask = draw.polygon2mask(shape, point)
            mask = cv2.fillPoly(np.zeros(shape, dtype=np.uint8), points, 1)
            # 对mask进行膨胀，填充孔洞
            # mask = binary_dilation(mask, skimage.morphology.disk(2))
            # mask = remove_small_holes(mask)

            # 画mask，代表每个像素所属的类.注意category_ids里面0代表云，但label2id里面1代表云，0代表背景
            #mask = (mask * (category_ids[i] + (0 if background_as_class0 else 1))).astype(np.uint8)
            mask = (mask * (category_id + (0 if background_as_class0 else 1))).astype(np.uint8)

            # 将mask加入semantic_seg，即两者每个像素取最大值
            semantic_seg = np.maximum(semantic_seg, mask)

        return semantic_seg

    @staticmethod
    # 对单个类别的mask进行图片膨胀处理
    def mask_postprocess(mask: np.ndarray, kernel_size=5) -> np.ndarray:
        # 把mask从PIL转换为bool的np
        mask = np.array(mask).astype(bool)
        # 膨胀
        mask = skimage.morphology.binary_dilation(mask, footprint=skimage.morphology.disk(2))
        # 填充内部孔洞
        mask = remove_small_holes(mask, area_threshold=10)
        # 检测连通域并删除面积小于1%图像面积的连通域
        mask = remove_small_objects(mask, min_size=0.01 * mask.shape[0] * mask.shape[1], connectivity=2)
        # mask转换为uint8并二值化
        mask = (mask * 255).astype(np.uint8)

        return mask

    '''    
    # 计算语义分割各种指标
    @classmethod
    def compute_mIOU_ave_sample(cls, pred_masks: List[PILImage], label_masks: List[PILImage], id2label: dict):
        precision_list_all = []
        recall_list_all = []
        F1_list_all = []
        IOU_list_all = []
        pixel_accuracy_list_all = []
        FWIOU_list_all = []
        # 遍历每个mask对
        for pred_mask, label_mask in zip(pred_masks, label_masks):
            # 计算pixel accuracy
            # 计算分类正确的像素点数
            pixel_right = np.sum(pred_mask == label_mask)
            pixel_accuracy = pixel_right / (pred_mask.size)

            # 遍历每个类别
            precision_list = []
            recall_list = []
            F1_list = []
            IOU_list = []
            for label in id2label.keys():
                # 计算这个类别的TP,FP,FN
                TP = np.sum((pred_mask == label_mask) & (pred_mask == label))
                FP = np.sum((pred_mask != label_mask) & (pred_mask == label))
                FN = np.sum((pred_mask != label_mask) & (label_mask == label))
                # 如果TP+FP+FN=0,说明这个类别没有出现在这张图片中,指标为1
                if TP + FP + FN == 0:
                    precision = 1
                    recall = 1
                    F1 = 1
                    IOU = 1
                else:
                    # 计算这个类别的precision,recall,F1,IOU
                    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
                    IOU = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

                precision_list.append(precision)
                recall_list.append(recall)
                F1_list.append(F1)
                IOU_list.append(IOU)

            # 计算类别平均precision,recall,F1,IOU,FIOU
            precision = np.mean(precision_list)
            recall = np.mean(recall_list)
            F1 = np.mean(F1_list)
            IOU = np.mean(IOU_list)
            # 计算FWIOU,根据每一类出现的频率对各个类的IoU进行加权求平均
            FWIOU = np.sum([IOU_list[i] * np.sum(label_mask == label) / (label_mask.size) for i, label in
                            enumerate(id2label.keys())])

            precision_list_all.append(precision)
            recall_list_all.append(recall)
            F1_list_all.append(F1)
            IOU_list_all.append(IOU)
            pixel_accuracy_list_all.append(pixel_accuracy)
            FWIOU_list_all.append(FWIOU)

        # 计算所有图片的平均指标
        precision = np.mean(precision_list_all)
        recall = np.mean(recall_list_all)
        F1 = np.mean(F1_list_all)
        IOU = np.mean(IOU_list_all)
        pixel_accuracy = np.mean(pixel_accuracy_list_all)
        FWIOU = np.mean(FWIOU_list_all)

        return pixel_accuracy, precision, recall, F1, IOU, FWIOU
        '''


    # 将语义分割结果转换为一个mask
    @classmethod
    def semantic_mask_merge(cls, semantic_seg_result: List[dict], label2id: dict) -> np.ndarray:
        mask_list = []

        # 如果semantic_seg_result为字典,则转化为list
        if isinstance(semantic_seg_result, dict):
            semantic_seg_result = [semantic_seg_result]

        for mask_info in semantic_seg_result:
            label = mask_info['label']
            # 若label是background，则跳过
            if label == 'background':
                continue

            mask = mask_info['mask']
            # 对mask进行处理
            mask = cls.mask_postprocess(mask)
            # mask中的255转化为1，再乘以对应的id。因为label2id中0代表背景，1代表云，所以不需要减1
            mask = (mask // 255) * (label2id[label])
            mask_list.append(mask)

        # 将mask_list中的mask进行合并
        mask = np.zeros_like(mask_list[0])
        for i in range(len(mask_list)):
            mask = np.maximum(mask, mask_list[i])

        return mask

    @classmethod
    def semantic_mask_to_points(cls, semantic_seg_result: List[dict], label2id: dict) -> dict:
        points_all = []
        category_ids_all = []

        # 如果semantic_seg_result为字典,则转化为list
        if isinstance(semantic_seg_result, dict):
            semantic_seg_result = [semantic_seg_result]

        for mask_info in semantic_seg_result:
            label = mask_info['label']
            # 若label是background，则跳过
            if label == 'background':
                continue

            # 将label转换为id，需要减1
            try:
                category_id = label2id[label] - 1
            except:
                continue

            mask = mask_info['mask']
            # 对mask进行处理
            mask = cls.mask_postprocess(mask)

            # 若mask中没有任何像素，则跳过
            if np.sum(mask) == 0:
                continue

            # 画出每个instance的轮廓，contours  是长度为M的tuple，M是轮廓个数。tuple中每个元素是array，大小为N*1*2，N是轮廓上点的个数
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找图像轮廓
            # 删除第二维
            contours = list(contours)
            contours = [np.squeeze(contour) for contour in contours]

            # contours = skimage.measure.find_contours(mask)
            # 如果contours长度为0，则把整个图作为轮廓
            if len(contours) == 0:
                contours = [np.array([[0, 0], [0, mask.shape[0]], [mask.shape[1], mask.shape[0]], [mask.shape[1], 0]])]

            # 将轮廓坐标转换为x,y
            contours = [np.round(np.flip(contour, axis=1)).astype(np.int32).tolist() for contour in contours]

            # 对应的类别
            category_ids = [category_id for _ in range(len(contours))]
            # 记录
            points_all.extend(contours)
            category_ids_all.extend(category_ids)

        return {'points': points_all, 'category_ids': category_ids_all}

    # 数据增强并转换为image_processor的输入，语义分割
    @classmethod
    def transform_aug_from_points(cls, examples: dict, aug=False, categories_keep=None):
        # 创建图像数据增强的函数
        if aug:
            transforms = albumentations.Compose([
                albumentations.Resize(512, 512),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.GaussNoise(p=0.3, var_limit=(10.0, 20.0)),
            ])
        else:
            transforms = None

        images, masks, instance_id_to_semantic_ids = [], [], []
        for image, points, category_id in zip(examples['image'], examples['points'], examples['category_id']):
            # 筛选category_id和points中的元素，只保留categories_keep中的元素
            if categories_keep is not None:
                points = [point for point, category in zip(points, category_id) if category in categories_keep]
                category_id = [category for category in category_id if category in categories_keep]
            # 如果没有符合的category_id，跳过
            if len(category_id) == 0:
                continue

            # 转换为np
            if isinstance(image, PILImage.Image):
                image = np.array(image.convert("RGB"))

            # points转换为mask，一个多元mask
            semantic_seg = cls.points_to_semantic_mask(points, image.shape[:2], category_id)

            # 进行数据增强
            if transforms is not None:
                out = transforms(image=image, mask=semantic_seg)
            else:
                out = {"image": image, "mask": semantic_seg}

            pixel_values = np.moveaxis(out["image"], -1, 0)  # pixel_values需要将channel放在第一维
            semantic_seg_transformed = out["mask"]

            images.append(pixel_values)
            masks.append(semantic_seg_transformed)

            # instance_id_to_semantic_id为{1:1,2:2...}，即每个实例的id与其语义id相同
            instance_id_to_semantic_id = {i: i for i in range(1, len(set(category_id)) + 1)}
            instance_id_to_semantic_ids.append(instance_id_to_semantic_id)

        return {'image': images, 'segmentation_maps': masks, 'target_size': examples['bbox_subfig_save_size'],
                'instance_id_to_semantic_id': instance_id_to_semantic_ids}


# 图像转文字预处理类
class Image2Text_preprocess:
    # 数据增强并转换为image_processor的输入
    @classmethod
    def transform_aug(cls, examples: dict, aug=False):
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

    @staticmethod
    # 定义图像转文字的data_collator
    def data_collator_git(batch: List[Dict], processor: GitProcessor):
        # 用GitProcessor处理数据
        batch_new = processor.__call__(text=[item['caption'] for item in batch],
                                       images=[item['image'] for item in batch],
                                       padding="max_length",
                                       return_tensors='pt')
        batch_new['labels'] = batch_new['input_ids'].clone()

        return batch_new

    @staticmethod
    # 定义图像转文字的data_collator
    def data_collator_blip(batch: List[Dict], processor: BlipProcessor):
        # 用GitProcessor处理数据
        batch_new = processor.__call__(text=[item['caption'] for item in batch],
                                       images=[item['image'] for item in batch],
                                       padding=True,
                                       max_length=128,
                                       return_tensors='pt')
        batch_new['labels'] = batch_new['input_ids'].clone()

        return batch_new

    @staticmethod
    # 定义图像转文字的data_collator
    def data_collator_blip2(batch: List[Dict], processor: Blip2Processor):
        # 用GitProcessor处理数据
        batch_new = processor.__call__(text=[item['caption']+processor.tokenizer.eos_token for item in batch],
                                       images=[item['image'] for item in batch],
                                       padding=True,
                                       max_length=128,
                                       return_tensors='pt')
        batch_new['labels'] = batch_new['input_ids'].clone()

        return batch_new

    @staticmethod
    # 定义图像转文字的data_collator
    def data_collator_blip_gen(batch: List[Dict], processor: BlipProcessor):
        # 用GitProcessor处理数据
        batch_new = processor.__call__(images=[item['image'] for item in batch],
                                       return_tensors='pt')
        # batch_new['labels']=batch_new['input_ids'].clone()

        return batch_new

    @staticmethod
    # 定义图像转文字的data_collator,用于生成
    def data_collator_git_gen(batch: List[Dict], processor: GitProcessor, QA=False):
        # 用GitProcessor处理数据
        batch_new = {}
        pixel_values = processor(images=[item['image'] for item in batch], return_tensors="pt").pixel_values

        if QA:
            input_ids = processor(text=[item['caption'].split('? ')[0] + '?' for item in batch],
                                  add_special_tokens=False).input_ids
            input_ids = [[processor.tokenizer.cls_token_id] + input_ids_ for input_ids_ in input_ids]
            input_ids = torch.tensor(input_ids)
            # 不需要labels，如果batch_size为1，不需要attention_mask
            batch_new = {'pixel_values': pixel_values, 'input_ids': input_ids}

        else:
            batch_new['pixel_values'] = pixel_values

        return batch_new

    # 将错误描述转换为caption的函数
    @staticmethod
    def error2caption(example, error_zh2en: dict = None, need_question=False):
        error_list = example['错误描述']
        # 若error_list中有元素不在error_zh2en中，则将其删除
        error_list = [error for error in error_list if error in error_zh2en]

        # 将错误描述转换为英文
        if error_zh2en is not None:
            error_list = [error_zh2en[error] for error in error_list]
        # 去重
        error_list = list(set(error_list))

        # 将错误描述转换为caption，即连接所有错误描述
        example['caption'] = ', '.join(error_list)
        # 若error_list为空，则caption为'no error'
        if len(error_list) == 0:
            example['caption'] = 'no error'

        # 加上问题
        if need_question:
            example['caption'] = 'Is there any error? ' + example['caption']

        return example


class  ClipSeg_utils:
    # 将一个图片的输出转换为一个mask
    @staticmethod
    def logits_to_mask(logits, texts, label2id):
        # 将logits转换为预测的mask
        # 选取分数最高的text
        mask = torch.argmax(logits, dim=0)
        mask = mask.detach().cpu().numpy().astype(np.uint8)
        # 建立mask中的id与真实id的对应关系
        mask_id_to_real_id = [label2id[text] for text in texts]

        # 将mask转换为对应的label
        mask = np.vectorize(lambda x: mask_id_to_real_id[x])(mask)
        return mask

    # 对一张图片进行预测
    @classmethod
    def predict_one_image(cls,image, texts, label2id, model, processor:CLIPSegProcessor):
        inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
        inputs=inputs.to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits
        #如果logits只有2维，加上第一维
        if len(logits.shape)==2:
            logits=logits.unsqueeze(0)
        mask = cls.logits_to_mask(logits, texts, label2id)
        return mask

    # 对一张图片进行预测
    @classmethod
    def predict_one_image_groupvit(cls,image:np.ndarray, texts, label2id, model, processor):
        inputs = processor(text=texts, images=image, padding=True, return_tensors="pt")
        inputs=inputs.to(model.device)
        outputs = model(**inputs, output_segmentation=True, return_loss=False)
        segmentation_logits = outputs.segmentation_logits
        # First, rescale logits to original image size
        logits = nn.functional.interpolate(segmentation_logits.detach().cpu(),
                                           size=image.shape[1:],  # (height, width)
                                           mode='bilinear',
                                           align_corners=False)

        logits = logits[0]
        mask = cls.logits_to_mask(logits, texts, label2id)

        return mask

    # 对数据集中所有图片进行预测
    @classmethod
    def predict_all_images(cls, dataset, label2id, model, processor):
        id2label={v:k for k,v in label2id.items()}
        masks = []
        for i in tqdm(range(len(dataset)), desc='predicting',total=len(dataset)):
            image = dataset[i]['image']
            #cate_ids = dataset[i]['labels']
            #如果dataset[i]有target_size，就用target_size，否则用image的size
            if 'target_size' in dataset[i]:
                target_size = dataset[i]['target_size']
            else:
                target_size = image.size[0:2]
            # 只选取有的类别进行预测。根据标签mask生成texts
            cate_ids=np.unique(dataset[i]['segmentation_maps']).tolist()
            texts = [id2label[cate_id] for cate_id in cate_ids if cate_id in id2label]
            #如果texts中没有background，加上background
            # if 'background' not in texts:
            #     texts=['background']+texts
            # 预测mask
            if isinstance(model,GroupViTModel):
                mask = cls.predict_one_image_groupvit(image, texts, label2id, model, processor)
            else:
                mask = cls.predict_one_image(image, texts, label2id, model, processor)
            # 将mask转换为原始图像大小
            mask = cv2.resize(mask, target_size[::-1], interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return masks
