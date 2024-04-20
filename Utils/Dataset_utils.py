import os

import pandas as pd

os.environ["TQDM_INTERVAL"] = '1'
import cv2

from typing import List, Tuple,Union

import skimage
from skimage import morphology
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
from datasets import Dataset


#对mask的各种处理
class MaskTransform:
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

    # points转换为mask和inst2class
    # 定义函数，将points勾出的多边形，转换为两层mask，第一层表示每个像素的类别，第二层表示每个像素所属的instance
    @staticmethod
    def points_to_instance_mask(points: List[List[int]], shape: tuple, category_ids: List, background_as_class=True) -> Tuple[
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

    # points转换为mask和inst2class。points描述了多个多边形
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


            # 画mask，代表每个像素所属的类.注意category_ids里面0代表云，但label2id里面1代表云，0代表背景
            mask = (mask * (category_id + (0 if background_as_class0 else 1))).astype(np.uint8)

            # 将mask加入semantic_seg，即两者每个像素取最大值
            semantic_seg = np.maximum(semantic_seg, mask)

        return semantic_seg


    # 对单个类别的二元mask进行图片膨胀处理，输出二元mask中值为0或255的mask
    @staticmethod
    def binary_mask_postprocess(mask: np.ndarray, min_area_threshold=0.001) -> np.ndarray:
        # 把mask从PIL转换为bool的np
        mask = np.array(mask).astype(bool)
        # 膨胀
        mask = skimage.morphology.binary_dilation(mask, footprint=skimage.morphology.disk(2))
        # 填充内部孔洞
        mask = remove_small_holes(mask, area_threshold=10)
        # 检测连通域并删除面积小于1%图像面积的连通域
        mask = remove_small_objects(mask, min_size=min_area_threshold * mask.shape[0] * mask.shape[1], connectivity=2)
        # mask转换为uint8并二值化
        mask = (mask * 255).astype(np.uint8)

        return mask

    # 对多元mask进行图片膨胀处理
    @classmethod
    def mask_postprocess_multi(cls,mask: np.ndarray,min_area_threshold=0.001) -> np.ndarray:
        new_mask = np.zeros_like(mask)
        #转化为多个二元mask
        for label_id in np.unique(mask):
            if label_id == 0:
                continue
            binary_mask = mask == label_id
            binary_mask = cls.binary_mask_postprocess(binary_mask,min_area_threshold)
            new_mask[binary_mask.astype(bool)] = label_id
        return new_mask


    # 将单个图片的语义分割结果转换为一个mask
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
            mask = cls.binary_mask_postprocess(mask)
            # mask中的255转化为1，再乘以对应的id。因为label2id中0代表背景，1代表云，所以不需要减1
            mask = (mask // 255) * (label2id[label])
            mask_list.append(mask)

        # 将mask_list中的mask进行合并
        mask = np.zeros_like(mask_list[0])
        for i in range(len(mask_list)):
            mask = np.maximum(mask, mask_list[i])

        return mask

    @classmethod
    def semantic_result_to_points(cls, semantic_seg_result: List[dict], label2id: dict) -> dict:
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
            mask = cls.binary_mask_postprocess(mask)

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

    @classmethod
    def semantic_mask_to_points(cls, semantic_mask: np.ndarray,min_area_threshold=0) -> dict:
        points_all = []
        category_ids_all = []

        for label_id in np.unique(semantic_mask):
            # 若label_id是0，则跳过
            if label_id == 0:
                continue
            # 获取对应于label_id的mask
            mask = (semantic_mask == label_id)

            # 对mask进行处理，膨胀，删除小区域
            mask = cls.binary_mask_postprocess(mask, min_area_threshold=min_area_threshold)

            # 画出每个instance的轮廓，contours  是长度为M的tuple，M是轮廓个数。tuple中每个元素是array，大小为N*1*2，N是轮廓上点的个数
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找图像轮廓
            # 删除第二维
            contours = list(contours)
            contours = [np.squeeze(contour) for contour in contours]

            # contours = skimage.measure.find_contours(mask)

            # 将轮廓坐标转换为x,y。如果使用cv2则不需要转换
            contours = [np.round(contour).astype(np.int32).tolist() for contour in contours]

            # 对应的类别
            category_ids = [label_id for _ in range(len(contours))]
            # 记录
            points_all.extend(contours)
            category_ids_all.extend(category_ids)

        return {'points': points_all, 'category_ids': category_ids_all}


    # 把mask转为category_id
    @staticmethod
    def mask_to_category_id(mask: np.ndarray):
        # 获取这张图片的category_id_unique
        category_id_unique_ = np.unique(mask).tolist()
        # 计算每种category_id_unique的面积占比
        area = [np.count_nonzero(mask == i) / mask.size for i in category_id_unique_]
        # 将category_id_unique_和area合并为一个字典
        category_id_unique_area = dict(zip(category_id_unique_, area))

        return category_id_unique_,category_id_unique_area



    # 画出mask
    @staticmethod
    def draw_mask(mask,show=True,save=False):
        # 选取10种颜色
        cmap = matplotlib.colormaps.get_cmap('Reds')
        color_nums = int(np.max(mask) + 1)
        cmap = ListedColormap(cmap(np.linspace(0, 255, color_nums).astype(np.uint8)))
        #像素值0对应白色
        cmap.set_under(color='white')
        # 创建边界规范
        bounds = np.array(list((np.unique(mask))) + [color_nums])  # 需要+1，否则少一种颜色
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
        plt.axis('off')
        if save:
            plt.savefig('mask.png')
        if show:
            plt.show()

    @staticmethod
    #将一个多元mask转换为多个二元mask
    def multi_mask_to_binary_masks(multi_mask: np.ndarray,label2id:dict=None) -> tuple:
        id2label={v:k for k,v in label2id.items()}
        ids_contained=np.unique(multi_mask).tolist()
        #背景也要作为一个类别的mask
        #ids_contained.remove(0)
        binary_masks=[]
        label_names=[]
        for id in ids_contained:
            binary_mask=(multi_mask==id)
            #binary_mask转换为包含0和1的float32类型
            binary_mask=binary_mask.astype(np.uint8)
            binary_masks.append(binary_mask)
            label_names.append(id2label[id])
        return binary_masks,label_names

    # 显示一个二值mask
    @staticmethod
    def visual_mask(result, img, label_name: str = 'cloud'):
        # res是List[List[dict],每个dict代表一种类别的mask，提取wall的mask
        cloud_mask = [i['mask'] for i in result if i['label'] == label_name]
        cloud_mask = cloud_mask[0] if len(cloud_mask) > 0 else np.zeros_like(result[0]['mask'])
        # 转换为np并把255转换为1
        cloud_mask = np.array(cloud_mask) / 255
        cloud_mask[cloud_mask == -1] = 1
        # back_mask=[i['mask']  for result in res for i in result if i['label']=='background'][0]
        # 对mask进行膨胀，并填充孔洞
        cloud_mask = morphology.binary_dilation(cloud_mask, morphology.disk(5))
        cloud_mask = morphology.remove_small_holes(cloud_mask)

        plt.subplot(1, 2, 1)
        plt.title(label_name)
        # 创建边界规范
        cmap = matplotlib.colormaps.get_cmap('rainbow')
        cmap = ListedColormap(cmap(np.linspace(0, 255, 2).astype(np.uint8)))
        bounds = np.array([0, 1, 2])  # 需要+1，否则少一种颜色
        norm = BoundaryNorm(bounds, cmap.N)
        # 绘制颜色条，高度为0.1，宽度为0.8
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ticks=bounds + 0.5,
            boundaries=bounds,
            orientation='vertical',
            format='%d',
            shrink=0.8,
        )
        plt.imshow(cloud_mask, cmap=cmap, norm=norm)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show(block=False)


#cate_id和caption相互转化
class CategoryId2Caption:
    # 把标签id转化为caption
    @staticmethod
    def category_id_to_caption(category_ids: list, label2id: dict,background_as_class0=True,category_area=None,language='zh'):
        id2label = {v: k for k, v in label2id.items()}
        captions = []
        for category_id in category_ids:
            # 跳过背景
            if background_as_class0 and category_id == 0:
                continue
            #获取标签名
            label_name = id2label[category_id]
            #如果category_area不为空，则加上面积占比。占比大于0.3则标签前面加上“大面积的”，否则加上“小面积的”
            if category_area is not None:
                if category_area[category_id] > 0.3:
                    label_name = ("大面积的" if language=='zh' else "large area of ") + label_name
                elif category_area[category_id] <= 0.3 and category_area[category_id] > 0.001:
                    label_name = ("小面积的" if language=='zh' else "small area of ") + label_name
                else:
                    continue
            captions.append(label_name)
        # 如果captions为空，则返回“卫星影像完全合格”
        if len(captions) == 0:
            return  "卫星图像完全合格。" if language=='zh' else "The satellite image has only background."

        caption_text = "卫星图像有" + '，'.join(captions) + "。" if language=='zh' else "A satellite image has " + ' and '.join(captions)
        return caption_text

    #把caption转化为标签id
    @staticmethod
    def caption_to_category_id(caption:str,label2id:dict,return_label_names:bool=False):
        #如果caption包含"合格"，则返回[]
        if "合格" in caption:
            return []
        else:
            #获取caption中出现的所有标签名
            label_names=[label_name for label_name in label2id.keys() if label_name in caption]
            #获取标签名对应的标签id
            category_ids=[label2id[label_name] for label_name in label_names]
            if return_label_names:
                return category_ids,label_names
            else:
                return category_ids

    #把标签从list转化为binary
    @staticmethod
    def label_to_binary(label:List[int],num_labels:int,reduce_labels:bool=True):
        if reduce_labels:
            binary_label=[0]*num_labels
            for i in label:
                if i>0:
                    binary_label[i-1]=1
        else:
            binary_label=[0]*num_labels
            for i in label:
                binary_label[i]=1
        return binary_label


#将image列和mask列的路径中的.替换为dataset_path
def replace_dot(dataset:Union[Dataset,pd.DataFrame],dataset_path:str,reverse:bool=False):

    def replace_head_dot(path:str,path_add):
        if path.startswith('.'):
            path=path_add+path[1:]
        return path


    if not reverse:
        dataset_path_add=os.path.dirname(dataset_path)+'/'
        if isinstance(dataset,Dataset):
            #如果有mask列
            if "mask" in dataset.column_names:
                dataset=dataset.map(lambda data:{"image":replace_head_dot(data["image"],dataset_path_add).replace('\\','/'),
                                                "mask":replace_head_dot(data["mask"],dataset_path_add).replace('\\','/')})
            else:
                dataset=dataset.map(lambda data:{"image":replace_head_dot(data["image"],dataset_path_add).replace('\\','/')})
        elif isinstance(dataset,pd.DataFrame):
            if "mask" in dataset.columns:
                dataset["mask"] = dataset["mask"].apply(
                    lambda x: replace_head_dot(x, dataset_path_add).replace('\\', '/'))

            dataset["image"]=dataset["image"].apply(lambda x:replace_head_dot(x,dataset_path_add).replace('\\','/'))
        else:
            raise TypeError("dataset type must be Dataset or DataFrame")
    else:
        dataset_path_reduce=os.path.dirname(dataset_path)
        #将image列和mask列的路径中的dataset_path中的替换为.
        if isinstance(dataset,Dataset):
            dataset=dataset.map(lambda data:{"image":data["image"].replace(dataset_path_reduce,"."),
                                            "mask":data["mask"].replace(dataset_path_reduce,".")})
        elif isinstance(dataset,pd.DataFrame):
            dataset["image"]=dataset["image"].apply(lambda x:x.replace(dataset_path_reduce,"."))
            dataset["mask"]=dataset["mask"].apply(lambda x:x.replace(dataset_path_reduce,"."))
        else:
            raise TypeError("dataset type must be Dataset or DataFrame")
    return dataset

#将字典中所有value保留5位小数。如果value是array或list，则保留5位小数
def dict_round(data:dict,n=5):
    #复制字典
    data=data.copy()

    for k,v in data.items():
        if isinstance(v,(int,float)):
            data[k]=round(v,n)
        elif isinstance(v,(list,np.ndarray)):
            data[k]=[round(i,n) for i in v]
        else:
            raise TypeError("dict value type must be int,float,list or np.ndarray")

    return data


class Mask2QA:
    label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '拼接痕迹': 4, '拼接错误': 5, '扭曲': 6, '模糊': 7,
                '光谱溢出': 8,
                '条状噪声': 9,'像素缺失': 10}
    # 将label分为3类
    label_area = ["云", "阴影", "拉花", "扭曲", "模糊", "光谱溢出"]
    label_seam = ["拼接痕迹", "拼接错误",'色差']
    label_pixel = ["条状噪声", '像素缺失']
    id2position = {
        0: "左上角",
        1: "上方",
        2: "右上角",
        3: "左边",
        4: "中间",
        5: "右边",
        6: "左下角",
        7: "下方",
        8: "右下角"
    }
    # 把mask转为每个category_id的描述，不包含0
    @classmethod
    def mask_to_attrs(cls, mask: np.ndarray,all_id:bool=True):
        # 获取这张图片的category_id_unique
        if not all_id:
            category_id_unique_ = np.unique(mask).tolist()
        else:
            category_id_unique_ = range(0,len(cls.label2id))
        # 删除0
        category_id_unique_ = [i for i in category_id_unique_ if i != 0]
        # 计算每种category_id_unique的面积占比
        area = [np.count_nonzero(mask == i) / mask.size for i in category_id_unique_]
        # 计算每种category_id_unique的连通域个数和面积最多的位置
        num = [cv2.connectedComponentsWithStats((mask == i).astype(np.uint8))[0] for i in category_id_unique_]  # 连通域个数
        area_pos = [cls.judge_9position_by_area(mask == i) for i in category_id_unique_]  # 面积最多的位置
        #计算每个category_id_unique的外接矩形
        rects=[cv2.boundingRect((mask == i).astype(np.uint8)) for i in category_id_unique_]
        #计算每个category_id_unique的位置，用质心
        center_pos=[cls.judge_9position_by_center(mask == i) for i in category_id_unique_]
        # 将以上几种属性合并为元组的列表
        attrs = list(zip(category_id_unique_, area, num, area_pos,center_pos,rects))
        return attrs

    # 根据attrs生成问答对
    @classmethod
    def attrs_to_qa(cls, attrs):
        id2label = {v: k for k, v in cls.label2id.items()}
        QAs = []
        # 获取全部area不为0的label_name
        label_names=[id2label[attr[0]] for attr in attrs if attr[1]!=0]
        #生成总体问题
        question="这张卫星图像是否有质量问题？"
        #如果全部label_name都是背景，则返回“没有质量问题”
        if all([label_name=="背景" for label_name in label_names]):
            answer="否，这张卫星图像没有质量问题。"
        else:
            #有其他label_name，则返回“有质量问题,问题类型有”+label_name
            answer="是，这张卫星图像有质量问题，问题类型有：" + "、".join(label_names) + "。"
        QAs.append({"question":question,"answer":answer})


        for attr in attrs:
            label_name = id2label[attr[0]]
            area = attr[1]
            num = attr[2]
            area_pos = attr[3]
            center_pos = attr[4]
            w, h = attr[5][2], attr[5][3]
            question = "这张卫星图像中是否存在" + label_name + "？"
            #如果面积占比为0，则返回“不存在”
            if area == 0:
                answer = "否，这张卫星图像中不存在" + label_name + "。"
            else:
                #对label类型分类
                if label_name in cls.label_area:
                    #如果面积大于0.4，返回“是，这张卫星图像几乎全是”
                    if area > 0.4:
                        answer = "是，这张卫星图像几乎全是" + label_name + "。"
                    else:
                        # 如果面积占比大于0.4，则返回“大面积的”，若在0.4-0.1之间，则返回“中等面积的”，若在0.1-0.01之间，则返回“小面积的”，若在0.01-0之间，则返回“微小的”
                        area2desc = {0.4: "大面积的", 0.1: "中等面积的", 0.01: "小面积的", 0: "微小的"}
                        area_desc = area2desc[max([i for i in area2desc.keys() if area > i])]
                        # 如果连通域个数大于1，则返回“多处”，否则返回“一处”
                        num_desc = "多处" if num > 1 else "一处"
                        # 组成回答
                        answer = "是，这张卫星图像中存在" + num_desc + area_desc + label_name + "，主要位于图像的" + area_pos
                        # 如果是微小的，把存在改成”可能存在“
                        if area_desc == "微小的":
                            answer = answer.replace("存在", "可能存在")
                elif label_name in cls.label_seam:
                    #根据长宽判断是横向还是纵向
                    hengzong="横向" if w>h else "纵向"
                    #答案格式为“是，这张卫星图像中存在一条{}的痕迹，位于图像的{}，是一条{}”
                    answer="是，这张卫星图像中存在一条"+label_name+"的痕迹，位于图像的"+center_pos+"，大致是"+hengzong+"的。"
                elif label_name in cls.label_pixel:
                    #答案格式为’是，这张卫星图像中存在{}处{}，主要位于图像的{}‘
                    answer="是，这张卫星图像中存在"+str(num)+"处"+label_name+"，主要位于图像的"+area_pos+"。"
                else:
                    raise Exception("{}不在label_area,label_seam,label_pixel中".format(label_name))

            QAs.append({"question": question, "answer": answer})

        return QAs

    # 根据mask生成问答对
    @classmethod
    def mask_to_qa(cls, mask:np.ndarray,all_id:bool=True)->list:
        attrs = cls.mask_to_attrs(mask,all_id=all_id)
        QAs = cls.attrs_to_qa(attrs)
        return QAs

    # 判断某个坐标在图中的九宫格中的哪个位置
    @classmethod
    def judge_9position_by_center(cls, img):
        #img若为全0，则返回“无”
        if np.count_nonzero(img)==0:
            return "无"
        # 获取x,y
        xy=cv2.connectedComponentsWithStats(img.astype(np.uint8))[3][1].astype(int)
        x,y=xy[0],xy[1]
        # 获取图像的长宽
        height, width = img.shape
        # 计算每个格子的长宽
        width_ = width // 3
        height_ = height // 3
        # 计算x,y在第几行第几列
        row = y // height_
        col = x // width_
        # 计算x,y在第几个格子
        id = row * 3 + col
        # 根据num返回“左上角”、“上方”、“右上角”、“左边”、“中间”、“右边”、“左下角”、“下方”、“右下角”
        return cls.id2position[id]

    # 判断图片九宫格中哪个格子为1的像素最多
    @classmethod
    def judge_9position_by_area(cls,img,return_num=1):
        # 获取图像的长宽
        height, width = img.shape
        # 计算每个格子的长宽
        width_ = width // 3
        height_ = height // 3
        # 计算每个格子的面积
        area = width_ * height_
        # 计算每个格子的像素个数
        num = [np.count_nonzero(img[i * height_: (i + 1) * height_, j * width_: (j + 1) * width_]) for i in range(3) for
               j in range(3)]
        # 计算每个格子的像素占比
        area_ratio = [i / area for i in num]
        if return_num==1:
        # 返回最大值的索引
            id = area_ratio.index(max(area_ratio))
            # 根据num返回“左上角”、“上方”、“右上角”、“左边”、“中间”、“右边”、“左下角”、“下方”、“右下角”
            return cls.id2position[id]
        elif return_num==2:
            id1=area_ratio.index(max(area_ratio))
            area_ratio[id1]=0
            id2=area_ratio.index(max(area_ratio))
            return cls.id2position[id1],cls.id2position[id2]

    # 从mask生成id
    @classmethod
    def judge_5position_by_area(cls,mask, label_id=None):
        # 选取mask中等于label_id的部分
        if label_id is not None:
            mask = mask == label_id
        # 判断为1的区域位于图片的左边还是中间还是右边，对应0,1,2
        left = mask[:, :mask.shape[1] // 3]
        right = mask[:, mask.shape[1] // 3 * 2:]
        middle1 = mask[:, mask.shape[1] // 3:mask.shape[1] // 3 * 2]
        # 判断哪个区域的1最多
        left_num = left.sum()
        right_num = right.sum()
        middle_num1 = middle1.sum()

        # 判断为1的区域位于图片的上边还是中间还是下边，对应3，4，5
        up = mask[:mask.shape[0] // 3, :]
        down = mask[mask.shape[0] // 3 * 2:, :]
        middle2 = mask[mask.shape[0] // 3:mask.shape[0] // 3 * 2, :]
        # 判断哪个区域的1最多
        up_num = up.sum()
        down_num = down.sum()
        middle_num2 = middle2.sum()

        max_id = np.argmax([left_num, right_num, middle_num1, up_num, down_num, middle_num2])
        return max_id

    # 从id生成标题
    @classmethod
    def get_title_from_id(cls,id, problem_type='拼接错误'):
        # id为0,1,2,3,4,5
        id2position = {0: '左边', 1: '右边', 2: '中间', 3: '上边', 4: '下边', 5: '中间'}
        id2hs = {0: '竖线', 1: '竖线', 2: '竖线', 3: '横线', 4: '横线', 5: '横线'}
        # 生成caption
        caption = '这张卫星影像存在' + problem_type + '，分界线在' + id2position[id] + '，是一条' + id2hs[id] + '。'
        return caption



