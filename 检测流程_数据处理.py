import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForSemanticSegmentation, AutoProcessor, \
    AutoModelForImageClassification
import numpy as np
from Utils.Dataset_utils import MaskTransform
from typing import List, Union, Tuple
import gc
from 检测流程_模型检测 import SegmentationProcess, ClassificationAndSegmentationProcess
from osgeo import gdal
from osgeo import gdal_array as ga  # 用于引入一个模块的同时为该模块取一个别名
from osgeo.gdalconst import GA_ReadOnly
from tqdm import tqdm
from PIL import Image as PILImage
from shp_write import write_shp
import torch
from matplotlib import pyplot as plt
import argparse

print('当前工作目录为：', os.getcwd())


class Seg_Big_Image:
    def __init__(self, big_image: np.ndarray,
                 seg_process: Union[SegmentationProcess, ClassificationAndSegmentationProcess],
                 patch_size: Union[int, List[int]] = 512,
                 overlap: Union[int, List[int]] = None, patch_size_ratio: Union[float, List[float]] = None,
                 overlap_ratio: Union[float, List[float]] = None, **kwargs):
        self.big_image = big_image
        self.seg_process = seg_process

        # 若patch_size_ratio不为None，则将patch_size和overlap转换为比例
        if patch_size_ratio is not None and overlap_ratio is not None:
            self.patch_size = int(patch_size_ratio * big_image.shape[0]) if isinstance(patch_size_ratio, float) else \
                [int(ratio * big_image.shape[0]) for ratio in patch_size_ratio]
            self.overlap = int(overlap_ratio * big_image.shape[0]) if isinstance(overlap_ratio, float) else \
                [int(ratio * big_image.shape[0]) for ratio in overlap_ratio]
        else:
            self.patch_size = patch_size
            # overlap默认为patch_size的0.1
            if isinstance(patch_size, int):
                self.overlap = overlap if overlap is not None else int(patch_size * 0.1)
            else:
                self.overlap = overlap if overlap is not None else [int(0.1 * patch_size_) for patch_size_ in
                                                                    patch_size]

    # 将一张大图切分为n*n的小图，并记录每个小图的坐标
    @staticmethod
    def split_image(image: np.ndarray, patch_size: int, overlap: int) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        patches = []
        height, width = image.shape[:2]
        for i in range(0, height, patch_size - overlap):
            for j in range(0, width, patch_size - overlap):
                if i + patch_size < height and j + patch_size < width:
                    patch = image[i:i + patch_size, j:j + patch_size]
                elif i + patch_size >= height and j + patch_size < width:
                    patch = image[i:, j:j + patch_size]
                elif i + patch_size < height and j + patch_size >= width:
                    patch = image[i:i + patch_size, j:]
                else:
                    patch = image[i:, j:]
                patches.append((patch, (i, j)))
        return patches

    # 根据左上角坐标，将小图的mask拼接起来，mask之间有重叠，重叠的部分取最大值
    def merge_image(self, masks: List[np.ndarray]) -> np.ndarray:
        # 先创建一个全0的大图
        height, width = self.big_image.shape[:2]
        mask = np.zeros((height, width)).astype(np.uint8)
        # 将每个小图的mask拼接到大图上
        for i in range(len(masks)):
            # 获取小图的mask和左上角坐标
            patch, coord = masks[i]
            coord1, coord2 = coord
            # 将小图的mask拼接到大图上
            mask[coord1:coord1 + patch.shape[0], coord2:coord2 + patch.shape[1]] = np.maximum(
                mask[coord1:coord1 + patch.shape[0], coord2:coord2 + patch.shape[1]], patch)
        return mask

    # 对图片列表做分割
    @staticmethod
    def seg_images(images: List[PILImage.Image], seg_process: SegmentationProcess) -> List[np.ndarray]:
        preds = []
        # 获取第一张图片的shape
        shape = images[0].size[0]
        for image in tqdm(images, desc="分割图片" + str(shape), total=len(images)):
            preds.append(seg_process.predict_one_image(image))
        return preds

    # 将一张大图切分为10*10的小图，然后对小图做分割，再把mask拼接起来
    def seg_big_image_once(self, patch_size, overlap) -> np.ndarray:
        # 先切分
        patches = self.split_image(self.big_image, patch_size, overlap)
        # 再分割
        masks = self.seg_images([PILImage.fromarray(patch) for patch, _ in patches], self.seg_process)
        # 再拼接
        mask = self.merge_image(list(zip(masks, [coord for _, coord in patches])))
        del masks
        del patches
        gc.collect()
        return mask

    def seg_big_image(self):
        # 如果patch_size和overlap是列表，则循环遍历
        if isinstance(self.patch_size, list) and isinstance(self.overlap, list):
            masks_final = np.zeros(self.big_image.shape[:2])
            for patch_size, overlap in zip(self.patch_size, self.overlap):
                mask_once = self.seg_big_image_once(patch_size, overlap)
                masks_final = np.maximum(masks_final, mask_once)
            return masks_final
        else:
            return self.seg_big_image_once(self.patch_size, self.overlap)


# 读取遥感图像，做分割，转化为shp文件
class DetectRemoteSense(MaskTransform, Seg_Big_Image):
    def __init__(self, img_path, label2id, shp_save_dir, seg_process, label_to_chose=range(11), **kwargs):
        self.img_path = img_path
        self.shp_save_dir = shp_save_dir
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.label_to_chose = label_to_chose

        # 获取图片信息
        # 如果img_path以jpg或png结尾，则使用PIL读取图片并转换为array
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img_arr = np.array(PILImage.open(img_path))
            img_arr = img_arr[:, :, :3]
            tfw_values = [0, 1, 0, 0, 0, -1]
        else:
            # 获取图片array
            img_arr = ga.LoadFile(self.img_path)
            # 若图片位4通道，则转换为3通道
            if img_arr.shape[0] == 4:
                img_arr = img_arr[:3]
            # 如果第一维是通道数，则转换为[h,w,c]
            if img_arr.shape[0] == 3:
                img_arr = img_arr.transpose(1, 2, 0)  # 转换为[h,w,c]

            # 获取tfw文件中的信息
            dr = gdal.Open(self.img_path, GA_ReadOnly)
            tfw_values = dr.GetGeoTransform()

        tfw_keys = ['左上角x坐标', 'x方向分辨率', 'x旋转参数', '左上角y坐标', 'y旋转参数', 'y方向分辨率']
        self.tfw_info = dict(zip(tfw_keys, tfw_values))

        # 从int32转换为uint8
        img_arr = img_arr.astype(np.uint8)
        self.img_arr = img_arr
        print('已加载图片', self.img_path)
        # 读取遥感图像的分割模型
        super().__init__(big_image=self.img_arr, seg_process=seg_process, **kwargs)


    # 将polygons_list和category_ids_list转化为shp文件
    def polygons2shp(self, polygons_list, category_ids_list):
        # 每个polygon外面套一个list
        polygons_list = [[polygon] for polygon in polygons_list]
        # 增加category_id列，并把label转换为中文
        field_values_list = [[category_id, self.id2label[category_id]] for category_id in category_ids_list]

        # self.shp_dir不存在则创建
        if not os.path.exists(self.shp_save_dir):
            os.makedirs(self.shp_save_dir)

        # 设置shp属性格式
        field_names = ['category_id', 'labels']
        field_types = ['N', 'C']
        field_sizes = [50, 50]
        field_decimals = [0, 0]
        shape_type = 5  # 5代表polygon

        write_shp(self.shp_save_dir + '/pred_polygons', shape_type, polygons_list,
                  field_names, field_types, field_sizes, field_decimals, field_values_list)

        print('label', field_values_list)
        print('多边形个数', len(polygons_list))
        print('shp文件保存成功', self.shp_save_dir)

        # 统计field_values_list中各个类别的个数
        label_count = {}
        for label in field_values_list:
            if label[1] not in label_count:
                label_count[label[1]] = 1
            else:
                label_count[label[1]] += 1
        print('各个类别的个数', label_count)

    # 根据tfw_info对polygons_list进行转换，xy转化为经纬度坐标
    def polygons_coord_transform(self, polygons_list):
        # 获取tfw信息
        tfw_info = self.tfw_info
        # 获取左上角坐标
        x0, y0 = tfw_info['左上角x坐标'], tfw_info['左上角y坐标']
        # 获取分辨率
        x_res, y_res = tfw_info['x方向分辨率'], tfw_info['y方向分辨率']
        # 获取旋转参数
        x_rotate, y_rotate = tfw_info['x旋转参数'], tfw_info['y旋转参数']

        # 对polygons_list进行转换
        polygons_list = [[[xy[0] * x_res + x0, xy[1] * y_res + y0] for xy in polygon] for polygon in polygons_list]

        return polygons_list

    # 检测遥感图像，生成shp文件
    def detect(self):
        # 获取mask
        big_mask = self.seg_big_image()
        # 如果mask中的值不在label_to_chosens中，则将其置为0
        big_mask = np.where(np.isin(big_mask, self.label_to_chose), big_mask, 0)
        # 获取多边形列表
        polygons_list, category_ids_list = tuple(self.semantic_mask_to_points(big_mask, min_area_threshold=0).values())
        # 将多边形坐标转换为经纬度坐标
        polygons_list = self.polygons_coord_transform(polygons_list)
        # 将多边形列表转换为shp文件
        self.polygons2shp(polygons_list, category_ids_list)
        # 将原图和mask可视化
        plt.figure(figsize=(20, 20))
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.img_arr)
        # plt.subplot(1, 2, 2)
        self.draw_mask(big_mask,save=True)


# 读取遥感图像，做分割，转化为shp文件，主函数
def detect_remote_sense(img_path, label2id, shp_save_dir,use_logits_process=False, logits_alpha=1,**kwargs):
    seg_model, seg_processor = load_model_and_processor(kwargs['seg_model_path'], kwargs['seg_model_path'],
                                                        task_type='semantic segmentation')
    classify_model, classify_processor = load_model_and_processor(kwargs['classify_model_path'],
                                                                  kwargs['classify_model_path'],
                                                                  task_type='classification')

    seg_process = ClassificationAndSegmentationProcess(segmentation_model=seg_model,
                                                       classification_model=classify_model,
                                                       segmentation_processor=seg_processor,
                                                       classification_processor=classify_processor,
                                                       label2id=label2id,
                                                       use_logits_process=use_logits_process,
                                                       logits_alpha=logits_alpha)
    detect_image = DetectRemoteSense(img_path, label2id, shp_save_dir, seg_process, **kwargs)
    detect_image.detect()


# 读取模型和processor
def load_model_and_processor(model_path, processor_path, task_type='classification'):
    device_map = {'': 0} if torch.cuda.is_available() else {'': 'cpu'}
    #device_map= {'': 'cpu'}
    # 读取模型
    if task_type == 'classification':
        model = AutoModelForImageClassification.from_pretrained(model_path, torch_dtype=torch.float32,
                                                                device_map=device_map)
    elif task_type == 'semantic segmentation':
        model = AutoModelForSemanticSegmentation.from_pretrained(model_path, torch_dtype=torch.float32,
                                                                 device_map=device_map)
    else:
        raise Exception('task_type must be classification or semantic segmentation')
    # 读取processor
    processor = AutoProcessor.from_pretrained(processor_path)
    # 打印模型类型
    print('model type:', type(model))
    print(model.get_memory_footprint() / 1024 / 1024 / 1024, 'GB')
    return model, processor


if __name__ == '__main__':
    print(torch.cuda.is_available())
    label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '拼接痕迹': 4, '拼接错误': 5, '扭曲': 6, '模糊': 7,
                '光谱溢出': 8, '条状噪声': 9}
    label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '模糊': 4, '光谱溢出': 5, '扭曲': 6, '拼接痕迹': 7,
                '拼接错误': 8, '条状噪声': 9, '像素缺失': 10}

    seg_model_path = 'segformer_b5_6label'
    classify_model_path = 'swin_9label'
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--img_path', type=str, default='D:\Desktop\遥感原始素材\云/云3k.jpg')
    py_parser.add_argument('--label2chose', type=str, default=list(label2id.values()))
    py_parser.add_argument('--patch_size', type=str, default=[500, 1000])
    py_parser.add_argument('--overlap', type=str, default=None)

    # img_path = "D:\Desktop\遥感原始素材\云/云3k.jpg"
    # shp_save_dir='./shp_save_dir'
    # 从parser中读取参数
    args = py_parser.parse_args()
    img_path = args.img_path
    label2chose = args.label2chose
    #如果label2chose是字符串，则转换为列表
    if isinstance(label2chose, str):
        label2chose = eval(label2chose)
    #如果patch_size是字符串，则转换为列表
    if isinstance(args.patch_size, str):
        args.patch_size = eval(args.patch_size)
    patch_size = args.patch_size
    overlap = args.overlap
    print('img_path', img_path)
    #print('shp_save_dir', shp_save_dir)

    #如果img_path不是文件夹
    if not os.path.isdir(img_path):
        shp_save_dir = os.path.dirname(img_path)

        detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                            classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                            label_to_chose=label2chose,use_logits_process=True)
    #如果img_path是文件夹
    else:
        #读取文件夹下以及所有子文件夹里的所有图片路径，如果图片以".img"或".tif"结尾，则读取
        img_path_list = []
        for root, dirs, files in os.walk(img_path):
            for file in files:
                if file.endswith(".img") or file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg"):
                    img_path_list.append(os.path.join(root, file))


        for img_path in img_path_list:
            shp_save_dir = os.path.dirname(img_path)
            detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                                classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                                label_to_chose=label2chose,use_logits_process=True)
