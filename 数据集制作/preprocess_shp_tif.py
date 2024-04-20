import copy
import math
import os
import random
from typing import List, Dict, Optional
import time

import albumentations
import numpy as np
import pandas as pd
import shapefile
import skimage.exposure
from osgeo import gdal
from osgeo import gdal_array as ga  # 用于引入一个模块的同时为该模块取一个别名
from osgeo.gdalconst import GA_ReadOnly
from skimage import io, draw
from skimage.transform import rescale
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
from shapely import ops
from PIL import Image

from Utils.Cv_Preprocess import Semantic_Segmentation_preprocess
from shp_write import write_shp
from Utils.Dataset_utils import MaskTransform, CategoryId2Caption


# 处理shp
class Process_shp:
    def __init__(self,
                 shp_path,
                 label2id: Dict[str, int] = None,
                 label_chose_list: List[str] = None,
                 tfw_info: dict = None,
                 img_shape: tuple = None):
        self.shp_path = shp_path
        self.label2id = None if label2id == [None] else label2id
        self.label_chose_list = label_chose_list
        self.tfw_info = tfw_info
        self.img_shape = img_shape

    # 将labelname转换为id，不需要完全匹配，只要labelname是label2id的key的子串就行
    @classmethod
    def labelname2id(cls, labelname, label2id: Dict[str, int]):
        # 如果某个label2id的key是labelname的子串，就返回对应的value。如果找不到，就报错
        for key in label2id.keys():
            if key in labelname:
                return label2id[key]
        raise Exception(f"找不到\'{labelname}\'对应的id")

    # 读取shp文件并获取bbox，最终返回df，每行是一个bbox(coco格式)，坐标为经纬度
    def get_shp_shape_records(self, add_label=None) -> pd.DataFrame:
        try:
            try:
                file = shapefile.Reader(self.shp_path)
                shape_records = file.shapeRecords()
            except UnicodeDecodeError:
                file = shapefile.Reader(self.shp_path, encoding="gbk")
                shape_records = file.shapeRecords()

            shp_types = []
            shp_bboxes = []
            shp_points = []
            shp_fields = []

            # 获取field名字
            fields = file.fields
            res = []
            for field in fields:
                res.append(field[0])

            # shape_record表示一个多边形,bbox是['min_lon', 'min_lat', 'max_lon', 'max_lat']
            for shape_record in shape_records:
                # type
                shp_type = shape_record.shape.shapeType
                if shp_type == 5:
                    shp_types.append('polygon')
                    points = shape_record.shape.points
                    bbox = shape_record.shape.bbox

                    # points是List[tuple],把points中的tuple拆开，
                    points = [i for point in points for i in point]

                    # 对36开头的数字进行处理
                    bbox = list(map(self.process_36_num, list(bbox)))
                    points = list(map(self.process_36_num, points))

                    field = shape_record.record
                    shp_bboxes.append(bbox)
                    shp_points.append(points)
                    shp_fields.append(field)

                else:
                    raise Exception("不是多边形")

            # 将bbox、points、fields,type转换为df，一共四列
            shp_df = pd.DataFrame({'bbox': shp_bboxes, 'points': shp_points, 'type': shp_types})
            # fields转换为df，每个field为一列
            shp_fields_df = pd.DataFrame(shp_fields, columns=res[1:])
            # 增加一列 错误描述，值为add_label
            if add_label:
                shp_fields_df['错误描述'] = add_label

            # 拆分fields，每个field为一列，列名为res[1:]
            shp_df = pd.concat([shp_df, shp_fields_df], axis=1)

            # 选择需要的label
            if self.label_chose_list:
                if isinstance(self.label_chose_list, str):
                    self.label_chose_list = [self.label_chose_list]

                shp_df = shp_df[shp_df['错误描述'].isin(self.label_chose_list)]
                # 如果没有选择的label，返回空
                if shp_df.empty:
                    return shp_df

            # 由错误描述生成category id
            if self.label2id:
                shp_df['category_id'] = shp_df['错误描述'].apply(lambda x: self.labelname2id(x, label2id=self.label2id))

            # 将bbox转换为像素坐标
            shp_df = self.get_bbox_pixel(shp_df)
            # 删除bbox或points为None的行
            shp_df = shp_df.dropna(subset=['bbox', 'points'])
            shp_df = shp_df.reset_index(drop=True)

            # bbox转换为[top-left corner x, top-left corner y, width, height]。符合coco
            shp_df['bbox'] = shp_df['bbox'].apply(lambda x: [x[0], x[1], x[2] - x[0], x[3] - x[1]])

            return shp_df

        except shapefile.ShapefileException as e:
            return repr(e)

    # 对36开头的数字进行处理
    @staticmethod
    def process_36_num(num: float, add=False):
        if not add:
            # 如果是36开头的数字，需要进行处理。转换为字符串，删除开头的36，再转换为float
            if num >= 36000000:
                num = float(str(num)[2:])
        else:
            # 如果数字小于36000000，需要进行处理。转换为字符串，开头增加36，再转换为float
            if num < 36000000:
                num = float('36' + str(num))
        return num

    # 将bbox的格式转换为像素坐标
    def get_bbox_pixel(self, shp_df: pd.DataFrame) -> pd.DataFrame:
        # 对bbox转换,经纬度转换为像素坐标[左上角x,左上角y,右下角x,右下角y]
        shp_df['bbox'] = shp_df['bbox'].apply(
            lambda x: self.lonlat_to_pixel(x, self.tfw_info, self.img_shape, do_normalize=False))
        # 对points转换,经纬度转换为像素坐标
        shp_df['points'] = shp_df['points'].apply(
            lambda x: self.lonlat_to_pixel(x, self.tfw_info, self.img_shape, do_normalize=False))

        return shp_df

    # 将单个bbox（或points）的单位转换为像素，[min_x,min_y,max_x,max_y]。注意min_x和min_y是左上角的坐标,min_lat对应max_y
    @staticmethod
    def lonlat_to_pixel(bbox: list, tfw_datas: dict, img_shape: Optional[tuple] = None, do_normalize: bool = False,
                        invert=False) -> list:

        # 获取tfw文件中的左上角坐标
        left_top_x = float(tfw_datas['左上角x坐标'])
        left_top_y = float(tfw_datas['左上角y坐标'])
        # 获取tfw文件中的x方向分辨率和y方向分辨率
        x_resolution = float(tfw_datas['x方向分辨率'])
        y_resolution = float(tfw_datas['y方向分辨率'])

        # 如果bbox长度为4，说明是[min_lon,min_lat,max_lon,max_lat]，转换为[min_lon,max_lat,max_lon,min_lat]
        # 因为经纬度坐标系和像素坐标系的y轴方向相反
        if len(bbox) == 4:
            bbox = [bbox[0], bbox[3], bbox[2], bbox[1]]

        if not invert:
            # 将bbox或points中偶数索引的坐标转换为像素坐标
            bbox[::2] = [round((lon - left_top_x) / x_resolution) for lon in bbox[::2]]
            bbox[1::2] = [round((lat - left_top_y) / y_resolution) for lat in bbox[1::2]]

            # 判断像素坐标是否超出图像范围
            if img_shape:
                if min(bbox) < 0 or max(bbox[::2]) > img_shape[1] or max(bbox[1::2]) > img_shape[0]:
                    print('bbox超出图像范围', img_shape)
                    return None

            # 归一化
            if img_shape and do_normalize:
                bbox[::2] = [x / img_shape[1] for x in bbox[::2]]
                bbox[1::2] = [y / img_shape[0] for y in bbox[1::2]]

        else:
            # 将像素坐标转换为经纬度坐标
            bbox[::2] = [round(lon * x_resolution + left_top_x, 6) for lon in bbox[::2]]
            bbox[1::2] = [round(lat * y_resolution + left_top_y, 6) for lat in bbox[1::2]]

        return bbox

    # 将单个bbox（或points）的由大图坐标转换为小图坐标，[min_x,min_y,max_x,max_y]。
    @staticmethod
    def coordi_big2small(bbox: list, top_left_x: int, top_left_y: int, invert=False) -> list:
        if not invert:
            # 由大图坐标转换为小图坐标
            bbox[::2] = [round(x - top_left_x) for x in bbox[::2]]
            bbox[1::2] = [round(y - top_left_y) for y in bbox[1::2]]

        else:
            # 由小图坐标转换为大图坐标
            bbox[::2] = [round(x + top_left_x) for x in bbox[::2]]
            bbox[1::2] = [round(y + top_left_y) for y in bbox[1::2]]

        return bbox


# 处理大图片
class BigImgProcess(Process_shp, MaskTransform, CategoryId2Caption):
    def __init__(self, img_path, shp_path=None, train_mode=True, label2id=None):
        super().__init__(shp_path, label2id=label2id)

        self.img_path = img_path

        # 获取图片信息
        # 如果img_path以jpg或png结尾，则使用PIL读取图片并转换为array
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img_arr = np.array(Image.open(img_path))
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
            # 对tfw_values进行36开头的转换
            tfw_values = [self.process_36_num(v) for v in tfw_values]

        tfw_keys = ['左上角x坐标', 'x方向分辨率', 'x旋转参数', '左上角y坐标', 'y旋转参数', 'y方向分辨率']
        self.tfw_info = dict(zip(tfw_keys, tfw_values))

        # 从int32转换为uint8
        img_arr = img_arr.astype(np.uint8)
        self.img_arr = img_arr

        # 获取图片shape
        self.img_shape = img_arr.shape

        # 是否为训练模式
        self.train_mode = train_mode

    # 获取shp_datas中所有points和category_id，返回两个列表
    @classmethod
    def get_points_category_id(cls, shp_datas: Optional[pd.DataFrame] = None):
        points_all = []
        category_id_all = []
        for points, category_id in zip(shp_datas['points'], shp_datas['category_id']):
            points_all.append(points)
            category_id_all.append(category_id)

        return points_all, category_id_all

    def tif_to_small_tif(self, **kwargs):
        # 训练模式只需要mask
        if self.train_mode:
            print('train mode')
            return self.tif_to_small_tif_mask(**kwargs)
        # 测试模式需要坐标
        else:
            print('test mode')
            return self.tif_to_small_tif_points(**kwargs)

    # 将大图切分为小图，同时也bbox\points需要转换为小图中的像素坐标
    def tif_to_small_tif_points(self, shp_datas: Optional[pd.DataFrame] = None, save_all=0,
                                split_num=2, max_edge=None, save_max_edge=1024):
        r"""
        :param shp_datas: 记录bbox、points、fields,type的df，get_shp_shape_records函数的返回值，每行代表一个bbox，单位为像素坐标
        :param save_all: 保存无标签图的概率
        :param split_num: 每个小图的边长不超过max_edge，如果max_edge为None，则split_num为切分的个数
        :param max_edge: 每个小图的边长不超过max_edge，如果max_edge为None，则split_num为切分的个数

        :return: 一个df，包含小图的路径、bbox、points、fields,type，每行对应一个小图
        """

        # 计算切分个数，使每个小图的边长不超过max_edge
        if max_edge is not None:
            split_num = math.ceil(max(self.img_shape[0], self.img_shape[1]) / max_edge)

        print('图片', self.img_path, '将被切分为', split_num, '* ', split_num, '个小图', '(每个小图的边长不超过',
              max_edge, ')')

        # 将tif等分的切分为2*2个小图，每个小图长宽比和原图相同，不一定是正方形
        indices_0 = np.arange(0, self.img_shape[0], self.img_shape[0] // split_num)
        indices_1 = np.arange(0, self.img_shape[1], self.img_shape[1] // split_num)
        # 确保indices0元素个数为split_num +1 个
        if len(indices_0) < split_num + 1:
            indices_0 = np.append(indices_0, self.img_shape[0])
        # 确保indices0最后一个元素为self.img_shape[0]
        indices_0[-1] = self.img_shape[0]
        # 确保indices1元素个数为split_num +1 个
        if len(indices_1) < split_num + 1:
            indices_1 = np.append(indices_1, self.img_shape[1])
        # 确保indices1最后一个元素为self.img_shape[1]
        indices_1[-1] = self.img_shape[1]

        # 得到每个小图的crop参数即[min_x,min_y,max_x,max_y]
        small_tifs_crop = [[indices_1[j], indices_0[i], indices_1[j + 1], indices_0[i + 1]] for i in
                           range(len(indices_0) - 1) for j in range(len(indices_1) - 1)]

        # 生成一个新的df，具有以下列
        #  ['subfig_path', 'bbox', 'points', 'subfig_id',
        # 'index_0', 'index_1', '错误描述', 'area',
        # 'subfig_save_shape', 'target_size', 'top_left', 'top_left_lat_lon',
        # 'fig_path', 'shp_path', 'tfw_data', 'image_id', 'category_id']
        df_new = pd.DataFrame(columns=['subfig_path', 'bbox', 'points', 'area', 'subfig_id',
                                       'index_0', 'index_1',
                                       'bbox_subfig_save_shape', 'target_size', 'top_left', 'top_left_lat_lon',
                                       'fig_path', 'shp_path', 'tfw_data', 'image_id', '错误描述', 'category_id'])

        # 一个个小图进行处理，并逐行添加到df_new中
        for i in range(len(small_tifs_crop)):
            # 根据crop参数切分大图以及bbox
            tranforms = albumentations.Compose([
                albumentations.Crop(*small_tifs_crop[i]),
            ], bbox_params=albumentations.BboxParams(format='coco', label_fields=['category_id']))

            out = tranforms(image=self.img_arr, bboxes=list(shp_datas['bbox'].values) if shp_datas is not None else [],
                            category_id=list(shp_datas.index) if shp_datas is not None else [])

            small_tif = out['image']

            random_value = random.random()  # 每个小图有一个随机值
            # 如果bbox为空，则有save_all的几率跳过
            if len(out['bboxes']) == 0:
                if random_value > save_all:
                    continue

            # 如果小图是低对比度，则跳过
            if skimage.exposure.is_low_contrast(small_tif):
                continue

            # 若shp_datas为空，则需要将shp_belong_this置为空
            if shp_datas is None:
                shp_belong_this = pd.DataFrame(columns=['subfig_path', 'bbox', 'points', 'area', 'subfig_id',
                                                        'index_0', 'index_1',
                                                        'bbox_subfig_save_shape', 'target_size', 'top_left',
                                                        'top_left_lat_lon',
                                                        'fig_path', 'shp_path', 'tfw_data', 'image_id', '错误描述',
                                                        'category_id'])

            else:
                # 筛选shp_datas中index为out['category_id']的行，即属于该小图的shp。一定不为空，需要深拷贝，防止修改原来的shp_datas
                shp_belong_this = copy.deepcopy(shp_datas.loc[out['category_id']])

            # 将shp_belong_this中的bbox转换为相对于小图的坐标
            shp_belong_this['bbox'] = out['bboxes']

            # 将shp_belong_this中的points转换为相对于小图的坐标
            shp_belong_this['points'] = shp_belong_this['points'].apply(
                lambda x: self.coordi_big2small(x, *small_tifs_crop[i][0:2]))

            # 清洗shp_belong_this，有可能返回空df
            shp_belong_this = self.clean_small_tif_df(shp_belong_this, small_tif.shape)

            # 如果shp_belong_this清洗后为空，则跳过
            if len(shp_belong_this) == 0 and not save_all:
                if random_value > save_all:
                    continue

            # 指定保存路径
            small_tif_path = os.path.dirname(self.img_path) + "/blocks{}_edge{}/{}.jpg".format(split_num, max_edge, i)
            # small_tif_path=small_tif_path.replace('D:/', 'C:/')
            if not os.path.exists(os.path.dirname(small_tif_path)):
                os.makedirs(os.path.dirname(small_tif_path))

            # 对图片rescale，使长边不大于2048
            rescale_max_edge = save_max_edge
            if max(small_tif.shape) > rescale_max_edge:
                rescale_factor = rescale_max_edge / max(small_tif.shape)
                small_tif_rescale = rescale(small_tif, rescale_factor, mode='constant', channel_axis=-1)
                small_tif_rescale = (small_tif_rescale * 255.0).astype('uint8')

                # 对应的bbox\points也要rescale，即缩小
                try:
                    if len(shp_belong_this) != 0:
                        shp_belong_this['bbox'] = shp_belong_this['bbox'].apply(
                            lambda x: [int(v * rescale_factor) for v in x])
                        shp_belong_this['points'] = shp_belong_this['points'].apply(
                            lambda x: [int(v * rescale_factor) for v in x])
                        shp_belong_this['area'] = shp_belong_this['area'].apply(
                            lambda x: int(x * rescale_factor * rescale_factor))

                except:
                    print('没有bbox，不用rescale bbox，只rescale图片')

                io.imsave(small_tif_path, small_tif_rescale)
            else:
                io.imsave(small_tif_path, small_tif)
                small_tif_rescale = small_tif

            # 获取top_left以及经纬度形式的top_left
            top_left = small_tifs_crop[i][1::-1]
            top_left_lat_lon = self.top_left_to_lon_lat(top_left, self.tfw_info)

            # df_new增加一行
            if len(shp_belong_this) != 0:
                df_new.loc[i] = copy.deepcopy(
                    [small_tif_path, list(shp_belong_this['bbox']), list(shp_belong_this['points']),
                     list(shp_belong_this['area']),
                     i, i // split_num, i % split_num, small_tif_rescale.shape, small_tif.shape,
                     top_left, top_left_lat_lon, self.img_path, self.shp_path, self.tfw_info, i,
                     shp_belong_this['错误描述'], list(shp_belong_this['category_id'])])
            else:
                df_new.loc[i] = copy.deepcopy(
                    [small_tif_path, list(shp_belong_this['bbox']), [], [], i, i // split_num, i % split_num,
                     small_tif_rescale.shape, small_tif.shape, top_left, top_left_lat_lon,
                     self.img_path, self.shp_path, self.tfw_info, i, [], []])

        return df_new

    # 将大图切分为小图，现在大图中画出整个mask再切分，这样可以保证小图中有mask
    def tif_to_small_tif_mask(self, shp_datas: Optional[pd.DataFrame] = None, mask_whole: Optional[np.ndarray] = None,
                              split_num=2, max_edge=None, save_max_edge=1024, save_all=0, main_label: list = None,
                              nonzero_pixel_threshold=0.001, **kwargs):
        r"""
        :param shp_datas: 记录bbox、points、fields,type的df，get_shp_shape_records函数的返回值，每行代表一个bbox，单位为像素坐标
        :param save_all: 保存无标签图的概率
        :param split_num: 每个小图的边长不超过max_edge，如果max_edge为None，则split_num为切分的个数
        :param max_edge: 每个小图的边长不超过max_edge，如果max_edge为None，则split_num为切分的个数

        :return: 一个df，包含小图的路径、bbox、points、fields,type，每行对应一个小图
        """

        # 计算切分个数，使每个小图的边长不超过max_edge
        if max_edge is not None:
            split_num = math.ceil(max(self.img_shape[0], self.img_shape[1]) / max_edge)

        print('图片', self.img_path, '将被切分为', split_num, '* ', split_num, '个小图', '(每个小图的边长不超过',
              max_edge, ')')

        # 将tif等分的切分为2*2个小图，每个小图长宽比和原图相同，不一定是正方形
        indices_0 = np.arange(0, self.img_shape[0], self.img_shape[0] // split_num)
        indices_1 = np.arange(0, self.img_shape[1], self.img_shape[1] // split_num)
        # 确保indices0元素个数为split_num +1 个
        if len(indices_0) < split_num + 1:
            indices_0 = np.append(indices_0, self.img_shape[0])
        # 确保indices0最后一个元素为self.img_shape[0]
        indices_0[-1] = self.img_shape[0]
        # 确保indices1元素个数为split_num +1 个
        if len(indices_1) < split_num + 1:
            indices_1 = np.append(indices_1, self.img_shape[1])
        # 确保indices1最后一个元素为self.img_shape[1]
        indices_1[-1] = self.img_shape[1]

        # 得到每个小图的crop参数即[min_x,min_y,max_x,max_y]
        small_tifs_crop = [[indices_1[j], indices_0[i], indices_1[j + 1], indices_0[i + 1]] for i in
                           range(len(indices_0) - 1) for j in range(len(indices_1) - 1)]

        # 生成一个新的df，具有以下列
        df_new = pd.DataFrame(columns=['image', 'mask', 'caption', 'cate_ids'])

        # 如果mask_whole为None，则生成mask_whole
        if mask_whole is None:
            # 如果shp_datas长度为0，则直接返回
            if len(shp_datas) == 0:
                return df_new

            # 获取所有的points和category_id
            points_all, category_id_all = self.get_points_category_id(shp_datas)
            # 获取不同的category_id
            category_id_unique = np.unique(category_id_all).tolist()
            # 转化为字符串
            category_id_unique = ''.join([str(i) for i in category_id_unique])
            # 生成mask
            mask_whole = Semantic_Segmentation_preprocess.points_to_semantic_mask(
                points_all, self.img_arr.shape, category_id_all, background_as_class0=True)
        else:
            # 获取不同的category_id
            category_id_unique = np.unique(mask_whole).tolist()
            # 转化为字符串
            category_id_unique = ''.join([str(i) for i in category_id_unique])

        # 一个个小图进行处理，并逐行添加到df_new中
        for i in tqdm(range(len(small_tifs_crop)), desc='切分小图', total=len(small_tifs_crop)):
            # 根据crop参数切分大图以及bbox
            tranforms = albumentations.Compose([
                albumentations.Crop(*small_tifs_crop[i]),
            ])

            out = tranforms(image=self.img_arr, mask=mask_whole)

            small_tif = out['image']
            small_mask = out['mask']
            # 确保两者大小相同
            assert small_tif.shape[0:2] == small_mask.shape[0:2], 'small_tif.shape!=small_mask.shape' + str(
                small_tif.shape) + str(small_mask.shape)

            # 转换为uint8
            small_tif = small_tif.astype(np.uint8)
            small_mask = small_mask.astype(np.uint8)

            # 如果小图是低对比度，则跳过
            if skimage.exposure.is_low_contrast(small_tif):
                continue

            if main_label is None:
                # 如果small_mask中的非零元素个数占比小于0.1，则跳过
                if np.count_nonzero(small_mask) / small_mask.size < nonzero_pixel_threshold and save_all == 0:
                    continue
            else:
                # 如果small_mask中的等于main_label的元素个数占比小于0.1，则跳过。main_label，需要对等于main_label的元素进行计数
                if sum([np.count_nonzero(small_mask == i) for i in
                        main_label]) / small_mask.size < nonzero_pixel_threshold and save_all == 0:
                    continue

            # 获取此mask中的cate_id 并计算每种category_id_unique的面积占比
            cate_ids, cate_area = MaskTransform.mask_to_category_id(small_mask)
            cate_ids_str = str(cate_ids).replace(',', '').replace('[', '').replace(']', '').replace(' ', '')

            # 生成caption
            caption_text = self.category_id_to_caption(cate_ids,
                                                       self.label2id,
                                                       background_as_class0=True,
                                                       category_area=cate_area)
            #获取self.img_path的文件名
            img_name = os.path.basename(self.img_path)
            #去掉后缀
            img_name = os.path.splitext(img_name)[0]
            # 指定保存路径，i补到4位
            small_tif_path = os.path.dirname(self.img_path) + \
                             "/{}_blocks{}_edge{}_label{}_rgb/{}.png".format(img_name,split_num, max_edge, cate_ids_str,
                                                                          str(i).zfill(4))
            # small_tif_path转化为绝对路径
            small_tif_path = os.path.abspath(small_tif_path)

            # small_tif_path=small_tif_path.replace('D:/', 'C:/')
            if not os.path.exists(os.path.dirname(small_tif_path)):
                os.makedirs(os.path.dirname(small_tif_path))

            # 指定mask保存路径
            small_mask_path = os.path.dirname(self.img_path) + \
                              "/{}_blocks{}_edge{}_label{}_mask/{}.png".format(img_name,split_num, max_edge, cate_ids_str,
                                                                            str(i).zfill(4))

            # small_mask_path转化为绝对路径
            small_mask_path = os.path.abspath(small_mask_path)

            # small_mask_path=small_mask_path.replace('D:/', 'C:/')
            if not os.path.exists(os.path.dirname(small_mask_path)):
                os.makedirs(os.path.dirname(small_mask_path))

            # 对图片rescale，使长边不大于1024
            rescale_max_edge = save_max_edge
            if max(small_tif.shape) > rescale_max_edge:
                rescale_factor = rescale_max_edge / max(small_tif.shape)
                small_tif_rescale = rescale(small_tif, rescale_factor, mode='constant', channel_axis=-1)
                small_tif_rescale = (small_tif_rescale * 255.0).astype('uint8')
                # 对应的mask也要rescale
                # small_mask增加一个维度
                small_mask = np.expand_dims(small_mask, axis=-1)
                small_mask_rescale = rescale(small_mask, rescale_factor, mode='constant', channel_axis=-1)
                small_mask_rescale = (small_mask_rescale * 255.0).astype('uint8')
                # 去掉增加的维度
                small_mask_rescale = np.squeeze(small_mask_rescale, axis=-1)
            else:
                small_tif_rescale = small_tif
                small_mask_rescale = small_mask

            io.imsave(small_tif_path, small_tif_rescale)
            io.imsave(small_mask_path, small_mask_rescale)

            # df_new增加一行
            df_new = pd.concat([df_new, pd.DataFrame({'image': [small_tif_path], 'mask': [small_mask_path],
                                                      'caption': [caption_text],
                                                      'cate_ids': [cate_ids]})], ignore_index=True)
        return df_new

    # 将df中的左上角坐标转换为经纬度
    @staticmethod
    def top_left_to_lon_lat(top_left: list, tfw_datas: dict):
        l = [top_left[0] * tfw_datas['y方向分辨率'] + tfw_datas['左上角y坐标'],
             top_left[1] * tfw_datas['x方向分辨率'] + tfw_datas['左上角x坐标']]
        return l

    # 计算多边形面积
    @staticmethod
    def polygon_area(polygon):
        """
        compute polygon area
        polygon: list with shape [n, 2], n is the number of polygon points
        """
        area = 0
        q = polygon[-1]
        for p in polygon:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return abs(area) / 2.0

    # 计算多边形面积
    @staticmethod
    def polygon_area_belong(polygon, img_shape):
        # 画出属于此小图的mask
        mask = skimage.draw.polygon2mask(img_shape[:2], polygon)
        # 计算mask面积
        area = np.sum(mask)
        return area

    # 对某个小图所属df的bbox、points进行清洗筛选
    def clean_small_tif_df(self,
                           shp_belong_this: pd.DataFrame,
                           small_tif_shape: list,
                           min_area=0.01,
                           max_area=0.95,
                           polygon_in_bbox_min_area=0.5):
        # 若长度为0，则直接返回
        if len(shp_belong_this) == 0:
            return shp_belong_this

        # 计算属于此区域的多边形面积
        shp_belong_this['area'] = shp_belong_this['points'].apply(
            lambda x: self.polygon_area_belong(np.array(x).reshape(-1, 2), small_tif_shape))

        # 筛选面积大于min_area*small_tif.shape[0]*small_tif.shape[1]的
        shp_belong_this = shp_belong_this[shp_belong_this['area'] >= min_area * small_tif_shape[0] * small_tif_shape[1]]
        # 若存在面积大于max_area*small_tif.shape[0]*small_tif.shape[1]的，则删除所有行
        if len(shp_belong_this[shp_belong_this['area'] >= max_area * small_tif_shape[0] * small_tif_shape[1]]) != 0:
            return shp_belong_this[0:0]

        # 若存在面积小于bbox面积的，则删除所有行
        if len(shp_belong_this[shp_belong_this['area'] <= shp_belong_this['bbox'].apply(
                lambda x: x[2] * x[3] * polygon_in_bbox_min_area)]) != 0:
            return shp_belong_this[0:0]

        return shp_belong_this


# 处理多个波段的遥感数据
class Cloud_Image_Process():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # 提取data_dir中以B30.TIF结尾的文件名
        self.b30 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('B30.TIF')][0]
        self.b20 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('B20.TIF')][0]
        self.b10 = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('B10.TIF')][0]
        # 合成rgb图像
        self.rgb = np.stack([io.imread(self.b30), io.imread(self.b20), io.imread(self.b10)], axis=2).astype(np.uint8)
        # 提取包含mask的TIF文件
        self.mask = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'mask' in f and '.TIF' in f][0]
        self.mask = io.imread(self.mask).astype(np.uint8)
        if self.mask[0, 0] != 0:
            self.mask = 255 - self.mask

        self.img_shape = self.rgb.shape

    # 将大图切分为小图，同时也bbox\points需要转换为小图中的像素坐标
    def tif_split_train(self, split_size=512):
        self.split_size = split_size

        # 删除rgb图边缘的一部分，使其能被split_size整除。需要删除两边
        self.rgb = self.rgb[
                   self.rgb.shape[0] // 2 - self.rgb.shape[0] // split_size * split_size // 2:self.rgb.shape[0] // 2 +
                                                                                              self.rgb.shape[
                                                                                                  0] // split_size * split_size // 2,
                   self.rgb.shape[1] // 2 - self.rgb.shape[1] // split_size * split_size // 2:self.rgb.shape[1] // 2 +
                                                                                              self.rgb.shape[
                                                                                                  1] // split_size * split_size // 2,
                   :]
        self.mask = self.mask[
                    self.mask.shape[0] // 2 - self.mask.shape[0] // split_size * split_size // 2:self.mask.shape[
                                                                                                     0] // 2 +
                                                                                                 self.mask.shape[
                                                                                                     0] // split_size * split_size // 2,
                    self.mask.shape[1] // 2 - self.mask.shape[1] // split_size * split_size // 2:self.mask.shape[
                                                                                                     1] // 2 +
                                                                                                 self.mask.shape[
                                                                                                     1] // split_size * split_size // 2]
        self.img_shape = self.rgb.shape

        # 计算切分个数，使每个小图的边长不超过max_edge
        split_num = math.ceil(max(self.img_shape[0], self.img_shape[1]) / split_size)
        print('图片', self.data_dir, '将被切分为', split_num, '* ', split_num, '个小图', '(每个小图的边长不超过',
              split_size, ')')

        # 将tif等分的切分为2*2个小图，每个小图长宽比和原图相同，不一定是正方形
        indices_0 = np.arange(0, self.img_shape[0], split_size)
        indices_1 = np.arange(0, self.img_shape[1], split_size)
        # 确保indices0元素个数为split_num +1 个
        if len(indices_0) < split_num + 1:
            indices_0 = np.append(indices_0, self.img_shape[0])
        # 确保indices0最后一个元素为self.img_shape[0]
        indices_0[-1] = self.img_shape[0]
        # 确保indices1元素个数为split_num +1 个
        if len(indices_1) < split_num + 1:
            indices_1 = np.append(indices_1, self.img_shape[1])
        # 确保indices1最后一个元素为self.img_shape[1]
        indices_1[-1] = self.img_shape[1]

        # 得到每个小图的crop参数即[min_x,min_y,max_x,max_y]
        small_tifs_crop = [[indices_1[j], indices_0[i], indices_1[j + 1], indices_0[i + 1]] for i in
                           range(len(indices_0) - 1) for j in range(len(indices_1) - 1)]

        # 记录小图保存路径
        rgb_save_path = []
        mask_save_path = []
        # 一个个小图进行处理，并逐行添加到df_new中
        for i in tqdm(range(len(small_tifs_crop))):
            # 根据crop参数切分大图以及bbox
            tranforms = albumentations.Compose([
                albumentations.Crop(*small_tifs_crop[i]),
            ])

            out = tranforms(image=self.rgb, mask=self.mask)
            small_tif_rgb = out['image']
            small_tif_mask = out['mask']

            # 如果小图是低对比度，则跳过
            if skimage.exposure.is_low_contrast(small_tif_rgb, 0.1) or skimage.exposure.is_low_contrast(small_tif_mask,
                                                                                                        0.1):
                continue
            # 如果mask只有一个元素，则跳过
            if len(np.unique(small_tif_mask)) < 2:
                continue

            # 指定保存路径
            small_tif_rgb_path = self.data_dir + "/blocks{}_rgb/{}.jpg".format(split_size, str.zfill(str(i), 4))
            small_tif_mask_path = self.data_dir + "/blocks{}_mask/{}.jpg".format(split_size, str.zfill(str(i), 4))

            # small_tif_path=small_tif_path.replace('D:/', 'C:/')
            if not os.path.exists(os.path.dirname(small_tif_rgb_path)):
                os.makedirs(os.path.dirname(small_tif_rgb_path))
            if not os.path.exists(os.path.dirname(small_tif_mask_path)):
                os.makedirs(os.path.dirname(small_tif_mask_path))

            # 保存小图
            io.imsave(small_tif_rgb_path, small_tif_rgb)
            io.imsave(small_tif_mask_path, small_tif_mask)

            # 记录小图保存路径
            rgb_save_path.append(small_tif_rgb_path)
            mask_save_path.append(small_tif_mask_path)

        return rgb_save_path, mask_save_path


# 将预测结果转换为shp
class Pred2Shp:
    def __init__(self, pred_df_dir, shp_dir, id2label, fig_path=None):
        self.pred_df = pd.read_json(pred_df_dir, lines=True)
        # 当前时间转换为字符串，精确到分钟,用-连接
        time_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.shp_dir = shp_dir + time_str
        self.id2label = id2label

        # 如果fig_path不为空，则筛选pred_df中fig_path等于fig_path的样本
        if fig_path is not None:
            self.pred_df = self.pred_df[[os.path.samefile(path, fig_path) for path in self.pred_df['fig_path']]]

    # 对xy格式的points进行放缩
    def scale_points(self, polygons, x_scale, y_scale):
        for polygon in polygons:
            for point in polygon:
                point[0] = point[0] * x_scale
                point[1] = point[1] * y_scale

        return polygons

    # 对xy格式的points进行加减
    def add_points(self, polygons, x_add, y_add):
        for polygon in polygons:
            for point in polygon:
                point[0] = point[0] + x_add
                point[1] = point[1] + y_add

        return polygons

    # 对polygons进行多边形合并
    def union_polygons(self, polygons, category_ids):
        # 如果polygons为空，则直接返回
        if len(polygons) == 0:
            return polygons, category_ids

        # 将polygons转换为shapely格式
        polygons_shapely = [Polygon(polygon) for polygon in polygons]
        polygons_shapely_new = {}
        # 对同一个category_id的polygon求并集，求并集后可能会出现MultiPolygon的情况，需要将其转换为Polygon
        for category_id in set(category_ids):
            # 找到category_id相同的polygon
            polygons_same_category_id = [polygons_shapely[i] for i in range(len(polygons_shapely)) if
                                         category_ids[i] == category_id]
            # 求并集
            union = ops.unary_union(polygons_same_category_id)
            # 将MultiPolygon转换为Polygon
            if type(union) == MultiPolygon:
                polygons_same_category_id = [polygon for polygon in union.geoms]
            else:
                polygons_same_category_id = [union]

            polygons_shapely_new[category_id] = polygons_same_category_id

        # 将polygons_shapely_new转换为list
        polygons_shapely = []
        category_ids = []
        for category_id in polygons_shapely_new:
            polygons_shapely.extend(polygons_shapely_new[category_id])
            category_ids.extend([category_id] * len(polygons_shapely_new[category_id]))

        # 将shapely格式的polygons转换为xy格式
        polygons_nms = [np.array(polygon.exterior.coords).tolist() for polygon in polygons_shapely]

        # 打印
        print('nms前：', len(polygons)), print('nms后：', len(polygons_nms))
        return polygons_nms, category_ids

    # 确保多边形头尾相连
    def ensure_close_polygon(self, polygon):
        # 如果polygon的第一个点和最后一个点不相同，则将第一个点添加到最后
        if polygon[0][0] != polygon[-1][0] or polygon[0][1] != polygon[-1][1]:
            polygon.append(polygon[0])
        return polygon

    # 对多边形进行近似
    def approximate_polygon(self, polygon, epsilon=0.1):
        # 确保多边形头尾相连
        polygon = self.ensure_close_polygon(polygon)
        # 将polygon转换为shapely格式
        polygon_shapely = Polygon(polygon)
        # 对polygon进行近似
        polygon_shapely = polygon_shapely.simplify(epsilon, preserve_topology=True)
        # 确保多边形有效
        if not polygon_shapely.is_valid:
            polygon_shapely = polygon_shapely.buffer(0)
        # 如果polygon_shapely是多个多边形，则取最大的一个
        if polygon_shapely.geom_type == 'MultiPolygon':
            polygon_shapely = max(polygon_shapely.geoms, key=lambda x: x.area)

        # 将polygon转换为list格式
        polygon = np.array(polygon_shapely.exterior.coords).tolist()

        # 如果polygon的点数小于3，则将其删除
        if len(polygon) < 3:
            return []
        return polygon

    # 将预测结果的polygon转换为经纬度
    def polygon_transform(self):
        # 记录新列
        polygons_pred = []
        category_ids_pred = []
        # 遍历每个样本
        for i in tqdm(range(len(self.pred_df))):
            # 获取样本的预测结果
            pred_result = self.pred_df.iloc[i]['pred_result']
            # 获取结果中的points和category_ids。注意，points是xy格式
            polygons = pred_result['points']
            category_ids = pred_result['category_ids']
            # 根据bbox_subfig_save_shape和target_size计算points的缩放比例
            y_scale = self.pred_df.iloc[i]['target_size'][0] / self.pred_df.iloc[i]['bbox_subfig_save_shape'][0]
            x_scale = self.pred_df.iloc[i]['target_size'][1] / self.pred_df.iloc[i]['bbox_subfig_save_shape'][1]
            polygons = self.scale_points(polygons, x_scale, y_scale)
            # polygons由小图坐标转换为大图坐标
            polygons = self.add_points(polygons, self.pred_df.iloc[i]['top_left'][1],
                                       self.pred_df.iloc[i]['top_left'][0])
            # polygons由大图坐标转换为经纬度坐标
            x_resulotion = self.pred_df.iloc[i]['tfw_data']['x方向分辨率']
            y_resulotion = self.pred_df.iloc[i]['tfw_data']['y方向分辨率']
            polygons = self.scale_points(polygons, x_resulotion, y_resulotion)
            top_left_x = self.pred_df.iloc[i]['tfw_data']['左上角x坐标']
            top_left_y = self.pred_df.iloc[i]['tfw_data']['左上角y坐标']
            polygons = self.add_points(polygons, top_left_x, top_left_y)
            # 记录到df
            polygons_pred.append(polygons)
            category_ids_pred.append(category_ids)

        self.pred_df['polygons_pred'] = polygons_pred
        self.pred_df['category_ids_pred'] = category_ids_pred

    # 将预测结果转换为shp
    def pred2shp(self):
        # 设置shp属性格式
        field_names = ['scores', 'labels']
        field_types = ['N', 'C']
        field_sizes = [50, 50]
        field_decimals = [2, 0]
        shape_type = 5  # 5代表polygon

        # 获取pred_polygons
        polygons_list = self.pred_df['polygons_pred'].values.tolist()
        # 获取每个多边形的属性信息
        category_ids_list = self.pred_df['category_ids_pred'].values.tolist()
        # 拼接list内部的list
        polygons_list = [polygon for polygons_per_sample in polygons_list for polygon in polygons_per_sample]
        category_ids_list = [category_id for category_ids_per_sample in category_ids_list for category_id in
                             category_ids_per_sample]
        # 删除无效的多边形和对应的category_id
        not_keep = [i for i in range(len(polygons_list)) if len(polygons_list[i]) < 3]
        polygons_list = [polygon for i, polygon in enumerate(polygons_list) if i not in not_keep]
        category_ids_list = [category_id for i, category_id in enumerate(category_ids_list) if i not in not_keep]

        # 进行nms
        polygons_list, category_ids_list = self.union_polygons(polygons_list, category_ids_list)

        # 每个polygon外面套一个list
        polygons_list = [[polygon] for polygon in polygons_list]
        # 增加scores列，score默认为1,并把label转换为中文
        field_values_list = [[1, self.id2label[category_id + 1]] for category_id in category_ids_list]
        print('score,label', field_values_list)
        print('多边形个数', len(polygons_list))

        # self.shp_dir不存在则创建
        if not os.path.exists(self.shp_dir):
            os.makedirs(self.shp_dir)

        write_shp(self.shp_dir + '/pred_polygons', shape_type, polygons_list,
                  field_names, field_types, field_sizes, field_decimals, field_values_list)
        print('shp文件保存成功', self.shp_dir)


if __name__ == '__main__':
    # #读取数据
    # df=pd.read_json('img_datasets/shp_tif_data_edge512_8192_label4_new/all.json',lines=True)
    # #检查目标检测数据集
    # #check_dataset_detection(df,max_index=3)
    # #检查全景分割数据集
    # selected_df=check_dataset_panorama(df,max_index=10,pred=True)
    #
    # #保存筛选后的数据
    # selected_df.to_json('img_datasets/shp_tif_data_edge512_8192_label4_new/train_selected.json',lines=True,orient='records')
    #
    # #读取txt，若某行有‘1’，则记录行号
    # with open('img_datasets/shp_tif_data_edge512_8192_label4_new/train_selected.txt','r') as f:
    #     lines=f.readlines()
    #     select_list=[]
    #     for i,line in enumerate(lines):
    #         if '1' in line:
    #             select_list.append(i)

    # 将云数据集切分为小图和mask
    dir_all = 'E:\云遮挡数据集_landsat7'
    # 记录所有小图保存路径
    rgb_save_path_all = []
    mask_save_path_all = []

    # 遍历dir_all下所有文件夹名
    for dir in os.listdir(dir_all):
        dir_path = os.path.join(dir_all, dir)
        # 判断是否为文件夹
        if os.path.isdir(dir_path):
            cloud_process = Cloud_Image_Process(dir_path)
            rgb_save_path, mask_save_path = cloud_process.tif_split_train(split_size=1024)
            rgb_save_path_all.extend(rgb_save_path)
            mask_save_path_all.extend(mask_save_path)

    # 将rgb和mask的路径转换为df
    df = pd.DataFrame({'rgb_path': rgb_save_path_all, 'mask_path': mask_save_path_all})
    # 保存df为json
    df.to_json('img_datasets/cloud_latsat7/all.json', lines=True, orient='records')
