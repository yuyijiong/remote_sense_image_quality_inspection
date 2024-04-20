from osgeo import gdal
from preprocess_shp_tif import Process_shp, BigImgProcess
import os
import pandas as pd
from multiprocess.pool import Pool
from typing import List, Dict, Tuple, Union

# 开始对栅格的操作
# GDAL所有操作都需要先注册格式
# 一次性注册所有的数据驱动，但是只能读不能写：gdal.AllRegister()
gdal.AllRegister()


# from 不使用.tif_shp_read import *

# 对一个文件夹操作，生成df
def get_df_from_dir(img_dir, max_edge=1000, error_type_name_list: Union[List[str], None] = None,
                    label2id: dict = None, test_only=False, save_all=0, main_label=None, nonzero_pixel_threshold=0.001,
                    add_label=None):
    shp_path, tfw_path, tif_path, img_path = None, None, None, None
    # 每个文件夹只有1个shp文件、tfw、tif文件，或img文件和shp文件
    # 遍历文件夹下所有文件，获取dir_path下的shp、tfw、tif文件路径
    for path in os.listdir(img_dir):
        if path.endswith('.shp'):
            shp_path = os.path.join(img_dir, path)
        elif path.endswith('.tif'):
            img_path = os.path.join(img_dir, path)
        elif path.endswith('.img'):
            img_path = os.path.join(img_dir, path)

    # 有一个文件找不到则报错
    if not shp_path:
        raise Exception(img_dir + '文件缺失')

    # 读取img文件
    pre_img = BigImgProcess(img_path, shp_path, train_mode=not test_only,label2id=label2id)

    # 读取shp文件并生成df
    if not test_only:
        pre_shp = Process_shp(shp_path, label2id=label2id, label_chose_list=error_type_name_list,
                              tfw_info=pre_img.tfw_info, img_shape=pre_img.img_shape)
        shp_datas = pre_shp.get_shp_shape_records(add_label=add_label)
    else:
        shp_datas = None

    #打印shp文件长度
    print('shp文件长度：',shp_datas)

    # 按小图切分shp
    shp_datas_split = pre_img.tif_to_small_tif(shp_datas=shp_datas, save_all=save_all, max_edge=max_edge,
                                               main_label=main_label,nonzero_pixel_threshold=nonzero_pixel_threshold)

    return shp_datas_split


# 遍历所有文件夹，拼接df
def get_df_from_all_dir(data_dir, **kwargs):
    dir_list = []  # 记录所有文件夹的绝对路径
    for dir in os.listdir(data_dir):
        # 如果是文件夹
        if os.path.isdir(os.path.join(data_dir, dir)) and not dir.startswith('.'):
            # 获取文件夹的绝对路径
            dir_path = os.path.join(data_dir, dir)
            dir_list.append(dir_path)

    # 将以上循环改为多进程，同时处理多个图片
    #pool=Pool(2)
    df_list = list(map(lambda x: get_df_from_dir(x, **kwargs), dir_list))

    # 删除df_list中的None
    df_list = [df for df in df_list if df is not None]

    # 拼接df
    df_all = pd.concat(df_list, axis=0)

    # 重置索引
    df_all.reset_index(drop=True, inplace=True)

    return df_all


def get_test_images(img_dir, max_edge_list: list = [1024]):
    data_dir = img_dir

    # 遍历data_dir下的所有文件夹
    test_only = True  # 是否只是test，没有标签

    # 遍历各种大小
    # pool=Pool(4)
    df_list = list(map(lambda x: get_df_from_dir(img_dir=data_dir,
                                                 label2id=None,
                                                 test_only=test_only,
                                                 max_edge=x,
                                                 error_type_name_list=None,
                                                 save_all=1), max_edge_list))
    # 拼接df_list
    df_all = pd.concat(df_list, axis=0)
    # 保存json
    save_path = 'img_datasets/{}_edge{}_{}_new/{}.json'.format(data_dir.split('/')[-1],
                                                               max_edge_list[0], max_edge_list[-1],
                                                               'train' if not test_only else 'test')
    print('save_path:', save_path, 'dataset length', len(df_all))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df_all.to_json(save_path, orient='records', lines=True, force_ascii=False)
    test_img_dataset_path = save_path

    return test_img_dataset_path


if __name__ == "__main__":
    data_dir = '../原始数据/阴影数据集_合'
    #label2id = {'背景':0,'地物变形': 1, '地物不完整': 2, '影像有拉花': 3, '影像模糊，纹理不清晰': 4}
    label2id={'背景':0,'云':1, '阴影':2,'拉花':3,'拼接痕迹':4,'拼接错误':5,'扭曲':6,'模糊':7,'光谱溢出':8,'条状噪声':9}

    # 遍历data_dir下的所有文件夹
    label_chose = list(label2id.keys())[0]
    label_chose = None
    test_only = False  # 是否只是test，没有标签

    # 遍历各种大小
    max_edge_list = [512,2048]

    # pool=Pool(4)
    df_list = list(map(lambda x: get_df_from_all_dir(data_dir=data_dir,
                                                     label2id=label2id,
                                                     test_only=test_only,
                                                     max_edge=x,
                                                     error_type_list=label_chose,
                                                     main_label=[8],
                                                     nonzero_pixel_threshold=0.001,
                                                     # add_label='阴影过大',
                                                     save_all=0), max_edge_list))

    # 拼接df_list
    df_all = pd.concat(df_list, axis=0)

    # 保存json
    save_path = '../img_datasets/{}_edge{}_{}_label{}_mask/{}.json'.format(data_dir.split('/')[-1],
                                                                        max_edge_list[0], max_edge_list[-1],
                                                                        label_chose if label_chose is not None else len(
                                                                            label2id),
                                                                        'train' if not test_only else 'test')
    print('save_path:', save_path, 'dataset length', len(df_all))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df_all.to_json(save_path, orient='records', lines=True, force_ascii=False)
