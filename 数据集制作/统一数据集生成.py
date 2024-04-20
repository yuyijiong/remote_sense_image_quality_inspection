import pandas as pd
from PIL import Image
import numpy as np
import os
from shutil import copyfile

from Utils.Dataset_utils import MaskTransform,replace_dot,CategoryId2Caption

label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '模糊': 4, '光谱溢出': 5, '扭曲': 6, '拼接痕迹': 7, '拼接错误': 8,
            '条状噪声': 9}


# 读取一系列df，并把格式转化为image、mask、caption、cate_ids
def get_df_from_df_path_list(df_path_list, **kwargs):
    df_list = [df_to_img_mask_caption(replace_dot(pd.read_json(df_path, orient='records', lines=True),df_path) if df_path.endswith('.json')
                                       else pd.read_csv(df_path, encoding='utf-8')
                                      , **kwargs)
               for df_path in df_path_list]
    df_all = pd.concat(df_list, axis=0)
    return df_all

# 读取一系列df，并把格式转化为image、mask、caption、cate_ids
def get_df_from_df_list(df_list, **kwargs):
    df_list = [df_to_img_mask_caption(df, **kwargs) for df in df_list]
    df_all = pd.concat(df_list, axis=0)
    return df_all


# 将df的格式转化为image、mask、caption、cate_ids。包含两部分：1.列的补全 2.图片复制
def df_to_img_mask_caption(df: pd.DataFrame, label2id: dict, img_mask_save_dir,language='zh'):

    # 如果既没有mask列，也没有caption列，报错
    if 'mask' not in df.columns and 'caption' not in df.columns:
        raise Exception('df中既没有mask列，也没有caption列')

    # 如果没有caption，生成caption
    if 'caption' not in df.columns or 'cate_ids' not in df.columns:
        df['caption'], df['cate_ids'] = zip(*df['mask'].map(lambda x: get_caption_from_mask(x, label2id,language=language)))

    # 如果没有mask列，mask列为'None'
    if 'mask' not in df.columns:
        df['mask'] = 'None'

    # 只保留image、mask、caption、cate_ids列
    df = df[['image', 'mask', 'caption', 'cate_ids']]

    #只保留cate_ids中包含9的行
    df=df[df['cate_ids'].map(lambda x:9 in x)]

    # 如果img_mask_save_dir下没有image和mask文件夹，创建image和mask文件夹
    if not os.path.exists(os.path.join(img_mask_save_dir, 'image')):
        os.makedirs(os.path.join(img_mask_save_dir, 'image'))
    if not os.path.exists(os.path.join(img_mask_save_dir, 'mask')):
        os.makedirs(os.path.join(img_mask_save_dir, 'mask'))

    # 复制image到img_mask_save_dir下的image文件夹中的以caption命名的文件夹中,文件名为长度为10的随机字母
    for img_path, caption in zip(df['image'], df['caption']):
        img_save_path = os.path.join(img_mask_save_dir, 'image', caption.replace('。', ''), ''.join(
            [np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) for _ in range(10)]) + '.png')

        # 创建文件夹，以caption命名
        if not os.path.exists(os.path.dirname(img_save_path)):
            os.makedirs(os.path.dirname(img_save_path))
        # 如果img_save_path已经存在，重新生成img_save_path
        while os.path.exists(img_save_path):
            img_save_path = os.path.join(img_mask_save_dir, 'image', caption.replace('。', ''), ''.join(
                [np.random.choice(list('abcdefghijklmnopqrstuvwxyz')) for _ in range(10)]) + '.png')
        copyfile(img_path, img_save_path)
        # df中的image列改为img_save_path
        df.loc[df['image'] == img_path, 'image'] = img_save_path

    # 复制mask到image路径下的mask文件夹中的以caption命名的文件夹中,文件名为长度为10的随机字母
    for new_img_path, mask_path in zip(df['image'], df['mask']):
        mask_save_path = new_img_path.replace('\\image\\', '\\mask\\')

        # 创建文件夹，以caption命名
        if not os.path.exists(os.path.dirname(mask_save_path)):
            os.makedirs(os.path.dirname(mask_save_path))
        # 复制
        if mask_path != 'None':
            copyfile(mask_path, mask_save_path)
        # df中的mask列改为mask_save_path
        df.loc[df['image'] == new_img_path, 'mask'] = mask_save_path.replace(img_mask_save_dir, '.\\')

        # df中的image列改为相对路径
        df.loc[df['image'] == new_img_path, 'image'] = new_img_path.replace(img_mask_save_dir, '.\\')
    return df


# 由mask生成caption
def get_caption_from_mask(mask_path, label2id,language='zh',need_area=False):
    # 读取mask
    mask = Image.open(mask_path)
    mask = np.array(mask)

    # 获取cate_ids和cate_area
    cate_ids, cate_area = MaskTransform.mask_to_category_id(mask)

    #是否需要面积
    if not need_area:
        cate_area=None
    # 由cate_ids生成caption
    caption = CategoryId2Caption.category_id_to_caption(cate_ids, label2id, category_area=cate_area,language=language)

    return caption, cate_ids


if __name__ == '__main__':
    # label2id = {'背景': 0, '地物变形': 1, '地物不完整': 2, '影像有拉花': 3, '影像模糊，纹理不清晰': 4}
    #label2id={'背景':0,'云':1, '阴影':2,'拉花':3,'拼接痕迹':4,'拼接错误':5,'扭曲':6,'模糊':7,'光谱溢出':8,'条状噪声':9}
    #label2id={'背景':0,'云':1, '阴影':2,'拉花':3,'模糊':4,'扭曲':5,'拼接错误':6,'拼接痕迹':7,'光谱溢出':8,'条状噪声':9}

    #将以上label2id翻译为英文
    # label2id = {'background': 0, 'cloud': 1, 'shadow': 2, 'striped stain': 3, 'splicing trace': 4, 'splicing dislocation': 5, 'distorted object': 6,
    #             'blur area': 7,'spectral overflow': 8,'striped noise': 9}

    # 读取df
    df_path_list = ['../所有统一数据集/统一数据集_除了拼接痕迹和光谱溢出_李昶严格筛选/train删除一些.json',
                    '../所有统一数据集/统一数据集_主要是光谱溢出/all删除黑mask和边缘mask.json',
                    '../所有统一数据集/统一数据集_ps光谱溢出/all删除黑mask和边缘mask.json',
                    '../所有统一数据集/统一数据集_吴浩拼接痕迹全部/all删除黑mask和边缘mask.json',
                    '../云数据集38/test.json']
    df_path_list = ['../所有统一数据集/统一数据集_最终版_9类_0.1云数据集38_无负样本_新label2id/all.json']


    # 设置保存路径
    img_mask_save_dir = '../所有统一数据集/统一数据集_全是条纹噪声'
    if not os.path.exists(img_mask_save_dir):
        os.makedirs(img_mask_save_dir)

    # 拼接并整理df
    df_all = get_df_from_df_path_list(df_path_list, label2id=label2id, img_mask_save_dir=img_mask_save_dir,language='zh')

    # 保存df
    df_all.to_json(img_mask_save_dir + '/all.json', orient='records', lines=True, force_ascii=False)
    print('df保存成功', img_mask_save_dir + '/all.json')
