# coding=gbk
import os
import time
import tkinter as tk
from tkinter import filedialog

import pandas as pd

from shp_to_dataset import get_df_from_all_dir
from 统一数据集生成 import get_df_from_df_list

label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '模糊': 4, '光谱溢出': 5, '扭曲': 6, '拼接痕迹': 7,
            '拼接错误': 8, '条状噪声': 9, '像素缺失': 10}


def main(img_path, label2chose, patch_size: str,img_mask_save_dir=None):
    id2label = {v: k for k, v in label2id.items()}
    # label2chose的格式为"1,2,3"，将其转化为[1,2,3]
    if label2chose == '':
        label2chose = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        label2chose = [int(i) for i in label2chose.split(',')]
    patch_size = [int(i) for i in patch_size.split(',')]
    # 打印所有的参数
    print('图片文件夹路径', img_path)
    print('选择的标签id', label2chose)
    print('选择的标签名称', [id2label[i] for i in label2chose])
    print('小块图像的大小', patch_size)


    # 如果img_path是文件夹

    # pool=Pool(4)
    df_list = list(map(lambda x: get_df_from_all_dir(data_dir=img_path,
                                                     label2id=label2id,
                                                     test_only=False,
                                                     max_edge=x,
                                                     error_type_name_list=[id2label[i] for i in label2chose],
                                                     main_label=None,
                                                     nonzero_pixel_threshold=0.0001,
                                                     # add_label='阴影过大',
                                                     save_all=0), patch_size))
    # 拼接df_list
    df_all = pd.concat(df_list, axis=0)

    # 设置保存路径
    if img_mask_save_dir is None:
        img_mask_save_dir = './统一数据集/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists(img_mask_save_dir):
        os.makedirs(img_mask_save_dir)
    df_all=get_df_from_df_list([df_all],label2id=label2id, img_mask_save_dir=img_mask_save_dir,language='zh')
    # 保存df
    df_all.to_json(img_mask_save_dir + '/all.json', orient='records', lines=True, force_ascii=False)
    print('df保存成功', img_mask_save_dir + '/all.json')


def run_program(file_path, label_param):
    # 在这里编写主程序的逻辑
    print("文件路径:", file_path)
    print("标签参数:", label_param)

#选择文件夹
def select_dir():
    path = filedialog.askdirectory()
    filepath_entry.config(text=path)
#选择文件夹
def select_save_dir():
    path = filedialog.askdirectory()
    savefilepath_entry.config(text=path)

#选择单个文件
def select_file():
    path = filedialog.askopenfilename()
    filepath_entry.config(text=path)

# def run_program_with_params():
#     path = path_label.cget("text")
#     label_param = label_entry.get()
#     run_program(path, label_param)


if __name__ == '__main__':
    # 创建主窗口
    window = tk.Tk()

    # 创建选择路径按钮
    select_dir_button = tk.Button(window, text="选择图片所在文件夹的文件夹", command=select_dir)
    select_dir_button.pack()

    # 创建保存路径选择按钮
    save_path_button = tk.Button(window, text="选择保存路径", command=select_save_dir)
    save_path_button.pack()

    # 显示路径的标签
    filepath_label = tk.Label(window, text="总体文件夹路径：")
    filepath_label.pack()
    filepath_entry = tk.Label(window)
    filepath_entry.pack()
    savefilepath_label = tk.Label(window, text="保存路径：")
    savefilepath_label.pack()
    savefilepath_entry = tk.Label(window)
    savefilepath_entry.pack()

    # 创建标签输入框
    label_label = tk.Label(window, text="需要选择的标签：")
    label_label.pack()
    label_entry = tk.Entry(window)
    label_entry.pack()

    # 创建补丁大小输入框
    patch_size_label = tk.Label(window, text="区块大小(像素)：")
    patch_size_label.pack()
    patch_size_entry = tk.Entry(window)
    patch_size_entry.pack()



    run_button = tk.Button(window, text="运行",
                           command=lambda: main(filepath_entry.cget("text"),
                                                label_entry.get() or "",
                                                patch_size_entry.get() or "500",
                                                img_mask_save_dir=savefilepath_entry.cget("text")))
    run_button.pack()

    # 显示label2id
    label2id_label = tk.Label(window, text="标签名称和id的对应关系：" + str(label2id))
    label2id_label.pack()

    # 运行主循环
    window.mainloop()
