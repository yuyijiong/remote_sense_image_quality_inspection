# coding=gbk
import tkinter as tk
from 检测流程_数据处理 import detect_remote_sense
import os
from tkinter import filedialog
from typing import List

label2id = {'背景': 0, '云': 1, '阴影': 2, '拉花': 3, '模糊': 4, '光谱溢出': 5, '扭曲': 6, '拼接痕迹': 7,
            '拼接错误': 8, '条状噪声': 9, '像素缺失': 10}


def main(img_path, label2chose, patch_size: str, overlap=None,logits_alpha="1"):
    id2label = {v: k for k, v in label2id.items()}
    # label2chose的格式为"1,2,3"，将其转化为[1,2,3]
    if label2chose == '':
        label2chose = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        label2chose = [int(i) for i in label2chose.split(',')]
    patch_size = [int(i) for i in patch_size.split(',')]
    if overlap is not None:
        overlap = [int(i) for i in overlap.split(',')]
    logits_alpha = float(logits_alpha)
    # 打印所有的参数
    print('图片文件夹路径', img_path)
    print('选择的标签id', label2chose)
    print('选择的标签名称', [id2label[i] for i in label2chose])
    print('小块图像的大小', patch_size)
    print('overlap', overlap)
    print('精确程度', logits_alpha)

    seg_model_path = './segformer_b5_6label'
    classify_model_path = './swin_9label'
    print('img_path', img_path)

    # 如果img_path不是文件夹
    if not os.path.isdir(img_path):
        shp_save_dir = os.path.dirname(img_path)

        detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                            classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                            label_to_chose=label2chose, use_logits_process=True,logits_alpha=logits_alpha)
    # 如果img_path是文件夹
    else:
        # 读取文件夹下以及所有子文件夹里的所有图片路径，如果图片以".img"或".tif"结尾，则读取
        img_path_list = []
        for root, dirs, files in os.walk(img_path):
            for file in files:
                if file.endswith(".img") or file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg") \
                        or file.endswith(".jpeg") or file.endswith(".bmp") or file.endswith(".gif") or file.endswith(
                    ".TIF") \
                        or file.endswith(".IMG") or file.endswith(".PNG") or file.endswith(".JPG") or file.endswith(
                    ".JPEG") \
                        or file.endswith(".BMP") or file.endswith(".GIF") or file.endswith(".tiff") or file.endswith(
                    ".TIFF") \
                        or file.endswith(".IMG"):
                    img_path_list.append(os.path.join(root, file))

        print('所有图片路径', img_path_list)
        for img_path in img_path_list:
            # shp_save_dir为图片所在文件夹下新建一个img_path的文件名的文件夹
            shp_save_dir = os.path.dirname(img_path) + '/' + os.path.basename(img_path).split('.')[0]
            detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                                classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                                label_to_chose=label2chose,use_logits_process=True,logits_alpha=logits_alpha)

    print('检测完成！')


def run_program(file_path, label_param):
    # 在这里编写主程序的逻辑
    print("文件路径:", file_path)
    print("标签参数:", label_param)

#选择文件夹
def select_dir():
    path = filedialog.askdirectory()
    filepath_entry.config(text=path)

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
    select_dir_button = tk.Button(window, text="选择文件夹", command=select_dir)
    select_dir_button.pack()

    # 创建选择文件按钮
    select_path_button = tk.Button(window, text="选择图片", command=select_file)
    select_path_button.pack()

    # 显示路径的标签
    filepath_entry = tk.Label(window)
    filepath_entry.pack()
    # 创建标签输入框
    label_label = tk.Label(window, text="需要选择的标签(输入数字，英文逗号分隔)：")
    label_label.pack()
    label_entry = tk.Entry(window)
    label_entry.pack()

    # 创建补丁大小输入框
    patch_size_label = tk.Label(window, text="区块大小(像素)：")
    patch_size_label.pack()
    patch_size_entry = tk.Entry(window)
    patch_size_entry.pack()

    # 创建补丁大小输入框
    overlap_label = tk.Label(window, text="区块之间重叠大小(像素)：")
    overlap_label.pack()
    overlap_entry = tk.Entry(window)
    overlap_entry.pack()

    # 创建精确程度输入框
    logit_a_label = tk.Label(window, text="精确程度：")
    logit_a_label.pack()
    logit_a_entry = tk.Entry(window)
    logit_a_entry.pack()

    run_button = tk.Button(window, text="运行",
                           command=lambda: main(filepath_entry.cget("text"),
                                                label_entry.get() or "",
                                                patch_size_entry.get() or "500",
                                                overlap=overlap_entry.get() or None,
                                                logits_alpha=logit_a_entry.get() or "0.5"))
    run_button.pack()

    # 显示label2id
    label2id_label = tk.Label(window, text="标签名称和id的对应关系：" + str(label2id))
    label2id_label.pack()

    # 运行主循环
    window.mainloop()
