# coding=gbk
import os
import time
import tkinter as tk
from tkinter import filedialog

import pandas as pd

from shp_to_dataset import get_df_from_all_dir
from ͳһ���ݼ����� import get_df_from_df_list

label2id = {'����': 0, '��': 1, '��Ӱ': 2, '����': 3, 'ģ��': 4, '�������': 5, 'Ť��': 6, 'ƴ�Ӻۼ�': 7,
            'ƴ�Ӵ���': 8, '��״����': 9, '����ȱʧ': 10}


def main(img_path, label2chose, patch_size: str,img_mask_save_dir=None):
    id2label = {v: k for k, v in label2id.items()}
    # label2chose�ĸ�ʽΪ"1,2,3"������ת��Ϊ[1,2,3]
    if label2chose == '':
        label2chose = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        label2chose = [int(i) for i in label2chose.split(',')]
    patch_size = [int(i) for i in patch_size.split(',')]
    # ��ӡ���еĲ���
    print('ͼƬ�ļ���·��', img_path)
    print('ѡ��ı�ǩid', label2chose)
    print('ѡ��ı�ǩ����', [id2label[i] for i in label2chose])
    print('С��ͼ��Ĵ�С', patch_size)


    # ���img_path���ļ���

    # pool=Pool(4)
    df_list = list(map(lambda x: get_df_from_all_dir(data_dir=img_path,
                                                     label2id=label2id,
                                                     test_only=False,
                                                     max_edge=x,
                                                     error_type_name_list=[id2label[i] for i in label2chose],
                                                     main_label=None,
                                                     nonzero_pixel_threshold=0.0001,
                                                     # add_label='��Ӱ����',
                                                     save_all=0), patch_size))
    # ƴ��df_list
    df_all = pd.concat(df_list, axis=0)

    # ���ñ���·��
    if img_mask_save_dir is None:
        img_mask_save_dir = './ͳһ���ݼ�/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists(img_mask_save_dir):
        os.makedirs(img_mask_save_dir)
    df_all=get_df_from_df_list([df_all],label2id=label2id, img_mask_save_dir=img_mask_save_dir,language='zh')
    # ����df
    df_all.to_json(img_mask_save_dir + '/all.json', orient='records', lines=True, force_ascii=False)
    print('df����ɹ�', img_mask_save_dir + '/all.json')


def run_program(file_path, label_param):
    # �������д��������߼�
    print("�ļ�·��:", file_path)
    print("��ǩ����:", label_param)

#ѡ���ļ���
def select_dir():
    path = filedialog.askdirectory()
    filepath_entry.config(text=path)
#ѡ���ļ���
def select_save_dir():
    path = filedialog.askdirectory()
    savefilepath_entry.config(text=path)

#ѡ�񵥸��ļ�
def select_file():
    path = filedialog.askopenfilename()
    filepath_entry.config(text=path)

# def run_program_with_params():
#     path = path_label.cget("text")
#     label_param = label_entry.get()
#     run_program(path, label_param)


if __name__ == '__main__':
    # ����������
    window = tk.Tk()

    # ����ѡ��·����ť
    select_dir_button = tk.Button(window, text="ѡ��ͼƬ�����ļ��е��ļ���", command=select_dir)
    select_dir_button.pack()

    # ��������·��ѡ��ť
    save_path_button = tk.Button(window, text="ѡ�񱣴�·��", command=select_save_dir)
    save_path_button.pack()

    # ��ʾ·���ı�ǩ
    filepath_label = tk.Label(window, text="�����ļ���·����")
    filepath_label.pack()
    filepath_entry = tk.Label(window)
    filepath_entry.pack()
    savefilepath_label = tk.Label(window, text="����·����")
    savefilepath_label.pack()
    savefilepath_entry = tk.Label(window)
    savefilepath_entry.pack()

    # ������ǩ�����
    label_label = tk.Label(window, text="��Ҫѡ��ı�ǩ��")
    label_label.pack()
    label_entry = tk.Entry(window)
    label_entry.pack()

    # ����������С�����
    patch_size_label = tk.Label(window, text="�����С(����)��")
    patch_size_label.pack()
    patch_size_entry = tk.Entry(window)
    patch_size_entry.pack()



    run_button = tk.Button(window, text="����",
                           command=lambda: main(filepath_entry.cget("text"),
                                                label_entry.get() or "",
                                                patch_size_entry.get() or "500",
                                                img_mask_save_dir=savefilepath_entry.cget("text")))
    run_button.pack()

    # ��ʾlabel2id
    label2id_label = tk.Label(window, text="��ǩ���ƺ�id�Ķ�Ӧ��ϵ��" + str(label2id))
    label2id_label.pack()

    # ������ѭ��
    window.mainloop()
