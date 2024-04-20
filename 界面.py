# coding=gbk
import tkinter as tk
from �������_���ݴ��� import detect_remote_sense
import os
from tkinter import filedialog
from typing import List

label2id = {'����': 0, '��': 1, '��Ӱ': 2, '����': 3, 'ģ��': 4, '�������': 5, 'Ť��': 6, 'ƴ�Ӻۼ�': 7,
            'ƴ�Ӵ���': 8, '��״����': 9, '����ȱʧ': 10}


def main(img_path, label2chose, patch_size: str, overlap=None,logits_alpha="1"):
    id2label = {v: k for k, v in label2id.items()}
    # label2chose�ĸ�ʽΪ"1,2,3"������ת��Ϊ[1,2,3]
    if label2chose == '':
        label2chose = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        label2chose = [int(i) for i in label2chose.split(',')]
    patch_size = [int(i) for i in patch_size.split(',')]
    if overlap is not None:
        overlap = [int(i) for i in overlap.split(',')]
    logits_alpha = float(logits_alpha)
    # ��ӡ���еĲ���
    print('ͼƬ�ļ���·��', img_path)
    print('ѡ��ı�ǩid', label2chose)
    print('ѡ��ı�ǩ����', [id2label[i] for i in label2chose])
    print('С��ͼ��Ĵ�С', patch_size)
    print('overlap', overlap)
    print('��ȷ�̶�', logits_alpha)

    seg_model_path = './segformer_b5_6label'
    classify_model_path = './swin_9label'
    print('img_path', img_path)

    # ���img_path�����ļ���
    if not os.path.isdir(img_path):
        shp_save_dir = os.path.dirname(img_path)

        detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                            classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                            label_to_chose=label2chose, use_logits_process=True,logits_alpha=logits_alpha)
    # ���img_path���ļ���
    else:
        # ��ȡ�ļ������Լ��������ļ����������ͼƬ·�������ͼƬ��".img"��".tif"��β�����ȡ
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

        print('����ͼƬ·��', img_path_list)
        for img_path in img_path_list:
            # shp_save_dirΪͼƬ�����ļ������½�һ��img_path���ļ������ļ���
            shp_save_dir = os.path.dirname(img_path) + '/' + os.path.basename(img_path).split('.')[0]
            detect_remote_sense(img_path, label2id, shp_save_dir, seg_model_path=seg_model_path,
                                classify_model_path=classify_model_path, patch_size=patch_size, overlap=overlap,
                                label_to_chose=label2chose,use_logits_process=True,logits_alpha=logits_alpha)

    print('�����ɣ�')


def run_program(file_path, label_param):
    # �������д��������߼�
    print("�ļ�·��:", file_path)
    print("��ǩ����:", label_param)

#ѡ���ļ���
def select_dir():
    path = filedialog.askdirectory()
    filepath_entry.config(text=path)

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
    select_dir_button = tk.Button(window, text="ѡ���ļ���", command=select_dir)
    select_dir_button.pack()

    # ����ѡ���ļ���ť
    select_path_button = tk.Button(window, text="ѡ��ͼƬ", command=select_file)
    select_path_button.pack()

    # ��ʾ·���ı�ǩ
    filepath_entry = tk.Label(window)
    filepath_entry.pack()
    # ������ǩ�����
    label_label = tk.Label(window, text="��Ҫѡ��ı�ǩ(�������֣�Ӣ�Ķ��ŷָ�)��")
    label_label.pack()
    label_entry = tk.Entry(window)
    label_entry.pack()

    # ����������С�����
    patch_size_label = tk.Label(window, text="�����С(����)��")
    patch_size_label.pack()
    patch_size_entry = tk.Entry(window)
    patch_size_entry.pack()

    # ����������С�����
    overlap_label = tk.Label(window, text="����֮���ص���С(����)��")
    overlap_label.pack()
    overlap_entry = tk.Entry(window)
    overlap_entry.pack()

    # ������ȷ�̶������
    logit_a_label = tk.Label(window, text="��ȷ�̶ȣ�")
    logit_a_label.pack()
    logit_a_entry = tk.Entry(window)
    logit_a_entry.pack()

    run_button = tk.Button(window, text="����",
                           command=lambda: main(filepath_entry.cget("text"),
                                                label_entry.get() or "",
                                                patch_size_entry.get() or "500",
                                                overlap=overlap_entry.get() or None,
                                                logits_alpha=logit_a_entry.get() or "0.5"))
    run_button.pack()

    # ��ʾlabel2id
    label2id_label = tk.Label(window, text="��ǩ���ƺ�id�Ķ�Ӧ��ϵ��" + str(label2id))
    label2id_label.pack()

    # ������ѭ��
    window.mainloop()
