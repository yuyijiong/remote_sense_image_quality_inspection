# 导入OrderedDict
import glob
import os
import pickle
import random
# 导入OrderedDict
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch import nn
import platform


def save(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


# 看看dataloder的输出
def visual_loader(train_loader, stop=0):
    # 测试输出
    # 打印测试集的列名
    test_df_f = pd.DataFrame(train_loader.train_dataset)
    print("列名", test_df_f.columns)
    # print(test_df_f.iloc[0]) #打印第一行
    # # 打印第一行的input_ids
    # print(test_df_f.iloc[0]['input_ids'])
    # # 打印第一行的attention_mask
    # print(test_df_f.iloc[0]['attention_mask'])
    # # 打印第一行的token_type_ids
    # print(test_df_f.iloc[0]['token_type_ids'])
    # # 打印第一行的labels
    # print(test_df_f.iloc[0]['labels'])

    # train_loader输出一个batch
    batch = next(iter(train_loader))
    print("batch的类型", type(batch))
    print("batch的长度", len(batch))
    print("batch的key", batch.keys())
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['token_type_ids'].shape)
    print(batch['labels'].shape)

    if stop:
        input("按任意键继续")


# 加载断点
def load_checkpoint(checkpoint_path, model, return_path=False):
    # 搜索文件夹名包含JD_opinion_qa_checkpoint的最新的文件夹，文件夹不为空
    checkpoint_dic_load = sorted(glob.glob(checkpoint_path + '*'), key=os.path.getmtime)[-1]
    # 加载checkpoint_dic_load文件夹下的checkpoint，必须以pt或pth结尾

    checkpoint_list = [ckpt for ckpt in os.listdir(checkpoint_dic_load) if
                       ckpt.endswith('.pt') or ckpt.endswith('.pth')]
    print('load checkpoint', checkpoint_dic_load + "/" + checkpoint_list[-1])
    # 判断此路径是否为空文件
    if os.path.getsize(checkpoint_dic_load + "/" + checkpoint_list[-1]) == 0:
        raise Exception("checkpoint is empty")

    checkpoint = torch.load(checkpoint_dic_load + "/" + checkpoint_list[-1])
    # 将checkpoint中的model_state_dict的key重命名，删除头部的“_orig_mod.”
    checkpoint_rename = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("_orig_mod.", "")  # remove `_orig_mod.`
        checkpoint_rename[name] = v

    model_new_dict = model.state_dict()
    # 将checkpoint中的model_state_dict的key与model的key对应起来
    checkpoint_rename = {k: v for k, v in checkpoint_rename.items() if k in model_new_dict}
    # 将model的key对应的value更新为checkpoint中的value
    model_new_dict.update(checkpoint_rename)
    model.load_state_dict(model_new_dict)
    print('load checkpoint success', checkpoint_dic_load + "/" + checkpoint_list[-1])
    if return_path:
        return model, checkpoint_dic_load + "/" + checkpoint_list[-1]
    else:
        return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class EarlyStoppingPeft(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #保存模型和config
        model.save_pretrained(self.path)
        model.config.save_pretrained(self.path)
        self.val_loss_min = val_loss



class DF_AEDA:
    def __init__(self, raw_df, col_name: str, punc_ratio: float = 0.3):

        self.raw_df = raw_df.dropna(axis=0).reset_index(drop=True)  # 删除空行，重建行索引
        self.col_name = col_name
        self.punc_ratio = punc_ratio

    def insert_punctuation_marks(self, sentence):
        sentence = str(sentence)
        PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
        words = sentence.split(' ')
        new_line = []
        q = random.randint(1, int(self.punc_ratio * len(words) + 1))
        qs = random.sample(range(0, len(words)), q)

        for j, word in enumerate(words):
            if j in qs:
                new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
                new_line.append(word)
            else:
                new_line.append(word)
        new_line = ' '.join(new_line)
        return new_line

    # 返回一个删除了一些行的df
    def drop_0(self, drop_pro=0.9):
        ind = np.where(self.raw_df['label'] == 0)[0].tolist()  # 标签为0的行索引

        ind_chose = random.sample(ind, round(len(ind) * (drop_pro)))  # 标签为0的行索引 随机抽取80%

        df_chose = self.raw_df.drop(index=ind_chose).reset_index(drop=True)  # 删掉这些行

        return df_chose

    # 输出一个增强过的数据帧
    def aeda_df(self, drop=False, drop_pro=0.9, multi_pro=True):
        if self.punc_ratio == 0:
            return self.raw_df.__deepcopy__()

        if drop:
            aeda_df = self.drop_0(drop_pro)  # 删除0标签
        else:
            aeda_df = self.raw_df.__deepcopy__()

        texts = aeda_df[self.col_name]  # 提取文本的列

        if multi_pro:
            texts_AEDA = Pool().map(self.insert_punctuation_marks, list(texts))  # 随机加标点，比例0.3
        else:
            texts_AEDA = map(self.insert_punctuation_marks, list(texts))  # 随机加标点，比例0.3

        texts_AEDA = pd.Series(texts_AEDA)

        aeda_df.loc[:, self.col_name] = texts_AEDA  # 修改

        aeda_df = aeda_df.dropna(axis=0)
        aeda_df = aeda_df.reset_index(drop=True)

        return aeda_df


# [*,512,768] 变为 [*,768]
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    # last_hidden_state [*,512,768]   attention_mask [*,512]
    # 相当于对512个词向量取平均，但是不算pad的词向量
    def forward(self, last_hidden_state, attention_mask):
        try:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        except:
            print(last_hidden_state.size(), attention_mask.size())
            raise
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)  # 把512个词向量相加，除了为0的，得到[*,768]
        sum_mask = input_mask_expanded.sum(1)  # 把512个mask相加，每一维都等于序列中不为0的字数,得到[*,768]
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 为了防止出现0，限制最小值
        mean_embeddings = sum_embeddings / sum_mask  # 除以不为0的字数
        return mean_embeddings


# 全连接（4,1），代表权重是1*4矩阵(不是4*1)
class weightConstraint_sumone(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = torch.softmax(w, dim=1)  # 使4个权重之和为1
            module.weight.data = w


#返回cache_dir
def get_cache_dir():
    system = platform.system()
    print(system)
    cache_dir = 'D:/Models' if system == 'Windows' else None
    return cache_dir
