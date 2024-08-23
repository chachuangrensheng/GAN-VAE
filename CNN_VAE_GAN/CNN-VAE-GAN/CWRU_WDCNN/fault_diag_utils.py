import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset


class Data_set(Dataset):
    '''封装数据集类'''

    def __init__(self, data, label):
        super().__init__()
        # self.x = torch.tensor(data)
        self.x = torch.tensor(np.array(data), dtype=torch.float32)
        self.y = torch.tensor(label).view(-1, 1)

    def __getitem__(self, idx):
        assert idx < len(self.y)
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def file_read(filename: str):
    '''读取数据'''
    label = filename.split('.')[0].split('/')[-1].split("\\")[-1]
    data = pd.read_csv(filename)
    array = data.to_numpy(dtype=np.float32)
    return get_idx(label), array


def get_idx(label: str):
    '''根据标签获得其索引'''
    # labels = ['X097_DE', 'X105_DE', 'X118_DE', 'X130_DE','X169_DE','X185_DE','X197_DE','X209_DE','X222_DE','X234_DE']
    labels = ['X097_DE', 'X105_DE', 'X118_DE', 'X130_DE']
    idx = labels.index(label)
    return idx


def get_label(idx):
    '''根据索引获得标签内容'''
    # labels = ['X097_DE', 'X105_DE', 'X118_DE', 'X130_DE','X169_DE','X185_DE','X197_DE','X209_DE','X222_DE','X234_DE']
    labels = ['normal', 'abnormal']
    return labels[idx]
