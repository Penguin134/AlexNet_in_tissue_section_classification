import numpy
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import numpy as np
import time


class iGEM_data(Dataset):
    def __init__(self, rge):
        self.path = os.getcwd() + '/../lung_image_sets'
        self.file_suffix = '.jpeg'
        self.file_prefix = ['lungaca', 'lungn', 'lungscc']
        self.label = {0: 'lung_aca', 1: 'lung_n', 2: 'lung_scc'}
        self.rge = rge
        self.length = (rge[1] - rge[0]) * 3

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        lung_type = index // (self.rge[1] - self.rge[0])
        index = index - (self.rge[1] - self.rge[0]) * lung_type
        img_path = f'{self.path}/{self.label[lung_type]}/{self.file_prefix[lung_type]}{index+1}{self.file_suffix}'
        # print(img_path)
        x = cv2.imread(img_path)
        if x is None:
            print(f'Error when: path={img_path}\n index={(self.rge[1] - self.rge[0]) * lung_type}')
            raise FileNotFoundError
        y = lung_type
        x = x/255.
        x = torch.from_numpy(x).float()
        return x, y


if __name__ == '__main__':
    myData = iGEM_data((0, 5000))
    print(myData.length)
    xx, yy = myData.__getitem__(1)
    print(xx.size())  # 768, 768, 3
    print(yy)  # 3
