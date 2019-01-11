from __future__ import print_function, division
import os
from PIL import Image
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torchvision import transforms, utils, datasets



class CSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples, transforms):
        self.path = path
        self.chunksize = chunksize
        self.len = int(nb_samples / self.chunksize)
        self.to_tensor = transforms

    def __getitem__(self, index):

        x = next(pd.read_csv(self.path, skiprows=index** self.chunksize+1,  chunksize=1, names=['anc', 'pos', 'nec']))
        anc = (x.anc.values)
        # pos = (x.pos.values)
        # neg = (x.neg.values)
        # print(anc[0])
        img_as_img = Image.open(anc[0])
        img_as_tensor = self.to_tensor(img_as_img)
        print(type(img_as_tensor))
        return img_as_tensor

    def __len__(self):
        return int(self.len)



if __name__ =="__main__":
    np_samples = 3
    data_transform = transforms.Compose([

              transforms.RandomSizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
          ])
    dataset = CSVDataset('./folder_to_csv_deep_fashion.csv', chunksize=1, nb_samples=np_samples ,transforms=data_transform)
    print(dataset.len)
    #
    #
    # for i in range(dataset.len):
    #     try:
    #         print(i, dataset[i])
    #
    #     except:
    #         print("end of index")

    loader = DataLoader(dataset, batch_size=10, num_workers=1, shuffle=False)

    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))


    dataset = pd.read_csv('./folder_to_csv_deep_fashion.csv', skiprows=1,  chunksize=1, names=['anc', 'pos', 'nec'])


    for i in range(3):
        # print(next(dataset))
        x=next(dataset)
        print("type x ", type(x))
        x=x.anc.values
        print(x)
