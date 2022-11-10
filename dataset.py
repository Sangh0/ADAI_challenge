import random
import numpy as np
from glob import glob
from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from util.transform_pillow import (
    RandomCrop, HorizontalFlip, RandomScale, Compose,
)
from util.preprocess import get_dataset, get_segmap


class SemanticSegmentationDataset(Dataset):
    
    def __init__(
        self,
        path,
        subset='train',
        crop_size=None,
        transforms_=None,
        ignore_index=255,
    ):
        assert subset in ('train', 'valid')
        self.subset = subset

        #train_folders = glob(path+'white*')
        #valid_folders = [path+'white1', path+'white5', path+'white10']
        train_folders = glob(path+'**')
        valid_folders = [path+'dark1', path+'dark4', path+'white1', path+'white8', path+'white16']
        for folder in valid_folders:
            train_folders.remove(folder)

        train_labels = sum([glob(folder+'/labels/*.png') for folder in train_folders], [])
        valid_labels = sum([glob(folder+'/labels/*.png') for folder in valid_folders], [])

        if subset == 'train':
            self.images = [file.replace('labels', 'images').replace('.png', '.jpg') \
                    for file in train_labels]
            self.labels = train_labels
            print('the total number of train data:', len(self.labels))
        else:
            self.images = [file.replace('labels', 'images').replace('.png', '.jpg') \
                    for file in valid_labels]
            self.labels = valid_labels
            print('the total number of valid data:', len(self.labels))

        assert len(self.images) == len(self.labels), 'image and label size does not match'

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.transforms_ = Compose([
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75)),
            RandomCrop(crop_size),
        ]) if transforms_ is not None else None

        self.valid_transforms_ = Compose([
            RandomCrop(crop_size),
        ])

        self.mapping_classes = {
            0: ignore_index, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 
            7: ignore_index, 8: ignore_index, 9: 1, 10: 0, 11: 2, 
            12: 3, 13: 4, 14: 5, 15: 6, 16: ignore_index, 17: 7, 
            18: 8, 19: 9, 20: 10, 21: 11, 22: 12, 23: ignore_index, 
            24: ignore_index, 25: 13, 26: 14, 27: ignore_index,
        }

        self.classes = 15

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = Image.open(self.images[idx]).convert('RGB')
        labels = Image.open(self.labels[idx]).convert('L')
        
        if self.transforms_ is not None:
            if self.subset == 'train':
                im_lb = dict(im=images, lb=labels)
                im_lb = self.transforms_(im_lb)
                images, labels = im_lb['im'], im_lb['lb']
            else:
                im_lb = dict(im=images, lb=labels)
                im_lb = self.valid_transforms_(im_lb)
                images, labels = im_lb['im'], im_lb['lb']
        images = self.totensor(images)
        labels = np.array(labels).astype(np.int32)[np.newaxis, :]
        labels = self.convert_label(labels)
        return images, labels

    def convert_label(self, label):
        for k in self.mapping_classes:
            label[label==k] = self.mapping_classes[k]
        return torch.LongTensor(label)


class EvalDataset(Dataset):

    def __init__(self, path):

        self.images, annos, labels = get_dataset(path)
        self.labels = get_segmap(images, annos, labels)
        
        assert len(self.images) == len(self.labels)
        print('The number of dataset is {len(self.labels)}')

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.mapping_classes = {
            0: ignore_index, 1: 0, 2: ignore_index, 3: ignore_index,
            4: 1, 5: 2, 6: ignore_index, 7: ignore_index, 8: ignore_index,
            9: 3, 10: ignore_index, 11: 4, 12: 5, 13: 6, 14: 7, 15: 8,
            16: ignore_index, 17: 9, 18: 10, 19: 11, 20: 12, 21: 13,
            22: 14, 23: ignore_index, 24: ignore_index, 25: 15, 26: 16,
            27: ignore_index,
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]

        labels = labels.astype(np.int32)[np.newaxis, :]

        images = self.totensor(images)
        labels = self.convert_label(labels)

        return images, labels

    def convert_label(self, label):
        for k in self.mapping_classes:
            label[label==k] = self.mapping_classes[k]
        return torch.LongTensor(label)
