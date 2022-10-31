import random
import json
import cv2
import numpy as np
from glob import glob
from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from util.transform import RandomHorizontalFlip, RandomResizedCrop, Compose

classes = {
    1: [255, 0, 0], # non
    2: [128, 0, 0], # road
    3: [255, 255, 0], # full_line
    4: [128, 128, 0], # dotted_line
    5: [0, 255, 0], # road_mark
    6: [0, 128, 0], # crosswalk
    7: [0, 255, 255], # speed_bump
    8: [0, 128, 128], # curb
    9: [0, 0, 255], # static
    10: [0, 0, 128], # sidewalk
    11: [255, 0, 255], # parking_place
    12: [128, 0, 128], # vehicle
    13: [255, 127, 80], # motorcycle
    14: [184, 134, 11], # bicycle
    15: [127, 255, 0], # pedestrian
    16: [0, 191, 255], # rider
    17: [255, 192, 203], # dynamic
    18: [165, 42, 42], # traffic_sign
    19: [210, 105, 30], # traffic_light
    20: [240, 230, 140], # pole
    21: [245, 245, 220], # building
    22: [0, 100, 0], # guadrail
    23: [64, 224, 208], # sky
    24: [70, 130, 180], # water
    25: [106, 90, 205], # mountain
    26: [75, 0, 130], # vegetation
    27: [139, 0, 139], # bridge
    28: [255, 20, 147], # undefined/area
}


def get_dataset(path):
    
    def load_files(files):
        images, annos, labels = [], [], []
        for file in tqdm(files):
            with open(file) as f:
                object_list = json.load(f)

                poly_list, label_list = [], []
                for obj in object_list['annotations']:
                    try:
                        seg = obj['segmentation'][0]
                        points = [
                            [int(seg[i]), int(seg[i+1])] for i in range(0, len(seg), 2)
                        ]
                        poly_list.append(np.array(points))
                        label = obj['category_id']
                        label_list.append(label)
                    except:
                        pass

                annos.append(poly_list)
                labels.append(label_list)

            img_file = file.replace('annotations', 'images').replace('.json', '.jpg')
            img = cv2.imread(img_file)[:, :, ::-1]
            images.append(img)

        return np.array(images), np.array(annos), np.array(labels)

    folders = glob(path+'/**')
    files = sum([glob(folder+'/annotations/*.json') \
                for folder in train_folders], [])

    return load_files(files)


def get_segmap(image_list, anno_list, label_list):

    def get_rgb(images, annos, labels):
        rgb_list = []
        for i in range(len(images)):
            img = images[i].copy()
            for j in range(len(annos[i])):
                cv2.fillPoly(img, [annos[i][j]], classes[labels[i][j]], cv2.LINE_8)
            rgb_list.append(img)

        return np.array(rgb_list)

    segmap_list = []
    for rgb in tqdm(get_rgb(image_list, anno_list, label_list))
        label = np.ones(rgb.shape[:2])
        for i, color in enumerate(classes.values()):
            label[(rgb==color).sum(2)==3] = i
        segmap_list.append(label)
    return np.array(segmap_list).astype(np.uint8)


class EvalDataset(Dataset):

    def __init__(self, path):
        
        self.images, annos, labels = get_dataset(path)
        self.labels = get_segmap(images, annos, labels)
        
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
            27: ignore_index, 28: 17, 29: ignore_index, 30: ignore_index,
            31: ignore_index, 32: ignore_index, 33: ignore_index,
            34: ignore_index, 35: ignore_index, 36: ignore_index,
            37: ignore_index, 38: ignore_index, 39: ignore_index,
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


