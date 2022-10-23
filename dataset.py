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


def get_dataset(path, inference=None):
    
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

    if inference is None:
        train_folders = glob(path+'/**')
        valid_folders = random.sample(train_folders, 5)
        for folder in valid_folders:
            train_folders.remove(folder)

        train_files = sum([glob(folder+'/annotations/*.json') \
                           for folder in train_folders], [])
        valid_files = sum([glob(folder+'/annotations/*.json') \
                           for folder in valid_folders], [])

        return {
            'train': load_files(train_files),
            'valid': load_files(valid_files),
        }

    else:
        folders = glob(path+'/**')
        test_files = sum([glob(folder+'/annotations/*.json')\
                          for folder in folders], [])
        return load_files(test_files)


def get_segmap(image_list, anno_list, label_list):

    def get_rgb(images, annos, labels):
        rgb_list = []
        for i in range(len(images)):
            img = images[i].copy()
            for j in range(len(annos[i])):
                cv2.fillPoly(img, [annos[i][j]], classes[labels[i][j]], cv2.LINE_AA)
            rgb_list.append(img)

        return np.array(rgb_list)

    rgb_list = get_rgb(image_list, anno_list, label_list)

    segmap_list = []
    for rgb in tqdm(rgb_list):
        label = np.ones(rgb.shape[:2])
        for i, color in enumerate(classes.values()):
            label[(rgb==color).sum(2)==3] = i
        segmap_list.append(label)
    return np.array(segmap_list).astype(np.uint8)


class SemanticSegmentationDataset(Dataset):
    
    def __init__(
        self,
        path,
        subset='train',
        crop_size=None,
        transform=None,
    ):
        assert subset in ('train', 'valid', 'test')
        inference = True if subset == 'test' else None
        files = get_dataset(path, inference)

        if subset in ('train', 'valid'):
            train_images, train_annos, train_labels = files['train']
            valid_images, valid_annos, valid_labels = files['valid']

            train_labels = get_segmap(train_images, train_annos, train_labels)
            valid_labels = get_segmap(valid_images, valid_annos, valid_labels)
        else:
            test_images, test_annos, test_labels = files
            test_labels = get_segmap(test_images, test_annos, test_labels)
        
        if subset == 'train':
            self.images = train_images
            self.labels = train_labels
        elif subset == 'valid':
            self.images = valid_images
            self.labels = valid_labels
        else:
            self.images = test_images
            self.labels = test_labels
        
        self.transform = transform

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform = Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(
                scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
                size=(crop_size)
            ),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        if self.transform is not None:
            im_lb = dict(im=images, lb=labels)
            im_lb = self.transform(im_lb)
            images, labels = im_lb['im'], im_lb['lb']
        images = self.totensor(images)
        labels = torch.LongTensor(labels)
        return images, labels