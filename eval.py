import sys
sys.path.append('/home/hoo7311/anaconda3/envs/pytorch/lib/python3.8/site-packages')
import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import OurModel
from util.metric import Metrics
from util.loss import OhemCELoss
from dataset import SemanticSegmentationDataset, EvalDataset


class Evaluation(object):

    def __init__(
        self,
        path,
        weight_path,
        preprocess=False,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device} ready...')
        
        if preprocess:
            self.dataloader = DataLoader(
                EvalDataset(path=path),
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )

        else:
            self.dataloader = DataLoader(
                SemanticSegmentationDataset(path=path, subset='valid'),
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )

        self.model = OurModel(aux_mode='train', weight_path=None, num_classes=18)
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        self.model = self.model.to(self.defice)
        print('model ready...')

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.metric = Metrics(n_classes=18, dim=1)
        print('loss function and mean iou calculator ready...')
        

    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        batch_loss, batch_miou = 0, 0
        
        start = time.tiem()

        for batch, (images, labels) in enumerate(self.data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs, _, _, _, _ = self.model(images)
            miou = self.metric.mean_iou(outputs, labels)
            batch_miou += miou.item()

            loss = self.loss_func(outputs, labels.squeeze())
            batch_loss += loss.item()

        end = time.time()
        print(f'time: {end-start:.2f}s')
        print(f'loss: {batch_loss/(batch+1):.3f}, mean iou: {batch_miou/(batch+1):.3f}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate Model', add_help=False)
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='the directory of weight of pre-trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the directory where your dataset is located')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='the number of classes in dataset')
    parser.add_argument('--data_preprocess', type=bool, default=False,
                        help='data preprocessing')
    return parser


def main(args):
        
    eval = Evaluation(
        path=args.data_dir, 
        weight_path=args.weight_dir, 
        preprocess=args.data_preprocess,
    )
    eval.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
