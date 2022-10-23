import argparse
import time

import torch
from torch.utils.data import DataLoader

from model.model import OurModel
from util.metric import Metrics
from util.loss import OhemCELoss
from dataset import SemanticSegmentationDataset


def eval(model, data_loader, loss_func, mean_iou, weight_path=None,
         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    with torch.no_grad():
        model.eval()
        init_time = time.time()
        batch_loss, batch_miou = 0, 0
        for batch, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            loss = loss_func(outputs, labels)
            batch_loss += loss.item()

            miou = mean_iou(outputs, labels)
            batch_miou += miou.item()

        end_time = time.time()

    print(f'time: {end_time-init_time:.2f}s')
    print(f'loss: {batch_loss/(batch+1):.3f}, miou: {batch_miou/(batch+1):.3f}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate Model', add_help=False)
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='the directory of weight of pre-trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the directory where your dataset is located')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='the number of classes in dataset')
    return parser

def main(args):

    test_data = SemanticSegmentationDataset(
        path=args.data_dir,
        subset='test',
        crop_size=None,
        transform=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    loss_func = OhemCELoss(thresh=0.7)
    metric = Metrics(n_classes=args.num_classes, dim=1)

    model = OurModel(weight_path=None, num_classes=args.num_classes)

    eval(model, test_loader, loss_func, metric.mean_iou, args.weight_dir)