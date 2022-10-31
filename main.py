import argparse
import sys
sys.path.append('/home/hoo7311/anaconda3/envs/pytorch/lib/python3.8/site-packages')

from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import SemanticSegmentationDataset
from model.model import OurModel
from train import Trainer


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training', add_help=False)
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='the directory of weight of pre-trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the directory where your dataset is located')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--num_classes', type=int, default=18,
                        help='the number of classes in dataset')
    parser.add_argument('--lr_scheduling', type=bool, default=True,
                        help='apply learning rate scheduler')
    parser.add_argument('--check_point', type=bool, default=True,
                        help='save a weight of model during training when a loss of validating is decreased')
    parser.add_argument('--early_stop', type=bool, default=False,
                        help='stop the training of model when a loss of validating is increased')
    parser.add_argument('--img_height', type=int, default=640,
                        help='the size of image height')
    parser.add_argument('--img_width', type=int, default=640, 
                        help='the size of image width')
    parser.add_argument('--train_log_step', type=int, default=100,
                        help='print out the logs of training every steps')
    parser.add_argument('--valid_log_step', type=int, default=20,
                        help='print out the logs of validating every steps')
    return parser

def main(args):
    
    train_data = SemanticSegmentationDataset(
        path=args.data_dir,
        subset='train',
        crop_size=(args.img_height, args.img_width),
        transforms_=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_data = SemanticSegmentationDataset(
        path=args.data_dir,
        subset='valid',
        crop_size=(1920, 1080),
        transforms_=True,
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    bisenetv2 = OurModel(aux_mode='train', weight_path=args.weight_dir, num_classes=args.num_classes)
    summary(bisenetv2, (3, args.img_height, args.img_width), device='cpu')

    model = Trainer(
        model=bisenetv2,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
    )

    history = model.fit(train_loader, valid_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
