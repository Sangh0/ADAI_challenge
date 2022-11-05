import sys
sys.path.append('/home/hoo7311/anaconda3/envs/pytorch/lib/python3.8/site-packages')
import os
import argparse
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import OurModel
from util.metric import Metrics
from util.loss import OhemCELoss
from util.transform_pillow import UnNormalize
from dataset import SemanticSegmentationDataset, EvalDataset


class Evaluation(object):

    def __init__(
        self,
        path,
        weight_path,
        batch_size,
        num_classes,
        preprocess=False,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device} ready...')
        
        if preprocess:
            self.dataloader = DataLoader(
                EvalDataset(path=path),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        else:
            self.dataloader = DataLoader(
                SemanticSegmentationDataset(path=path, subset='valid'),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        self.model = OurModel(aux_mode='train', weight_path=None, num_classes=num_classes)
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
        print('model ready...')

        self.metric = Metrics(n_classes=num_classes, dim=1)
        print('mean iou calculator ready...')
        
        self.un_normalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        self.labels_info = {
            0: [255, 0, 0],
            1: [128, 0, 0],
            2: [255, 255, 0],
            3: [128, 128, 0],
            4: [0, 255, 0],
            5: [0, 128, 0],
            6: [0, 255, 255],
            7: [0, 128, 128],
            8: [0, 0, 255],
            9: [0, 0, 128],
            10: [255, 0, 255],
            11: [128, 0, 128],
            12: [255, 127, 80],
            13: [184, 134, 11],
            14: [127, 255, 0],
            15: [0, 191, 255],
            16: [255, 192, 203],
            17: [165, 42, 42],
        }

   
    @torch.no_grad()
    def test(self):
        image_list, label_list, output_list = [], [], []
        miou_list = []
        self.model.eval()
        batch_loss, batch_miou = 0, 0

        start = time.time()

        for batch, (images, labels) in enumerate(tqdm(self.dataloader)):
            images, labels = images.to(self.device), labels.to(self.device)
            image_list.append(images)
            label_list.append(labels)
            
            outputs, _, _, _, _ = self.model(images)
            output_list.append(outputs)
            miou = self.metric.mean_iou(outputs, labels)
            miou_list.append(miou.item())
            batch_miou += miou.item()

        end = time.time()
        print(f'time: {end-start:.2f}s')
        print(f'mean iou: {batch_miou/(batch+1):.3f}')
        return {
            'image': torch.cat(image_list, dim=0), 
            'label': torch.cat(label_list, dim=0), 
            'output': torch.cat(output_list, dim=0),
            'miou': miou_list,
        }
    
    
    def label2color(self, labels):
        B, H, W = labels.size()
        image = np.zeros(shape=(B, H, W, 3), dtype=np.int32)
        for i in self.labels_info.keys():
            image[(labels.unsqueeze(dim=0)==i).all(axis=0) = self.labels_info[i]]
        return image
    

    def visualize(self, images, labels, outputs, mious, counts, save=False):
        if save:
            folder = './figures'
            os.makedirs(folder, exist_ok=True)
        
        for i in range(counts):
            rgb_output = torch.argmax(outputs[i], dim=0)
            fig, ax = plt.subplots(2, 2, figsize=(20,12))
            fig.suptitle(f'Mean IOU score: {mious[i]}*100:.2f', sioze=20)
            ax[0,0].imshow(self.un_normalize(images[i]).permute(1,2,0))
            ax[0,0].axis('off')
            ax[0,0].set_title('Input Image')
            ax[0,1].imshow(self.label2color(labels[i]))
            ax[0,1].axis('off')
            ax[0,1].set_title('Label Image')
            ax[1,0].imshow(self.label2color(rgb_output))
            ax[1,0].axis('off')
            ax[1,0].set_title('Output Image')
            ax[1,1].imshow(self.un_normalize(images[i]).permute(1,2,0))
            ax[1,1].imshow(self.label2color(rgb_output), alpha=0.5)
            ax[1,1].axis('off')
            ax[1,1].set_title('Overlay Image')
            plt.show()
            
            if save:
                plt.savefig(folder+f'/figure_{i+1}.png')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate Model', add_help=False)
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='the directory of weight of pre-trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the directory where your dataset is located')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='the number of classes in dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--data_preprocess', type=bool, default=False,
                        help='data preprocessing')
    return parser


def main(args):
        
    eval = Evaluation(
        path=args.data_dir, 
        weight_path=args.weight_dir, 
        batch_size=args.batch_size,
        preprocess=args.data_preprocess,
    )
    eval.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
