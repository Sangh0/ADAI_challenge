import numpy as np
import torch
import torch.nn.functional as F

class Metrics(object):
    def __init__(self, n_classes=18, dim=1, smooth=1e-10):
        self.n_classes = n_classes
        self.dim = dim
        self.smooth = smooth
    
    @torch.no_grad()
    def mean_iou(self, pred, label):
        if pred.shape[1] != self.n_classes:
            raise ValueError(f'The number of classes does not match, # of classes: {self.n_classes}, pred classes: {pred.shape[1]}')
        
        pred = F.softmax(pred, dim=self.dim)
        pred = torch.argmax(pred, dim=self.dim)
        pred = pred.contiguous().view(-1)

        label = label.contiguous().view(-1)

        iou_per_class = []

        for clas in range(self.n_classes):
            true_class = pred == clas
            true_label = label == clas

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect+self.smooth) / (union+self.smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)

    @torch.no_grad()
    def pixel_acc(self, pred, label):
        pred = F.softmax(pred, dim=self.dim)
        pred = torch.argmax(pred, dim=self.dim)
        
        correct = torch.eq(pred, label).int()
        accuracy = correct.sum() / correct.numel()

        return accuracy
