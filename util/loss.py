import torch
import torch.nn as nn

class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float())).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduce='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() //  16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)