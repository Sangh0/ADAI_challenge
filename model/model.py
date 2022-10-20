import torch
import torch.nn as nn

from .model_utils import DetailBranch, SemanticBranch, BGALayer, SegHead


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        assert aux_mode in ('train', 'eval', 'pred')
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SemanticBranch()
        self.bga = BGALayer()

        self.head = SegHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == 'train':
            self.aux2 = SegHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegHead(128, 128, n_classes, up_factor=32)

        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head)
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            return logits,
        elif self.aux_mode == 'pred':
            pred = logits.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class OurModel(nn.Module):

    def __init__(self, weight_path, num_classes=28):
        super(OurModel, self).__init__()
        self.model = BiSeNetV2(n_classes=19)
        self.model.load_state_dict(torch.load(weight_path))
        
        input_head = self.model.head.conv_out[1].in_channels
        self.model.head.conv_out[1] = nn.Conv2d(input_head, num_classes, kernel_size=1, stride=1)
        
        input_aux2 = self.model.aux2.conv_out[1].in_channels
        self.model.aux2.conv_out[1] = nn.Conv2d(input_aux2, num_classes, kernel_size=1, stride=1)
        
        input_aux3 = self.model.aux2.conv_out[1].in_channels
        self.model.aux3.conv_out[1] = nn.Conv2d(input_aux3, num_classes, kernel_size=1, stride=1)
        
        input_aux4 = self.model.aux2.conv_out[1].in_channels
        self.model.aux4.conv_out[1] = nn.Conv2d(input_aux4, num_classes, kernel_size=1, stride=1)
        
        input_aux5_4 = self.model.aux2.conv_out[1].in_channels
        self.model.aux5_4.conv_out[1] = nn.Conv2d(input_aux5_4, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        return self.model(x)