import math

import torch
import torch.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
import timm


def preprocess(module, replacement=nn.ReLU(inplace=True), parent=True):
    # for name, child in module.named_children():
    #     if isinstance(child, nn.SiLU):
    #         setattr(module, name, replacement)
    #     else:
    #         preprocess(child, replacement, parent=False)
    # if parent:
    #     module.classifier = torch.nn.Identity()
    #     module.global_pool = torch.nn.Identity()
    #     module.conv_head = torch.nn.Identity()
    #     module.bn2 = torch.nn.Identity()
    pass

class EfficientNetLiteCustom(nn.Module):
    def __init__(
        self, pretrain
    ):
        super(EfficientNetLiteCustom, self).__init__()
        
        # Stem
        #model = timm.create_model('test_efficientnet.r160_in1k', pretrained=pretrain)
        #model = timm.create_model('mobilevit_xxs.cvnets_in1k', pretrained=pretrain, features_only=True)
        model = timm.create_model('test_efficientnet_gn.r160_in1k', pretrained=pretrain, features_only=True)
        old_weight = model.conv_stem.weight.data.mean(dim=1).unsqueeze(1)
        new_layer = nn.Conv2d(1,24,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        with torch.no_grad():
            new_layer.weight.copy_(old_weight)
        model.conv_stem = new_layer
        preprocess(model)
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        return output[-3:-2]
