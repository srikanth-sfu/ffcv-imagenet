import math

import torch
import torch.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
import timm
from timm.models.layers import BatchNormAct2d, ConvBnAct
from timm.models._efficientnet_blocks import EdgeResidual

def preprocess(module, replacement=nn.ReLU(inplace=True), parent=True):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, replacement)
        else:
            preprocess(child, replacement, parent=False)
    # if parent:
    #     module.classifier = torch.nn.Identity()
    #     module.global_pool = torch.nn.Identity()
    #     module.conv_head = torch.nn.Identity()
    #     module.bn2 = torch.nn.Identity()


class EfficientNetLiteCustom(nn.Module):
    def __init__(
        self, pretrain
    ):
        super(EfficientNetLiteCustom, self).__init__()
        
        # Stem
        #model = timm.create_model('test_efficientnet.r160_in1k', pretrained=pretrain)
        #model = timm.create_model('mobilevit_xxs.cvnets_in1k', pretrained=pretrain, features_only=True)
        model = timm.create_model('test_efficientnet_gn.r160_in1k', pretrained=pretrain)
        self.model = model
        self.preprocess()
       
    
    def preprocess(self):
        model = self.model
        f1, f2 = 20, 28
        model.conv_stem = torch.nn.Conv2d(1,f1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.bn1 = BatchNormAct2d(f1)
        model.blocks[0][0].conv = torch.nn.Conv2d(f1, f1, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False, groups=f1)
        model.blocks[0][0].bn1 = BatchNormAct2d(f1)

        model.blocks[1][0].conv_exp = nn.Conv2d(f1, f1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, groups=f1)
        model.blocks[1][0].bn1 = BatchNormAct2d(f1)
        model.blocks[1][0].conv_pwl = nn.Conv2d(f1, f2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.blocks[1][0].bn2 = BatchNormAct2d(f2)

        model.blocks[2][0].conv_exp = nn.Conv2d(f2, f2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, groups=f2)
        model.blocks[2][0].bn1 = BatchNormAct2d(f2)
        model.blocks[2][0].conv_pwl = nn.Conv2d(f2, f2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=f2)
        model.blocks[2][0].bn2 = BatchNormAct2d(f2)
        
        model.blocks[3][0].conv_pw = nn.Conv2d(f2, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        preprocess(model)
        print(model)
        self.model = model
     
    def forward(self, x):
        x = x.mean(dim=1).unsqueeze(1)
        output = self.model(x)
        return output
