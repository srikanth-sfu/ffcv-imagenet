import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, RandomResizedCrop, RandomHorizontalFlip
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from timm.models.efficientnet import _create_effnet, _gen_test_efficientnet, _gen_efficientnet 
from functools import partial
from timm.models._efficientnet_builder import decode_arch_def, round_channels
import timm

# EfficientNet Inverse Scaling Block
class SimpleEffNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

def replace_layer(layer):
    layer.out_channels = layer.in_channels = max(layer.in_channels, layer.out_channels)
    layer.groups = layer.in_channels
    old_conv = layer
    layer = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=layer.groups,  # <-- your intended group count
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode
    )
    return layer
    

def replace_block(model, blockid):
    block = old_block = model.blocks[blockid]
    for i in range(len(block)):
        block[i].se.conv_reduce = replace_layer(block[i].se.conv_reduce)
        block[i].se.conv_expand = replace_layer(block[i].se.conv_expand) 
    return block

def replace_blocks(model, start, end):
    for block in range(start,end):
        model.blocks[block] = replace_block(model, block)
    return model 
def TinyEffNet(width_mult=1.0, depth_mult=1.0):
    model = _gen_efficientnet('efficientnet_b0', channel_multiplier=width_mult, depth_multiplier=depth_mult, pretrained=False)
    model = replace_blocks(model,4,7)
    print(model)
    return model 

# === FFCV ImageNet loader ===
def build_ffcv_loader(ffcv_path, batch_size, num_workers=4, device='cuda'):
    image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
    ]

    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(torch.device(device), non_blocking=True)]

    loader = Loader(
        ffcv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        pipelines = {
            'image': image_pipeline,
            'label': label_pipeline
        }
    )
    return loader


if __name__ == "__main__":
	# === 4 Configs ===
	configs = [
		{"name": "effnet_inv0", "phi": 0},
		{"name": "effnet_inv1", "phi": 1},
		{"name": "effnet_inv2", "phi": 2},
		{"name": "effnet_inv3", "phi": 3},
	]

	alpha, beta = 1.2, 1.1

	models = {}
	for cfg in configs:
		phi = cfg["phi"]
		width_mult = alpha ** (-phi)
		depth_mult = beta ** (-phi)
		model = TinyEffNet(width_mult=width_mult, depth_mult=depth_mult)
		models[cfg["name"]] = model

