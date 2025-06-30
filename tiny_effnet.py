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

def TinyEffNet(width_mult=1.0, depth_mult=1.0):
    model = _gen_efficientnet('efficientnet_b0', channel_multiplier=width_mult, depth_multiplier=depth_mult)
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

