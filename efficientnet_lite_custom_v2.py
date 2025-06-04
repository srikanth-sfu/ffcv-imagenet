from urllib.request import urlopen
from PIL import Image
import timm
from timm.models.layers import BatchNormAct2d, ConvBnAct
from timm.models._efficientnet_blocks import EdgeResidual
import torch
nn = torch.nn

#img = Image.open(urlopen(
#    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
#))
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

model = timm.create_model('test_efficientnet_gn.r160_in1k', pretrained=False)
model = model.eval()
f1, f2 = 12, 24
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
checkpoint = torch.load('a2c8eee2-9c46-4b9c-bf10-3ac14361952e/final_weights.pt', map_location='cpu')

# Get state dict (adjust key if needed)
state_dict = checkpoint.get('state_dict', checkpoint)

# Strip known prefixes
def strip_prefix(state_dict, prefix_to_strip):
    state_dict = {k.replace('conv_exp.conv', 'conv_exp'): v for k,v in state_dict.items()}
    state_dict = {k.replace('conv_dw.conv', 'conv_dw'): v for k,v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if "blur_filter" not in k}
    return {
        k[len(prefix_to_strip):] if k.startswith(prefix_to_strip) else k: v
        for k, v in state_dict.items()
    }


# Try common prefixes
state_dict = strip_prefix(state_dict, 'module.model.')


model.load_state_dict(state_dict)
os._exit(1)
preprocess(model)
# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
data_config['crop_pct'] = 1
data_config['input_size'] = (1,256,192)
transforms = timm.data.create_transform(**data_config, is_training=False)
model.classifier = torch.nn.Identity()
model.global_pool = torch.nn.Identity()
model.conv_head = torch.nn.Identity()
model.bn2 = torch.nn.Identity()
#model.blocks = model.blocks[:-2]
model.blocks = model.blocks[:3] 
img = torch.autograd.Variable(
        torch.randn(1, 256, 192)
    )
output = model(img.unsqueeze(0))  # unsqueeze single image into batch of 1

torch.onnx.export(
    model, 
    (img).unsqueeze(0), 
    "model.onnx", 
    export_params=True, 
    opset_version=11,            # Can be 11–17; 12+ is safe for most use cases
    do_constant_folding=True,    # Fold constant ops for optimization
    input_names=['input'], 
    output_names=['output'],
)

