from mmdet.apis import init_detector, inference_detector
import torch

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config_file = 'configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
model = init_detector(config_file, device=device)  # or device='cuda:0'
for param in model.backbone.parameters():
    param.requires_grad = False
print(model)