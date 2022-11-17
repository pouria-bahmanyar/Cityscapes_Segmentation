import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
from torch import nn
import torchvision.transforms.functional as TF

def custom_DeepLabv3_50(out_channel=20):
  model = deeplabv3_resnet50(pretrained=True, progress=True)
  model.classifier = DeepLabHead(2048, out_channel)

  #Set the model in training mode
  model.train()
  return model

def custom_DeepLabv3_101(out_channel=20):
  model = deeplabv3_resnet101(pretrained=True, progress=True)
  model.classifier = DeepLabHead(2048, out_channel)

  #Set the model in training mode
  model.train()
  return model