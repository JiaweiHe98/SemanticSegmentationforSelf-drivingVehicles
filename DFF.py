from .base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from utils.helpers import set_trainable
from itertools import chain
from torchsummary import summary

class ResNet(nn.Module):

  def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
    super(ResNet, self).__init__()

    # only output strides of 8 or 16 are supported
    assert output_stride in [8,16]

    # get a attribute 'backbone' from class 'models'
    # The same as model = models.resnet101(True)
    model = getattr(models, backbone)(pretrained)

    # The first conv2d layer 
    self.layer0 = nn.Sequential(*list(model.children())[:4])
    # sequential layers
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4

    if output_stride == 16:
      s3,s4,d3,d4 = (2,1,1,2)
    elif output_stride == 8:
      s3,s4,d3,d4 = (1,1,2,4)
    
    if output_stride == 8:
      for name,model in self.layer3.named_modules():
        if 'conv1' in name and (backbone == 'resnet34' or backbone == 'resnet18'):
          model.dilation, model.padding, model.stride = (d3,d3), (d3,d3), (s3,s3)
        elif 'conv2' in name:
          model.dilation, model.padding, model.stride = (d3,d3), (d3,d3), (s3,s3)
        elif 'downsample.0' in name:
          model.stride = (s3,s3)
    
    for name,model in self.layer4.named_modules():
      if 'conv1' in name and (backbone == 'resnet34' or backbone == 'resnet18'):
        model.dilation, model.padding, model.stride = (d4,d4), (d4,d4), (s4,s4)
      elif 'conv2' in name:
        model.dilation, model.padding, model.stride = (d4,d4), (d4,d4), (s4,s4)
      elif 'downsample.0' in name:
        model.stride = (s4, s4)

  def forward(self,x):
    x = self.layer0(x)
    x = self.layer1(x)
    # Deep Feature Flow doesn't use low_level_features at all
    low_level_features = x
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # result shape is [-1,2048,Height/16,Width/16]
    return x, low_level_features

def assp_branch(in_channels, out_channles, kernel_size, dilation):
  padding = 0 if kernel_size == 1 else dilation
  return nn.Sequential(
          nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
          nn.BatchNorm2d(out_channles),
          nn.ReLU(inplace=True))