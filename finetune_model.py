import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from net1d import Net1D, MultiHeadECGFounder, ConvAdapter1D

import torch.nn as nn
import torch


def freeze_backbone_except_adapters(backbone):
  # freeze everything
  for p in backbone.parameters():
      p.requires_grad = False
  # unfreeze adapters
  for m in backbone.modules():
      if isinstance(m, ConvAdapter1D):
          for p in m.parameters():
              p.requires_grad = True


def ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=False,
                         use_adapter=False, adapter_reduction=16,
                         adapter_dropout=0.0):
  model = Net1D(
      in_channels=12, 
      base_filters=64,
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],
      m_blocks_list=[2,2,2,3,3,4,4],
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes,
      use_adapter=use_adapter,
      adapter_reduction=adapter_reduction,
      adapter_dropout=adapter_dropout)

  checkpoint = torch.load(pth, map_location=device, weights_only=False)
  state_dict = checkpoint.get('state_dict', checkpoint)

  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  model.load_state_dict(state_dict, strict=False)

  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)

  if linear_prob:
    if use_adapter:
      freeze_backbone_except_adapters(model)
    else:
      for name, param in model.named_parameters():
          if 'dense' not in name:
              param.requires_grad = False

  model.to(device)

  return model


def ft_1lead_ECGFounder(device, pth, n_classes,linear_prob=False):
  model = Net1D(
      in_channels=1, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  checkpoint = torch.load(pth, map_location=device)
  state_dict = checkpoint['state_dict']

  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  model.load_state_dict(state_dict, strict=False)

  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)

  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
            param.requires_grad = False
  model.to(device)

  return model


def ft_multihead_ECGFounder(
    device,
    pth,
    n_medal_classes,
    n_ptb_classes,
    n_theta=0,
    physics_hidden=0,
    physics_dropout=0.0,
    linear_prob=False,
    use_adapter=True,
):
  model = MultiHeadECGFounder(
      in_channels=12, 
      base_filters=64,
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],
      m_blocks_list=[2,2,2,3,3,4,4],
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      use_bn=False,
      use_do=False,
      n_medal_classes=n_medal_classes,
      n_ptb_classes=n_ptb_classes,
      n_theta=n_theta,
      physics_hidden=physics_hidden,
      physics_dropout=physics_dropout,
      verbose=False,
      use_adapter=use_adapter,
  )

  checkpoint = torch.load(pth, map_location=device, weights_only=False)
  state_dict = checkpoint.get('state_dict', checkpoint)
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)

  if linear_prob:
    if use_adapter:
      freeze_backbone_except_adapters(model.backbone)
    else:
      for p in model.backbone.parameters():
          p.requires_grad = False

  model.to(device)

  return model

