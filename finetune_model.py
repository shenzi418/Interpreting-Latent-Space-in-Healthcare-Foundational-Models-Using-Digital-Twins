import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from net1d import Net1D, MultiHeadECGFounder

import torch.nn as nn
import torch


def ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=False):
  model = Net1D(
      in_channels=12, 
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
  # freezing model
  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
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
    linear_prob=False,
):
  """
  Build a MultiHeadECGFounder model:

  - Loads encoder weights from the original ECGFounder checkpoint (pth).
  - Strips 'dense.' weights as in ft_12lead_ECGFounder.
  - Optionally freezes the encoder if linear_prob=True (only heads trainable).
  """
  model = MultiHeadECGFounder(
      in_channels=12, 
      base_filters=64, #32 64
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
      verbose=False,
  )

  checkpoint = torch.load(pth, map_location=device)
  state_dict = checkpoint.get('state_dict', checkpoint)
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
  # Debug if needed:
  # print("Missing:", missing, "Unexpected:", unexpected)

  if linear_prob:
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False

  model.to(device)

  return model

