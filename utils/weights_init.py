import torch
import torch.nn as nn

def weights_init(m, mean=0.0, std=0.02):
  if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
      torch.nn.init.normal_(m.weight, mean, std)