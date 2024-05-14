import numpy as np
import torch.nn as nn

from .mrf_unet_api import ChildNet, initialize
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class MRFUNet(nn.Module):
    def __init__(self, image_channels, embedding_size, channel_step, choices, **kwargs):
        super(MRFUNet, self).__init__(**kwargs)
        
        choices = np.array([int(c) for c in choices.split(',')])      
        self.childnet = ChildNet(image_channels=image_channels, 
                                 num_classes=embedding_size, 
                                 channel_step=channel_step, 
                                 choices=choices)
        self.childnet.head = nn.Identity()
        initialize(self.childnet)

    def forward(self, x):
        out = self.childnet(x)
        return [out]
  