import torch.nn as nn

from .mrf_unet_api import MRFSuperNet, initialize
from mmseg.models.builder import BACKBONES

    
@BACKBONES.register_module()
class MRFUNetSearch(nn.Module):
    def __init__(self, image_channels, embedding_size, channel_step, **kwargs):
        super(MRFUNetSearch, self).__init__(**kwargs)
        
        self.mrfunet = MRFSuperNet(image_channels=image_channels, 
                                   num_classes=embedding_size, 
                                   channel_step=channel_step)
        self.mrfunet.head = nn.Identity()
        initialize(self.mrfunet)
        
    def forward(self, x, choices_one_hot):       
        out = self.mrfunet(x, choices_one_hot=choices_one_hot)
        return [out]
