#  Configuration for training the network

_base_ = [
    # MRF-UNet network architecture
    '../_base_/models/mrf_unet.py',
    # FLAIR data loading
    '../_base_/datasets/oem_regionalwise_uda.py',
    ]

# Random seed
seed = 0

# Modify the backbone architecture config
choices = '5,3,1,0,5,1,6,0,2,6,5,4,4,0,0,0,9,0,4,7,0,4,3,0,7,5' # Net-C1 architecture choices
# choices = '5,0,3,6,5,2,5,5,4,3,5,4,4,2,0,3,6,0,3,5,0,4,9,2,6,5' # Net-C2 architecture choices

embedding_size=16
num_classes = 9

model = dict(
     backbone=dict(choices=choices),
     decode_head=dict(
          in_channels=[embedding_size],
          in_index=[0],
          channels=embedding_size,
          num_classes=num_classes,),)

n_gpus = 1

# Data loader
data = dict(workers_per_gpu=1,  samples_per_gpu=1,)