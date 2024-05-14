# model configuration
norm_cfg = dict(type="BN", requires_grad=True)   # using one GPU, BN is used instead of SyncBN
model = dict(
     type="EncoderDecoder",
     pretrained=None,
     backbone=dict(
          type="MRFUNet",
          image_channels=3,
          embedding_size=48, 
          channel_step=5),
     decode_head=dict(type="LinearHead",
          input_transform="resize_concat",
          dropout_ratio=0,
          norm_cfg=norm_cfg,
          align_corners=False,),
     test_cfg = dict(mode="whole"),
     train_cfg = dict(),
     init_cfg=dict() 
)
