# model configuration
norm_cfg = dict(type="BN", requires_grad=True)   # using one GPU, BN is used instead of SyncBN
embedding_size = 48
num_classes = 9
model = dict(
     type="EncoderDecoderSearch",
     pretrained=None,
     backbone=dict(
          type="MRFUNetSearch",
          image_channels=3,
          embedding_size=embedding_size,
          channel_step=5),
     decode_head=dict(type="LinearHead",
          in_channels=[embedding_size],
          in_index=[0],
          input_transform="resize_concat",
          channels=embedding_size,
          dropout_ratio=0,
          num_classes=num_classes,
          norm_cfg=norm_cfg,
          align_corners=False,
          loss_decode=dict(type="RecallCrossEntropyLoss", num_classes=num_classes),),
          # loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)),
     test_cfg = dict(mode="whole"),
     train_cfg = dict(),
     init_cfg=dict() 
)
