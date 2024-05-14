# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# optimizer
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0001)
# optimizer = dict(_delete_=True, type="AdamW", lr=3e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict()
