# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:  
#   Add MRF_UNet NAS support options

# DACS_NAS UDA
uda = dict(
    type='DACS_NAS',
    source_only=False,
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False,
    # MRF_UNet NAS
    long_burnin=10000,
    short_burnin=10,
    tau=1.0,
)
use_ddp_wrapper = True
