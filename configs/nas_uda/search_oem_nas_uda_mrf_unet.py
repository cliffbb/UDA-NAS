# Configuration for architecture search

_base_ = [
    '../_base_/default_runtime.py',
    # MRF-UNet network architecture
    '../_base_/models/mrf_unet_search.py',
    # OEM data loading
    '../_base_/datasets/oem_regionalwise_uda.py',
    # DACS UDA self-training
    '../_base_/uda/dacs_nas.py',
    # AdamW optimizer
    '../_base_/schedules/adamw.py',
    # Linear learning rate warmup
    '../_base_/schedules/poly10warm.py']

# Random seed
seed = 0

# Modify the DACS_NAS confgi
uda = dict(
    # Consistency regularizer
    consistency_regularizer='confidence_based',  # ['confidence_based' | 'energy_based']
    # MRF options
    long_burnin=10000,
    short_burnin=100,
    tau=1,
    # Increased alpha
    alpha=0.999,
    # Pseudo-label crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    # pallete
    cmap = 'oem',)

# Data loader 
data = dict(workers_per_gpu=1, samples_per_gpu=8,)

# Optimizer hyperparameters
optimizer_config = None
optimizer = dict(_delete_=True, type="AdamW", lr=3e-3, betas=(0.9, 0.999), weight_decay=0.05)
lr_config = dict(_delete_=True,
                 policy="poly",
                 warmup="linear",
                 warmup_iters=1500,
                 warmup_ratio=1e-5, 
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False
                 )

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=140000)

# Logging configuration
checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=1) 
evaluation = dict(interval=4000, metric='mIoU') 
