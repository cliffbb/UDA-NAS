#######
from mmcv.runner.optimizer import (
    OPTIMIZER_BUILDERS, DefaultOptimizerConstructor, OPTIMIZERS)
from mmcv.utils import build_from_cfg
from mmseg.utils import get_root_logger


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg, paramwise_cfg=paramwise_cfg)
    
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        # if not self.paramwise_cfg:
        params = []
        params.append({'params': model.parameters()})
        params.append({'params': model.potentials()})
        
        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
