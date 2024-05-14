from .builder import DATASETS
from .custom import CustomDataset

import os.path as osp


@DATASETS.register_module()
class FLAIRDataset(CustomDataset):
    """FLAIR dataset
    ``img_suffix`` and ``seg_map_suffix`` are set to ".tif" 
    """
    CLASSES = ('building', 'pervious surface', 'impervious surface', 'bare soil', 
               'water', 'coniferous', 'deciduous', 'brushwood', 'vineyard',
               'herbaceous vegetation', 'agricultural land', 'plowed land', 'other')
    
    PALETTE = [[241, 91, 181], [229, 228, 233], [157, 2, 8], [55, 7, 23], [68, 96, 239],
               [18, 42, 18], [81, 183, 136], [232, 92, 4], [113, 9, 182], [79, 119, 46],
               [255, 242, 13], [244, 140, 5], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(FLAIRDataset, self).__init__(img_suffix=".tif", 
                                        seg_map_suffix=".tif",
                                        # reduce_zero_label=False,
                                        **kwargs)
        
        assert osp.exists(self.img_dir) and osp.exists(self.ann_dir) and self.split is not None

