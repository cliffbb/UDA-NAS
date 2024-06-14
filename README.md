<h1 align="center">Unsupervised Domain Adaptation Architecture Search with Self-Training for Land Cover Mapping</h1>

<p align="justify">We proposed a simple UDA-NAS framework to search for lightweight neural networks for land cover mapping tasks under domain shift. The framework integrates Markov random field neural architecture search into a self-training UDA scheme to search for efficient and effective networks under a limited computation budget.</p> 

The [paper]([https://arxiv.org/abs/2404.14704](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Broni-Bediako_Unsupervised_Domain_Adaptation_Architecture_Search_with_Self-Training_for_Land_Cover_CVPRW_2024_paper.html)) is accepted at 2024 CVPR Workshop. 
<p>&nbsp;</p>

<h1 align="center">
<img src="doc/UDA-NAS1.png" alt="framework fig">
</img>
</h1>


### Requirements
The source code depends on the packages in the `requirements.txt` file.

Run the command below to install the packages.
```Shell
pip install -r requirements.txt
``` 

### Datasets
* Download the [OpenEarthMap dataset](https://open-earth-map.org/) and based on the regional-wise UDA settings in the [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.pdf) organise the folder structure as follows: 
```
data
|- OpenEarthMap
|  |- images
|  |  |- aachen 
|  |  |- abancay
|  |  |- ...
|  |  |- ...
|  |  |- ...
|  |  |- zachodniopomorskie
|  |  |- zanzibar
|  |- labels
|  |  |- aachen 
|  |  |- abancay
|  |  |- ...
|  |  |- ...
|  |  |- ...
|  |  |- zachodniopomorskie
|  |  |- zanzibar
|  |- splits
|  |  |- source.txt
|  |  |- target_train.txt
|  |  |- target_test.txt
|  |  |- target_val.txt
```

* Download the [FLAIR #1 dataset](https://ignf.github.io/FLAIR/) and based on the UDA settings in [GeoMultiTaskNet](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Marsocci_GeoMultiTaskNet_Remote_Sensing_Unsupervised_Domain_Adaptation_Using_Geographical_Coordinates_CVPRW_2023_paper.pdf) organise the folder structure as follows:
```
data
|- FLAIR1
|  |- images
|  |  |  |- D006_2020 
|  |  |  |- D008_2019
|  |  |  |- ...
|  |  |  |- ...
|  |  |  |- ...
|  |  |  |- D083_2020
|  |  |  |- D085_2019
|  |- labels
|  |  |  |- D006_2020 
|  |  |  |- D008_2019
|  |  |  |- ...
|  |  |  |- ...
|  |  |  |- ...
|  |  |  |- D083_2020
|  |  |  |- D085_2019
|  |- splits
|  |  |- source.txt
|  |  |- target_train.txt
|  |  |- target_test.txt
|  |  |- target_val.txt
```


### Usage
##### Searching
* Learning pairwise MRF. 
```
  python tools/train.py configs/nas_uda/search_oem_nas_uda_mrf_unet.py 
```
* Inference over the learnt MRF for different architecture `choices` config.
```
 python tools/inference.py --ckp-path /path/to/search/checkpoint.pth
```

##### Training
Training the found architectures to select optimal solution. Modify the architecture `choices` config in the config file and run the commands below to train the network.
```
python tools/train.py configs/nas_uda/ \
    train_oem_nas_uda_mrf_unet_confidence_based.py

python tools/train.py configs/nas_uda/ \
    train_flair_nas_uda_mrf_unet_confidence_based.py
```

##### Testing and Pretrained network
 Download pretrained weights of the found networks on [OpenEarthMap](https://drive.google.com/file/d/1b7lO2WHOKbgKvhKPd2iEd80F-XGDNcTR/view?usp=sharing) and [ FLAIR #1](https://drive.google.com/file/d/1dTpu2-phL00mBqVnNumfODTO650yE4-n/view?usp=sharing), unzip them into the `pretrained` folder and run the commands below.
```
python tools/test.py \
    configs/nas_uda/test_oem_nas_uda_mrf_unet_confidence_based.py \
    pretrained/openearthmap/net_c_1.pth \
    --test-set --eval mIoU --show-dir results

python tools/test.py \
    configs/nas_uda/test_flair_nas_uda_mrf_unet_confidence_based.py \
    pretrained/openearthmap/net_c_1.pth \
    --test-set --eval mIoU --show-dir results
```

### Citation
```BibTeX
@InProceedings{Broni-Bediako_2024_CVPR,
    author    = {Broni-Bediako, Clifford and Xia, Junshi and Yokoya, Naoto},
    title     = {Unsupervised Domain Adaptation Architecture Search with Self-Training for Land Cover Mapping},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {543-553}
}
```

### Acknowledgement
<p align="justify">This code is heavily borrowed from <a href="https://github.com/zifuwanggg/MRF-UNets">MRF-UNet</a> and <a href="https://github.com/lhoyer/DAFormer?tab=readme-ov-file">DAFormer</a>. Thanks to the authors for making their code publically available.</p>


### License
<p align="justify">This work is licensed under the MIT License, however, please refer to the licences of the <a href="https://github.com/zifuwanggg/MRF-UNets">MRF-UNet</a> and <a href="https://github.com/lhoyer/DAFormer?tab=readme-ov-file">DAFormer</a> if you are using this code for commercial matters.</p>
