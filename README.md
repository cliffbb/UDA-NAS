<h1 align="center">Unsupervised Domain Adaptation Architecture Search with Self-Training for Land Cover Mapping</h1>

<p align="justify">We proposed a simple UDA-NAS framework to search for lightweight neural networks for land cover mapping tasks under domain shift. The framework integrates Markov random field neural architecture search into a self-training UDA scheme to search for efficient and effective networks under a limited computation budget.</p> 

The [paper](https://arxiv.org/abs/2404.14704) accepted at 2024 CVPR Workshop. 

<p align="center">
<img src="doc/UDA-NAS1.png" alt="framework fig">
</img>
</p>


### Requirements
The source code depends on the packages in the `requirements.txt` file.

Run the command below to install the packages.
```Shell
pip install -r requirements.txt
``` 

### Datasets
* [OpenEarthMap dataset](https://open-earth-map.org/)

* [FLAIR #1 dataset](https://github.com/IGNF/FLAIR-1)


### Usage
* Searching: Learning pairwise MRF and inference over the learnt MRF via diverse M-best loopy inference


* Training:  The found architectures are re-trained to to select optimal solution.



### Citation
```BibTeX
@InProceedings{CliffUDA-NAS2024,
  title     = {Unsupervised Domain Adaptation Architecture Search with Self-Training for Land Cover Mapping},
  author    = {Clifford Broni-Bediako, Junshi Xia, Naoto Yokoya},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW)},
  year      = {2024}
}
```

### Acknowledgement
<p align="justify">This code is heavily borrowed from <a href="https://github.com/zifuwanggg/MRF-UNets">MRF-UNet</a> and <a href="https://github.com/lhoyer/DAFormer?tab=readme-ov-file">DAFormer</a>. Thanks to the authors for making their code publically available.</p>


### License
<p align="justify">This work is licensed under the MIT License, however, please refer to the licences of the <a href="https://github.com/zifuwanggg/MRF-UNets">MRF-UNet</a> and <a href="https://github.com/lhoyer/DAFormer?tab=readme-ov-file">DAFormer</a> if you are using this code for commercial matters.</p>