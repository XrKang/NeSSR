# NeSSR

****
> [Continuous Spatial-Spectral Reconstruction via Implicit Neural Representation](https://link.springer.com/article/10.1007/s11263-024-02150-3), in *International Journal of Computer Vision* (IJCV) 2024.
> Ruikang Xu, Mingde Yao, Chang Chen, Lizhi Wang, and Zhiwei Xiong. 

## Datesets

Prepare the ICVL dataset for training, it can be downloaded from this [link](https://icvl.cs.bgu.ac.il/hyperspectral/).

Processed test dataset for inference can be downloaded from [BaiduYun](https://pan.baidu.com/s/1jCoAI96V73lwVp1DplYmjA?pwd=ijcv) (Access code: ijcv).


****

## Dependencies
* Python 3.8.8, PyTorch 1.8.0, torchvision 0.9.0.
* NumPy 1.24.2, OpenCV 4.7.0, Tensorboardx 2.5.1, Pillow, Imageio. 
****

## Quick Start

### Spectral Reconstruction from RGB Images with Arbitrary Spectral Resolution 

* Download the pre-trained model from [BaiduYun](https://pan.baidu.com/s/1jCoAI96V73lwVp1DplYmjA?pwd=ijcv) (Access code: ijcv).



* Inference with pre-trained model:
```
cd ./SpectralRec && python test.py
```

### Spatial Reconstruction from RGB-HSI Pairs with Arbitrary Spatial Resolution 

* Download the pre-trained model from [BaiduYun](https://pan.baidu.com/s/1jCoAI96V73lwVp1DplYmjA?pwd=ijcv) (Access code: ijcv). 

* Inference with pre-trained model:
```
cd ./SpatialRec && python test.py
```

### Spatial-Spectral Reconstruction from RGB-MSI Pairs with Arbitrary Spatial-Spectral Resolution 

* Download the pre-trained model from [BaiduYun](https://pan.baidu.com/s/1jCoAI96V73lwVp1DplYmjA?pwd=ijcv) (Access code: ijcv).

* Inference with pre-trained model:
```
cd ./JointRec && python test.py
```

****
## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn.

****


## Citation
If you find our work helpful, please cite the following paper.
```
@article{xu2024continuous,
  title={Continuous Spatial-Spectral Reconstruction via Implicit Neural Representation},
  author={Xu, Ruikang and Yao, Mingde and Chen, Chang and Wang, Lizhi and Xiong, Zhiwei},
  journal={International Journal of Computer Vision},
  pages={1--23},
  year={2024},
  publisher={Springer}
}
```
