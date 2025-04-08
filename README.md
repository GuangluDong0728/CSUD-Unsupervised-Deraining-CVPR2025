# [CVPR2025] Channel Consistency Prior and Self-Reconstruction Strategy Based Unsupervised Image Deraining
This repository is the official implementation of CVPR 2025 "Channel Consistency Prior and Self-Reconstruction Strategy Based Unsupervised Image Deraining". The testing and training code, as well as all the pre-trained weights, have been open sourced!!!

<p align="center">
<a href="https://arxiv.org/abs/2503.18703"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
    
<p align="center">
    <img src="imgs/CSUD.png" width="1000px"/>
</p>

 **[Channel Consistency Prior and Self-Reconstruction Strategy Based Unsupervised Image Deraining](https://arxiv.org/abs/2503.18703)**
 </br>
[Guanglu Dong](https://github.com/GuangluDong0728) $^{1}$,
Tianheng Zheng $^{1}$,
Yuanzhouhan Cao $^{2}$,
Linbo Qing $^{1}$,
Chao Ren $^{1}$ \*

$^{1}$ Sichuan University,
$^{2}$ Beijing Jiaotong University

<p align="center">
<img src="imgs/realshot.png" :height="100px">

# Environment Prepare
You can refer to the environment preparation process of [BasicSR](https://github.com/XPixelGroup/BasicSR), which mainly includes the following two steps:

1. 

    ```bash
    pip install -r requirements.txt
    ```

2. 

    ```bash
    python setup.py develop
    ```

# Downloading Our Weights

1. **Download Pretrained Weights:**
   - Navigate to [this link](https://drive.google.com/drive/folders/1QYcP8mR-18SrXYNn_Tqzel03vaQIKyk5?usp=sharing) to download our weights. Our CSUD is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NAFNet](https://github.com/megvii-research/NAFNet), please use NAFNet (the version with width of 32) to test all of our models.

2. **Save to `experiments` Directory:**
   - Once downloaded, place the weights into the `experiments` directory.
     
# Training and Testing

## Training
To train our model, you need to download the train/test datasets and put them into the `datasets` directory. Then please open the `options/train/train_unsupervised.yml` file and update the paths, and just run the command:

```bash
python basicsr/train_unsupervised.py -opt options/train/train_unsupervised.yml
```

## Testing
To test our model, please open the `options/test/Derain/test_deraining.yml` file and update the paths, and just run the command:

```bash
python basicsr/test.py -opt options/test/Derain/test_deraining.yml
```

# Acknowledgements

This project is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR), and [NAFNet](https://github.com/megvii-research/NAFNet).
