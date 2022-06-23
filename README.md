## [R-Net: Recursive Decoder with Edge Refinement Network for Salient Object Detection]
by XX

## Introduction
Fully convolutional neural (FCN) networks based on encoder-decoder structures are adopted for salient object detection (SOD) and achieve state-of-the-art performance. However, most of the previous works aggregate multilevel features layer by layer and ignore long-range dependencies of spatial context, leading to imbalanced feature processes and context separation. In addition, the burry edges obtained from previous SOD methods inevitably influence the detection accuracy. To handle these problems, this study proposes an effective recursion-based SOD network R-Net adopting a cascade structure, which contains a recursive decoder module (RDM), a long-range dependency module (LRDM), and an edge refinement module (ERM). The RDM can scale different layer features to the same size and realize multiple feature fusion while recovering data layer by layer. The LRDM bridges the encoder and decoder by weighting multilevel features to establish the long-range feature dependency of the spatial context. And the ERM attached to the decoder introduces the shallow reference feature to refine the burry edges for obtaining delicate detection results. The SOD experimental results on DUTS-TE, HKU-IS, PASCAL-S, ECSSD, and DUT-OMRON datasets demonstrate that the proposed R-Net is more robust under different complex scenes compared with some state-of-the-art methods.


## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.2](http://pytorch.org/)
- [OpenCV 4.5.3.56](https://opencv.org/)
- [Numpy 1.19.5](https://numpy.org/)
- [pillow 8.3.2](https://pypi.org/project/Pillow/)


## Clone repository

```shell
git clone git@github.com:ZEROICEWANG/R-Net.git
cd R-Net/
```

## Download dataset

Download the following datasets and unzip them into `data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)


## Download model

- If you want to test the performance of R-Net, please download the model([Baidu](https://pan.baidu.com/s/14vwSXzfG_FtJ3fLN2jUmSg?pwd=ruk0) [Google](https://drive.google.com/file/d/114plxfFnGQIVlhDBVyRFcaiq7m1q6Ojz/view?usp=sharing)) into `models/R-Model` folder


## Training

```shell
    python3 train.py
```

## Testing

```shell
    python3 predict.py
```
- After testing, saliency maps of `PASCAL-S`, `ECSSD`, `HKU-IS`, `DUT-OMRON`, `DUTS-TE` will be saved in `predict_result/R-Net/` folder.

## Saliency maps & Trained model
- saliency maps: [Baidu](https://pan.baidu.com/s/16i-zyViOvgn9APK4iTWEMw?pwd=eic1) [Google](https://drive.google.com/file/d/15zvZBk1_MUsg3mjOW7daenDpNjEfpCv0/view?usp=sharing)
- trained model: [Baidu](https://pan.baidu.com/s/14vwSXzfG_FtJ3fLN2jUmSg?pwd=ruk0) [Google](https://drive.google.com/file/d/114plxfFnGQIVlhDBVyRFcaiq7m1q6Ojz/view?usp=sharing)
