<div align="center">

  ## [Enhanced Generative Structure Prior for Chinese Text Image Super-Resolution](https://arxiv.org/pdf/2508.07537)

[Xiaoming Li](https://csxmli2016.github.io/), [Wangmeng Zuo](https://scholar.google.com/citations?hl=en&user=rUOpCEYAAAAJ&view_op=list_works), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)


S-Lab, Nanyang Technological University


<img src="./Imgs/pipeline.png" width="800px">
<p align="center"> The whole framework of our MARCONet++ </p>

<!-- 
<p align="justify">Faithful text image super-resolution (SR) is challenging because each character has a unique structure and usually exhibits diverse font styles and
layouts. While existing methods primarily focus on English text, less attention has been paid to more complex scripts like Chinese. In this paper, we introduce a high-quality text image SR framework designed to restore the precise strokes of low-resolution (LR) Chinese characters. Unlike methods that rely on character recognition priors to regularize the SR task, we propose a novel structure prior that offers structure-level guidance to enhance visual quality. Our framework incorporates this structure prior within a StyleGAN model, leveraging its generative capabilities for restoration. To maintain the integrity of character structures while accommodating various font styles and layouts, we implement a codebook-based mechanism that restricts the generative space of StyleGAN. Each code in the codebook represents the structure of a specific character, while the vector $w$ in StyleGAN controls the character's style, including typeface, orientation, and location. Through the collaborative interaction between the codebook and style, we generate a high-resolution structure prior that aligns with LR characters both spatially and structurally. Experiments demonstrate that this structure prior provides robust, character-specific guidance, enabling the accurate restoration of clear strokes in degraded characters, even for real-world LR Chinese text with irregular layouts.  </p>
-->

<img src="./Imgs/prior.gif" width="800px">
<p align="center"> Character Structure Prior Pretraining </p>
</div>


## MARCONet *VS.* MARCONet++

<div align="center">
<img src="./Imgs/marconet_vs_marconetplus.jpg" width="800px">
<p align="center"> More accurate alignment between character structural prior (green) and the degraded image </p>
</div>



## TODO
- [x] Release the inference code and model.
- [x] Release the synthetic pipeline of our training images.
- [] Release the training code (no plans to release for now). 


## Getting Start

```
git clone https://github.com/csxmli2016/MARCONetPlusPlus
cd MARCONetPlusPlus
conda create -n mplus python=3.8 -y
conda activate mplus
pip install -r requirements.txt
BASICSR_EXT=True pip install basicsr
```
> Please carefully follow the installation steps, especially the final one with **BASICSR_EXT=True**.
> When torchvision > 0.15.2, there may be some problems in BasicSR.

## Pre-trained Models
Download the pre-trained models
```
python checkpoints/download_github.py
```


## Inference for *Regular Chinese Text Image* SR
```
CUDA_VISIBLE_DEVICES=0 python test_marconetplus.py 
```
```
# Parameters:
-i: LR path, default: ./Testsets/LQs
-o: save path, default: None will automatically make the saving dir with the format of '[LR path]_TIME_MARCONet'
```

### ⚠️ Common Problems and Solution

If you encounter the following error:

```NameError: name 'fused_act_ext' is not defined```

Try running the following command:

```bash
export BASICSR_JIT='True'
```

✅ This has resolved the issue in most of our cases.


## W interpolation from three characters with different styles
<img src="./Imgs/w-interpolation.gif"  width="600px">







## ‼️ Failure Case
Despite its high-fidelity performance, MARCONet++ still struggles in some real-world scenarios as it highly relies on:

- Real world character **Recognition** on complex degraded text images
- Real world character **Detection** on complex degraded text images
- Text line detection and segmentation
- Domain gap between our synthetic and real-world text images


<img src="./Imgs/failure_case.jpg"  width="600px">

Restoring complex text with high fidelity under such conditions remains challenging. We encourage any potential collaborations to jointly tackle this challenge in robust, high-fidelity text restoration.



## License
This project is licensed under <a rel="license" href="https://github.com/csxmli2016/MARCONet/blob/main/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.


## Citation
```
@InProceedings{li2025marconetplus,
  author = {Li, Xiaoming and Zuo, Wangmeng and Loy, Chen Change},
  title = {Enhanced Generative Structure Prior for Chinese Text Image Super-Resolution},
  booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2025}
}
```


