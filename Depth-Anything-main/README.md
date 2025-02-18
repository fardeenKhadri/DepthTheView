<div align="center">
<h2>Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data</h2>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://scholar.google.com/citations?user=NmHgX-wAAAAJ)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Xiaogang Xu**](https://xiaogang00.github.io/)<sup>3,4</sup> · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

<sup>1</sup>HKU&emsp;&emsp;&emsp;&emsp;<sup>2</sup>TikTok&emsp;&emsp;&emsp;&emsp;<sup>3</sup>CUHK&emsp;&emsp;&emsp;&emsp;<sup>4</sup>ZJU

&dagger;project lead&emsp;*corresponding author

**CVPR 2024**

<a href="https://arxiv.org/abs/2401.10891"><img src='https://img.shields.io/badge/arXiv-Depth Anything-red' alt='Paper PDF'></a>
<a href='https://depth-anything.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/LiheYoung/Depth-Anything'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/papers/2401.10891'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a>
</div>

This work presents Depth Anything, a highly practical solution for robust monocular depth estimation by training on a combination of 1.5M labeled images and **62M+ unlabeled images**.


You can easily load our pre-trained models by:
```python
from depth_anything.dpt import DepthAnything

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
```

## Usage 

### Installation

```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

### Running

```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```
Arguments:
- ``--img-path``: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- ``--pred-only`` is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.
- ``--grayscale`` is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

**If you want to use Depth Anything on videos:**
```bash
python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
```

### Gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/LiheYoung/Depth-Anything).

### Import Depth Anything to your project

If you want to use Depth Anything in your own project, you can simply follow [``run.py``](run.py) to load our models and define data pre-processing. 

<details>
<summary>Code snippet (note the difference between our data pre-processing and that of MiDaS)</summary>

```python
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

image = cv2.cvtColor(cv2.imread('your image path'), cv2.COLOR_BGR2RGB) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
depth = depth_anything(image)
```
</details>

#
Easily use Depth Anything through [``transformers``](https://github.com/huggingface/transformers) within 3 lines of code! Please refer to [these instructions](https://huggingface.co/docs/transformers/main/model_doc/depth_anything) (credit to [@niels](https://huggingface.co/nielsr)).


```python
from transformers import pipeline
from PIL import Image

image = Image.open('Your-image-path')
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = pipe(image)["depth"]
```
</details>

