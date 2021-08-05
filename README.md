This repository contains a modified generate.py with some added features. Additional options can be viewed with generate.py -h

By default, a video and a <=8MB gif will be generated for the input prompt, in addition to the image itself.

The main additions include:
* Negative prompts (via -np): accuracy for negative prompts is optimised to be as low as possible
* A learning rate scheduler
* Allowing for a configurable amount of overtime to let the adaptive scheduler complete image generation
* Generating a gif, with automatic skip rate to stay within a size limit
* Video generation via ffmpeg (including adding the final image as a thumbnail)
* CUDA device selection
* Using the (now released) ViT-B/16 model by default (note: this will use more VRAM than ViT-B/32) 

(A more complete documentation of changes may be added later)

The original readme is available below, with only the 'Advanced options' section being updated.

# VQGAN-CLIP Overview
A repo for running VQGAN+CLIP locally. This started out as a Katherine Crowson VQGAN+CLIP derived Google colab notebook.

Original notebook: [![Open In Colab][colab-badge]][colab-notebook]

[colab-notebook]: <https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

Some example images:

<img src="./samples/Cartoon3.png" width="256px"></img><img src="./samples/Cartoon.png" width="256px"></img><img src="./samples/Cartoon2.png" width="256px"></img>
<img src="./samples/Bedroom.png" width="256px"></img><img src="./samples/DemonBiscuits.png" width="256px"></img><img src="./samples/Football.png" width="256px"></img>
<img src="./samples/Fractal_Landscape3.png" width="256px"></img><img src="./samples/Games_5.png" width="256px"></img>

Environment:
* Tested on Ubuntu 20.04
* GPU: Nvidia RTX 3090
* Typical VRAM requirements:
  * 24 GB for a 900x900 image
  * 10 GB for a 512x512 image
  *  8 GB for a 380x380 image

# Set up
Example set up using Anaconda to create a virtual Python environment with the prerequisites:

```
conda create --name vqgan python=3.9
conda activate vqgan

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops 

git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers.git
```
You will also need at least 1 VQGAN pretrained model. E.g.
```
mkdir checkpoints

curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt' #ImageNet 16384
```
The `download_models.sh` script is an optional way to download a number of models. By default, it will download just 1 model.

See https://github.com/CompVis/taming-transformers#overview-of-pretrained-models for more information about pre-trained models.

By default, the model .yaml and .ckpt files are expected in the `checkpoints` directory.
See https://github.com/CompVis/taming-transformers for more information on datasets and models.

# Run
To generate images from text, specify your text prompt as shown in the example below:
```
python generate.py -p "A painting of an apple in a fruit bowl"
```
<img src="./samples/A_painting_of_an_apple_in_a_fruitbowl.png" width="256px"></img>

## Multiple prompts
Text and image prompts can be split using the pipe symbol in order to allow multiple prompts. For example:

```
python generate.py -p "A painting of an apple in a fruit bowl | psychedelic | surreal | weird"
```
<img src="./samples/Apple_weird.png" width="256px"></img>

Image prompts can be split in the same way. For example:

```
python generate.py -p "A picture of a bedroom with a portrait of Van Gogh" -ip "samples/VanGogh.jpg | samples/Bedroom.png"
```

## "Style Transfer"
An input image with style text and a low number of iterations can be used create a sort of "style transfer" effect. For example:

```
python generate.py -p "A painting in the style of Picasso" -ii samples/VanGogh.jpg -i 80 -se 10 -opt AdamW -lr 0.25
```

| Output        | Style         |
| ------------- | ------------- |
| <img src="./samples/vvg_picasso.png" width="256px"></img> | Picasso |
| <img src="./samples/vvg_sketch.png" width="256px"></img>  | Sketch  |
| <img src="./samples/vvg_psychedelic.png" width="256px"></img>  | Psychedelic  |

## Feedback example
By feeding back the generated images and making slight changes, some interesting effects can be created.

The example `zoom.sh` shows this by applying a zoom and rotate to generated images, before feeding them back in again.
To use `zoom.sh`, specifying a text prompt, output filename and number of frames. E.g.
```
./zoom.sh "A painting of a red telephone box spinning through a time vortex" Telephone.png 150
```
<img src="./samples/zoom.gif" width="256px"></img>

## Random text example
Use `random.sh` to make a batch of images from random text. Edit the text and number of generated images to your taste!
```
./random.sh
```


## Advanced options
To view the available options, use "-h".
```
python generate.py -h
```

# Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```
```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Katherine Crowson - https://github.com/crowsonkb

Public Domain images from Open Access Images at the Art Institute of Chicago - https://www.artic.edu/open-access/open-access-images
