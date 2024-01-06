# sky-dehazing
Implements dehazing evaluation with sky segmentation

Enviroment installation micromamba

```bash

micromamba create -n sky-dehazing python==3.10.9
micromamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
micromamba install -c conda-forge yapf=0.40.1 ftfy regex -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

```

Enviroment installation conda

```bash

conda create -n sky-dehazing python==3.10.9
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge yapf=0.40.1 ftfy regex -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .

```

## Running python simple demo script

The python `demo_1.py` script and `demo_dir.py` can be used to do the sky segmentation for 1 image or a dir with images. 



## Copy the initial dataset from k80:

Directory paths:

```bash
/home/dgonzalez/map-net/HazeWorld/gt_new
/home/dgonzalez/map-net/HazeWorld/hazy_new
```
