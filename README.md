# sky-dehazing
Implements dehazing evaluation with sky segmentation

micromamba create -n sky-dehazing python==3.10.9
micromamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
micromamba install -c conda-forge yapf=0.40.1 ftfy regex -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
