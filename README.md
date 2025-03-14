# Multimodal-Aware Fusion Network For Referring Remote Sensing Image Segmentation
Code for our GRSL 2025 paper"[Multimodal-Aware Fusion Network for Referring Remote Sensing Image Segmentation]"
![Pipeline Image](MAFN.png)

Contributed by Leideng Shi, Juan Zhang*.

## Getting Started

### Installation
Install the dependencies.

The code was tested on Ubuntu 20.04.6, with Python 3.7 and PyTorch v1.12.1.

1. Clone this repository.

    ~~~
    git clone https://github.com/Roaxy/MQN.git 
    ~~~
2. Create a new Conda environment with Python 3.7 then activate it:
   
    ~~~
   conda create -n MQN python==3.7
   conda activate MQN
    ~~~
3. Install pytorch v1.12.1 (CUDA 10.2 is used in this example).

    ~~~
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
    ~~~
    
4. Install the requirements.
    
    ~~~
    pip install -r requirements.txt
    ~~~
    


## Datasets

## Training
After the preparation, you can start training with the following commands. We use DistributedDataParallel from PyTorch for training. To run on 2 GPUs (with IDs 0, 1) on a single node:
```
sh ./train.sh
```
## Testing
```
# default setting
sh ./test.sh
```
You may modify line 11 in `test.sh` to use `val` instead of `test`. By default, we set the split to `test`.
## Visualization
## Acknowledgements
