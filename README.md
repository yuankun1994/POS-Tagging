# POS-Tagging
## Introduction
Part-of-speech tagging (POS tagging), also called grammatical tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech. \
This projext is an implementation of POS Tagging base on PyTorch and TorchText.

## Requirements 
Python 3.7.9 \
PyTorch 1.7.1 \
TorchText 0.8.1 \
Transformers 4.2.2 \
Numpy 1.20.0 

## Installation 
1. Setup python virtual environment.
    ```
    conda create -n pos_tagging python=3.7 -y 

    conda activate pos_tagging
    ```
2. Install dependencies.
    ```
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

    pip install torchtext 

    pip install transformers
    ```
3. Clone this repo. 
    ```
    git clone https://github.com/KunyFox/POS-Tagging.git 

    cd POS-Tagging
    ```

## Usage
 Train:
```
python train.py YOUR_CFG_FILE
```
Such as:
```
python train.py configs/bertnn.py
```
Most settings used for training and evaluation are set in your runfiles. Each runfile should correspond to a single experiment.