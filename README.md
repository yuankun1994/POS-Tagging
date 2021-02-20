# POS-Tagging
## Introduction
Part-of-speech tagging (POS tagging), also called grammatical tagging is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech. \
This project is an implementation of POS Tagging base on PyTorch and TorchText.

## Requirements 
Python 3.7.9 \
PyTorch 1.7.1 \
TorchText 0.8.1 \
Transformers 4.2.2 \
Numpy 1.20.0 

## Dataset 
Here, we adopt the Universal Dependencies dataset in torchtext.data, which contains about 12543 training examples, 2002 validation examples and 2077 testing examples. The dataset has 3 'field': 'text', 'udtags' and 'ptbtags'. There are 18 different tags in 'udtages' and 51 tags in 'ptbtags'. 
```
---------------------------------------------------------------------------------------------
| Field     |                                    Tags                                       |
|-----------|-------------------------------------------------------------------------------|
|'udtags'   | NOUN,PUNCT,VERB,PRON,ADP,DET,PROPN,ADJ,AUX,ADV,CCONJ,PART,NUM,SCONJ,X,INTJ,SYM|
|-----------|-------------------------------------------------------------------------------|
|           | NN,IN,DT,NNP,PRP,JJ,RB,.,VB,NNS,,,CC,VBD,VBP,VBZ,CD,VBN,VBG,MD,TO,PRP$,-RRB-, |
|'ptbtags'  |-LRB-,WDT,WRB,:,``,'',WP,RP,UH,POS,HYPH,JJR,NNPS,JJS,EX,NFP,GW,ADD,RBR,$,PDT,  |
|           |RBS,SYM,LS,FW,AFX,WP$,XX                                                       |
---------------------------------------------------------------------------------------------
```

## Methods 
Here, we provide 3 method for POS tagging: bidirection-LSTM, BERT+LSTM and BERT. And we train these model with 'udtags' and 'ptbtags'.

| METHOD(UDTAGS) | VAL-ACC(%) | LINK |
| :----- | :--------: | :--: |
|Bidirec-LSTM | 89.79 | [download](https://drive.google.com/file/d/1Hn9F6AWO3cCJTL4U2p6B_R2JXuiKzBzy/view?usp=sharing) |
|Bert-LSTM | 91.79 | [download](https://drive.google.com/file/d/1rM3rSLeuZho9AWzOoy-MENNBvTuZimZb/view?usp=sharing) |
|Bert | 92.49 | [download](https://drive.google.com/file/d/1zR5VsW_MnmkJ_sUSl7dlTKND-U_qr4b9/view?usp=sharing) |

| METHOD(PTBTAGS) | VAL-ACC(%) | LINK |
| :----- | :--------: | :--: |
|Bidirec-LSTM | 88.05 | [download](https://drive.google.com/file/d/1f8n8kD00bgf2KQS1TVu8zgIYU2sEHTOx/view?usp=sharing) |
|Bert-LSTM | 91.07 | [download](https://drive.google.com/file/d/1pLUmrUYpT1AtyIsg8aG7co2ENC30b6vA/view?usp=sharing) |
|Bert | 91.33 | [download](https://drive.google.com/file/d/11gNwjaEln7lRyhcu6u3Gt1te_wjif2KF/view?usp=sharing) |

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
python train.py configs/bertnn_ud.py
```
Most settings used for training and evaluation are set in your runfiles. Each runfile should correspond to a single experiment.