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
Here, we adopt the dataset from torchtext.data, which contains about 12543 training examples, 2002 validation examples and 2077 testing examples. The dataset has 3 'field': 'text', 'udtags' and 'ptbtags'. There are 18 different tags in 'udtages' and 51 tags in 'ptbtags'. 
|   Field  |   Tags  |
| :------: | :------ |
| 'udtags' | NOUN,PUNCT,VERB,PRON,ADP,DET,PROPN,ADJ,AUX,ADV,CCONJ,PART,NUM,SCONJ,X,INTJ,SYM |
| 'ptdtags'| NN,IN,DT,NNP,PRP,JJ,RB,.,VB,NNS,,,CC,VBD,VBP,VBZ,CD,VBN,VBG,MD,TO,PRP$,-RRB-,-LRB-,WDT,WRB,:,``,'',WP,RP,UH,POS,HYPH,JJR,NNPS,JJS,EX,NFP,GW,ADD,RBR,$,PDT,RBS,SYM,LS,FW,AFX,WP$,XX |

## Methods 
Here, we provide 3 method for POS tagging: bidirection-LSTM, BERT+LSTM and BERT. And we train these model with 'udtags'.
| Method | Val-Acc(%) | Link |
| :----- | :--------: | :--: |
|Bidirec-LSTM | 89.79 | [download](https://drive.google.com/file/d/1-fSytr2cvh2ZpscLvVttPZiF773Bc1mn/view?usp=sharing) |
|Bert-LSTM | 91.79 | [download](https://drive.google.com/file/d/1SdHb7hoKKWFCngyuLEwlXHzdGQ2_AOv3/view?usp=sharing) |
|Bert | 92.49 | [download](https://drive.google.com/file/d/1MX2JKLjCipH2j-RXQmckzYHoh1fVptB1/view?usp=sharing) |

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