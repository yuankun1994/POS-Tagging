import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertModel, BertPreTrainedModel, BertConfig



class BertNN(nn.Module):
    '''
    Parameters:
    -----------
        output_dim: int 
            The number of output features.
        bert_cfg： dict
            Config of bert.
        dropout: float (default 0.0)
            The dropout rate.
        pad_idx: int (default 0)
            The value index of padding elements.
    '''
    def __init__(self, 
                 output_dim, 
                 bert_cfg=dict(bert_dir='bert-base-uncased'),
                 dropout=0.0, 
                 pad_idx=0):
        
        super(BertNN, self).__init__()

        self._output_dim = output_dim 
        self._bert_cfg = bert_cfg 
        self._dropout = dropout
        self._pad_idx = pad_idx 

        self._build_layers()

    def _build_layers(self):
        if 'bert_dir' in self._bert_cfg:
            self.bert_encoder = BertModel.from_pretrained(self._bert_cfg['bert_dir'])
        else:
            cfg = BertConfig(**self._bert_cfg)
            self.bert_encoder = BertModel(cfg)
        self.bert_out_dim = self.bert_encoder.config.hidden_size
    
        self._fc = nn.Linear(self.bert_out_dim, self._output_dim)
        self._dropout = nn.Dropout(self._dropout)

    def init_weights(self):
        nn.init.kaiming_normal_(self._fc.weight)

    def forward(self, sent, masks=None, tag_pad_idx=None):
        sent = sent.permute(1, 0)   # [sen_len, batch] -> []batch, sen_len]
        encoded_layers = self.bert_encoder(input_ids=sent, attention_mask=masks, use_cache=True, return_dict=False)[0]
        feats = encoded_layers.permute(1, 0, 2)     #[sen_len, batch size, output dim]
        feats = self._fc(self._dropout(feats))

        return feats

    @property 
    def padding_idx(self):
        return self._pad_idx