import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertModel, BertPreTrainedModel, BertConfig



class BertLSTM(nn.Module):
    '''
    Parameters:
    -----------
        output_dim: int 
            The number of output features.
        hidden_dim: int 
            The number of features in the hidden state.
        num_layers: int (default 2)
            The number of recurrent layers.
        bidirectional: bool (deafult True)
            Wether to employ a bidirectional LSTM.
        bert_cfg： dict
            Config of bert.
        dropout: float (default 0.0)
            The dropout rate.
        pad_idx: int (default 0)
            The value index of padding elements.
    '''
    def __init__(self, 
                 output_dim, 
                 hidden_dim,
                 num_layers=2,
                 bidirectional=True,
                 bert_cfg=dict(bert_dir='bert-base-uncased'),
                 dropout=0.0, 
                 pad_idx=0):
        
        super(BertLSTM, self).__init__()

        self._output_dim = output_dim 
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers 
        self._bidirectional = bidirectional
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
        hidden_dim = self._hidden_dim // 2 if self._bidirectional else self._hidden_dim
        self._lstm = nn.LSTM(self.bert_out_dim,
                             hidden_dim,
                             batch_first=True,
                             num_layers=self._num_layers,
                             bidirectional=self._bidirectional)
        self._fc = nn.Linear(self._hidden_dim, self._output_dim)
        self._dropout = nn.Dropout(self._dropout)

    def init_weights(self):
        nn.init.kaiming_normal_(self._fc.weight)
         

    def forward(self, sent, masks=None, tag_pad_idx=None):
        sent = sent.permute(1, 0)   # [sen_len, batch] -> [batch, sen_len]
        feats = self.bert_encoder(input_ids=sent, attention_mask=masks, use_cache=True, return_dict=False)[0]
        feats, _ = self._lstm(feats)
        feats = feats.permute(1, 0, 2)     #[sen_len, batch size, output dim]
        feats = self._fc(self._dropout(feats))

        return feats

    @property 
    def padding_idx(self):
        return self._pad_idx