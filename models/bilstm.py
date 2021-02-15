import torch 
import torch.nn as nn 


class BidirectionalLSTM(nn.Module):
    '''
    Parameters:
    -----------
        input_dim: int 
            The size of the dictionary of embeddings.
        embedding_dim: int 
            The size of each embedding vector.
        hidden_dim: int 
            The number of features in the hidden state.
        output_dim: int 
            The number of output features.
        num_layers: int (default 2)
            Number of recurrent layers.
        bidirectional: bool (deafult false)
            Wether to employ a bidirectional LSTM.
        dropout: float (default 0.0)
            The dropout rate.
        pad_idx: int (default 0)
            The value index of padding elements.
    '''
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=2, 
                 bidirectional=False, 
                 dropout=0.0, 
                 pad_idx=0):
        
        super(BidirectionalLSTM, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim 
        self._hidden_dim = hidden_dim 
        self._output_dim = output_dim 
        self._num_layers = num_layers 
        self._bidirectional = bidirectional 
        self._dropout = dropout
        self._pad_idx = pad_idx 

        self._build_layers()

    def _build_layers(self):
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim, padding_idx=self._pad_idx)
        self._lstm = nn.LSTM(self._embedding_dim, 
                            self._hidden_dim, 
                            num_layers=self._num_layers, 
                            bidirectional=self._bidirectional,
                            dropout=self._dropout if self._num_layers > 1 else 0)
        self._fc = nn.Linear(self._hidden_dim * 2 if self._bidirectional else self._hidden_dim, self._output_dim)
        self._dropout = nn.Dropout(self._dropout)

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.1)

    def forward(self, text):
        embedded = self._dropout(self._embedding(text))
        outputs, (hidden, cell) = self._lstm(embedded)
        predictions = self._fc(self._dropout(outputs))
        # [sent len, batch size, output dim]
        return predictions

    @property 
    def num_embeddings(self):
        return self._num_embeddings 

    @property
    def is_bidirectional(self):
        return self._bidirectional 

    @property 
    def padding_idx(self):
        return self._pad_idx