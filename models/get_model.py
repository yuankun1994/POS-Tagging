from .bilstm import BidirectionalLSTM 
from .bertnn import BertNN
from .bert_lstm import BertLSTM


model_dict = {
    'BidirectionalLSTM' : BidirectionalLSTM,
    'BertNN': BertNN,
    'BertLSTM': BertLSTM
}

def get_model(model_cfg):
    assert isinstance(model_cfg, dict) 
    assert 'type' in model_cfg
    cfg = model_cfg.copy()
    model_type = cfg.pop('type')
    assert model_type in model_dict 
    return model_dict[model_type](**cfg)

    