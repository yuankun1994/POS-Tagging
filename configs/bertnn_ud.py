model = dict(
    type = 'BertNN',
    output_dim = 18, 
    bert_cfg=dict(bert_dir='bert-base-uncased'),
    dropout = 0.4, 
    pad_idx = 0
)
data = dict(
    type = 'bert',
    batch_size = 32, 
    device = 'cuda'
)
optimizer = dict(
    type = 'Adam',
    lr = 5e-5,
    weight_decay = 1e-5
)
loss = dict(
    type = 'CrossEntropyLoss',
    ignore_index = 0
)
tag = 'udtags'
tag_pad_idx = 0
resume = None 
num_epoch = 10
device = 'cuda'
work_dir = './work_dirs/bertnn'