model = dict(
    type = 'BidirectionalLSTM',
    num_embeddings = 8866,
    embedding_dim = 100, 
    hidden_dim = 128, 
    output_dim = 18, 
    bidirectional = True, 
    dropout = 0.25, 
    pad_idx = 1
)
data = dict(
    type = 'no-bert',
    batch_size = 128, 
    device = 'cuda',
    min_feeq = 2
)
optimizer = dict(
    type = 'Adam',
    lr = 1e-3,
    weight_decay = 1e-5
)
loss = dict(
    type = 'CrossEntropyLoss',
    ignore_index = 0
)
tag_pad_idx = 0
resume = None 
num_epoch = 10
device = 'cuda'
work_dir = './work_dirs/bidirec_lstm'