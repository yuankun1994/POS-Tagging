import torch 
import functools

from torchtext import data
from torchtext import datasets

from transformers import BertTokenizer

def get_data_iter(batch_size, device='cuda', min_feeq=2):
    TEXT = data.Field(lower = True)
    UD_TAGS = data.Field(unk_token = None)
    PTB_TAGS = data.Field(unk_token = None)

    fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

    TEXT.build_vocab(train_data, 
                    min_freq = min_feeq,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)

    # UD_TAGS.build_vocab(train_data)
    # PTB_TAGS.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device)

    return (train_iterator, valid_iterator, test_iterator) #, (TEXT, UD_TAGS, PTB_TAGS)

def get_bert_iter(batch_size, device='cuda', max_input_length=512, bert_dir='bert-base-uncased',
                  init_token_idx=101, pad_token_idx=0, unk_token_idx=100):
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    TEXT = data.Field(use_vocab = False,
                     lower = True,
                     preprocessing = functools.partial(cut_and_convert_to_id,
                                      tokenizer = tokenizer, 
                                      max_input_length = max_input_length),
                     init_token = init_token_idx,
                     pad_token = pad_token_idx,
                     unk_token = unk_token_idx)

    UD_TAGS = data.Field(unk_token = None,
                         init_token = '<pad>',
                         preprocessing = functools.partial(cut_to_max_length,
                                     max_input_length = max_input_length))
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
    UD_TAGS.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device)
    return (train_iterator, valid_iterator, test_iterator)



def cut_and_convert_to_id(tokens, tokenizer, max_input_length):
    tokens = tokens[:max_input_length-1]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    return tokens


def cut_to_max_length(tokens, max_input_length):
    tokens = tokens[:max_input_length-1]
    return tokens