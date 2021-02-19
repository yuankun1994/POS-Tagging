import os 
import time 
import logging
import argparse
import torch 
import torch.nn as nn 
import numpy as np 
from torch.nn import functional as F 

from data import get_data_iter, get_bert_iter
from models import get_model 
from utils import get_cfg_dict, set_random_seeds, categorical_accuracy, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Demo of face detection and verification")
    parser.add_argument('config', type=str, default=None,
                        help='The dir of config file.')
    args = parser.parse_args()
    return args 


def convert_str_to_token(string):
    


def main(args):
    logging.basicConfig(level = logging.INFO, 
            format = '%(asctime)s - %(levelname)s - %(message)s')

    cfg = get_cfg_dict(args.config)

    logging.info("Building model ...")
    model = get_model(cfg.model).to(cfg.device)
    logging.info("There are {} trainable parameters in {}".format(count_parameters(model), cfg.model['type']))
    if cfg.resume:
        assert os.path.isfile(cfg.resume)
        logging.info("Loading checkpoints from {}".format(cfg.resume))
        model.load_state_dict(tirch.load(cfg.resume))
    else:
        model.init_weights() 
    if not os.path.isdir(cfg.work_dir):
        os.mkdir(cfg.work_dir)

    logging.info("Building optimizer ...")
    optim_cfg = cfg.optimizer 
    optim_type = optim_cfg.pop('type')
    optim_cfg.update({'params': model.parameters()})
    optimizer = getattr(torch.optim, optim_type)(**optim_cfg)

    logging.info("Building datasets ...") 
    data_type = cfg.data.pop('type')
    data_func = get_bert_iter if data_type == 'bert' else get_data_iter
    train_iterator, valid_iterator, test_iterator = data_func(**cfg.data) 

    logging.info("Building loss ...")
    loss_cfg = cfg.loss 
    loss_type = cfg.loss.pop('type')
    loss_func = getattr(nn, loss_type)(**loss_cfg)

    num_epoch = cfg.num_epoch 

    logging.info("Sart training")

    for ep in range(num_epoch):
        start = time.time()
        train_loss, train_acc = train(model, loss_func, optimizer, cfg, train_iterator)
        val_loss, val_acc = evaluate(model, loss_func, cfg, valid_iterator)
        end = time.time() 
        ckpt_file = os.path.join(cfg.work_dir, '{:.5f}_{}_{}_epoch{}.pt'.format(val_acc, cfg.model['type'], cfg.tag, ep))
        torch.save(model.state_dict(), ckpt_file)
        logging.info('Epoch [{}/{}] time: {:.2f}'.format(ep, num_epoch, end - start))
        logging.info('train_loss: {:.5f}, train_acc: {:.5f}'.format(train_loss, train_acc))
        logging.info('eval_loss : {:.5f}, eval_acc : {:.5f}'.format(val_loss, val_acc))
        logging.info("Save model to {}".format(ckpt_file))


if __name__ == '__main__':
    args = parse_args()
    main(args)
