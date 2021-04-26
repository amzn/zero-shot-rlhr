import sys 
import os 
import random 
import numpy as np 
import yaml 
import argparse
import logging
from time import strftime, gmtime
import json
import re

import torch 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import apex

import utils
from trainer import do_train, do_rl_train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='bert', choices=['bert', 'bert-rl', 'distilbert', 'distilbert-rl'], 
                        help="train mode, a vanilla bert/distilbert or rl")
    parser.add_argument("--model_config", type=str, default='bert-base-uncased',
                        help="model_type")
    parser.add_argument('--model_path', type=str, default=None,
                        help="The pretrained model to recover for RL, should be a path")
    parser.add_argument('--para', type=str, default='para.yml',
                        help="the path to the parameter file, has to be a yaml file")
    parser.add_argument("--dataset", type=str, default='wos', choices=['yelp', 'wos', 'qba'], 
                        help="train which dataset, yelp, wos or qba")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--learning_rate", type=float,
                        help="if want to use a learning rate for rl different from the pretraining")
    parser.add_argument("--save_steps", type=int, default=-1,
                        help="steps to save the model")
    parser.add_argument("--logging_steps", type=int, default=-1,
                        help="steps to log")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=-1,
                        help="batch size per GPU")
    parser.add_argument("--num_training_epochs", type=int, default=-1,
                        help="training epoches")
    parser.add_argument("--max_source_seq_length", type=int, default=240,
                        help="training epoches")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="training epoches")
    parser.add_argument("--pos_reward", type=float, default=1.0,
                        help="reward for positive cases")
    parser.add_argument("--negative_sample_num", type=int, default=10,
                        help="number of sampled negative examples for training pretrained models")
    parser.add_argument("--K", type=int, default=2,
                        help="number of path sampled")
    parser.add_argument("--rl_pretrain", action='store_true',
                        help="if it is rl_pretrain, the target will be set as 0.2 and 0.8")
    parser.add_argument("--use_path", action='store_true',
                        help="use deduction path or label text as the target")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    with open(args.para) as fin:
        paras = yaml.safe_load(fin)
    paras['mode'] = args.mode
    if args.model_path is not None and os.path.exists(args.model_path):
        paras['model_path'] = args.model_path
    if args.model_config != None:
        paras['model_config'] = args.model_config
    paras['dataset'] = args.dataset 
    if args.dataset == 'wos':
        paras['path_max_length'] = 2
    elif args.dataset == 'yelp':
        paras['path_max_length'] = 4
    elif args.dataset == 'qba':
        paras['path_max_length'] = 3
    paras['pos_reward'] = args.pos_reward
    paras['local_rank'] = args.local_rank
    paras['no_cuda'] = args.no_cuda 
    paras['rl_pretrain'] = args.rl_pretrain
    paras['use_path'] = args.use_path
    paras['fp16'] = args.fp16
    paras['fp16_opt_level'] = args.fp16_opt_level
    paras['negative_sample_num'] = args.negative_sample_num
    paras['K'] = args.K
    paras['max_source_seq_length'] = args.max_source_seq_length
    paras['max_seq_length'] = args.max_seq_length

    if args.learning_rate is not None:
        paras['learning_rate'] = args.learning_rate
    if args.save_steps != -1:
        paras['save_steps'] = args.save_steps
    if args.logging_steps != -1:
        paras['logging_steps'] = args.logging_steps
    if args.per_gpu_train_batch_size != -1:
        paras['per_gpu_train_batch_size'] = args.per_gpu_train_batch_size
    if args.num_training_epochs != -1:
        paras['num_training_epochs'] = args.num_training_epochs

    return paras 


def prepare(paras, logger):
    # Setup CUDA, GPU & distributed training
    if paras['local_rank'] == -1 or paras['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not paras['no_cuda'] else "cpu")
        paras['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(paras['local_rank'])
        device = torch.device("cuda", paras['local_rank'])
        torch.distributed.init_process_group(backend='nccl')
        paras['n_gpu'] = 1
    logger.info(json.dumps(paras, indent=4))
    paras['device'] = device

    if paras['fp16']:
        try:
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    return paras


def set_random_seed(paras):
    random.seed(paras['random_seed'])
    torch.manual_seed(paras['random_seed'])
    np.random.seed(paras['random_seed'])
    if paras['n_gpu'] > 0:
        torch.cuda.manual_seed_all(paras['random_seed'])

# get the model and tokenizer
def get_model_and_tokenizer(paras):
    if paras['mode'] == 'bert' or paras['mode'] == 'bert-rl':
        tokenizer = BertTokenizer.from_pretrained(paras['model_config'])
        model = BertForSequenceClassification.from_pretrained(paras['model_path'] if paras['model_path'] is not None else paras['model_config'])
    elif paras['mode'] == 'distilbert' or paras['mode'] == 'distilbert-rl':
        tokenizer = DistilBertTokenizer.from_pretrained(paras['model_config'])
        model = DistilBertForSequenceClassification.from_pretrained(paras['model_path'] if paras['model_path'] is not None else paras['model_config'])
    
    return model, tokenizer


def train_bert(paras):
    # specify the current run dir
    current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
    paras['current_time'] = current_time
    paras['log_dir'] = os.path.join(paras['output_dir'], current_time)
    paras['result_dir'] = os.path.join(paras['output_dir'], current_time, paras['result_dir'])
    paras['cache_dir'] = os.path.join(paras['output_dir'], current_time, paras['cache_dir'])
    paras['cached_feature_dir'] = os.path.join(paras['cached_feature_dir'], paras['dataset'])
    utils.make_all_dirs(paras)
    # define the logger
    logger_name = os.path.join(paras['log_dir'], "log.txt")
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()

    paras = prepare(paras, logger)
    set_random_seed(paras)

    if paras['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()

    model, tokenizer = get_model_and_tokenizer(paras)

    if paras['local_rank'] == 0:
        torch.distributed.barrier()
    
    if paras['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()

    # read and save / load the training data
    train_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'train.json')
    taxonomy_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'taxonomy.json')
    seen_label_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'seen_labels.txt')
    cached_file = os.path.join(paras['cached_feature_dir'], '{}_cached_examples_for_training.pt'.format(paras['mode']))
    taxonomy_cached_file = os.path.join(paras['cached_feature_dir'], '{}_cached_taxonomy.json'.format(paras['mode']))
    training_data = utils.load_and_cache_examples(
        train_file=train_file, taxonomy_file=taxonomy_file, seen_label_file=seen_label_file, 
        tokenizer=tokenizer, local_rank=paras['local_rank'], cached_features_file=cached_file, 
        taxonomy_cache_file=taxonomy_cached_file, logger=logger
    )

    # convert the tokens to ids
    cached_file = os.path.join(paras['cached_feature_dir'], '{}_cached_features_for_training_sampling_{}.pt'.format(paras['mode'], str(paras['negative_sample_num'])))
    training_features = utils.convert_examples_to_id_for_bert(
        examples=training_data['examples'], taxonomy=training_data['taxonomy'], seen_labels=training_data['seen_labels'], 
        local_rank=paras['local_rank'], tokenizer=tokenizer, cached_features_file=cached_file, 
        negative_sample_num=paras['negative_sample_num'], use_path=paras['use_path'], 
        logger=logger, shuffle=True, rl_pretrain=paras['rl_pretrain']
    )

    if paras['local_rank'] == 0:
        torch.distributed.barrier()

    # train bert
    do_train(paras, training_features, model, tokenizer, logger)


def train_bert_rl(paras):
    # extract the dir name from the input
    if paras['model_path'] is not None:
        current_time = re.findall("/(2020.+?)/.*", paras['model_path'])[0]
    else:
        current_time = strftime("%Y-%b-%d-%H_%M_%S", gmtime())
    paras['current_time'] = current_time
    paras['log_dir'] = os.path.join(paras['output_dir'], current_time)
    paras['cache_dir'] = os.path.join(paras['output_dir'], current_time, paras['rl_cache_dir'])
    paras['cached_feature_dir'] = os.path.join(paras['cached_feature_dir'], paras['dataset'])
    utils.make_all_dirs(paras)

    # define the logger
    logger_name = os.path.join(paras['log_dir'], "rl_log.txt")
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()

    paras = prepare(paras, logger)
    set_random_seed(paras)

    if paras['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()
    
    # get the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(paras)

    if paras['local_rank'] == 0:
        torch.distributed.barrier()
    
    if paras['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()
    
    # read and save / load the training data
    train_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'train.json')
    taxonomy_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'taxonomy.json')
    seen_label_file = os.path.join(paras['data_dir'], paras['dataset'], 'split', 'seen_labels.txt')
    cached_file = os.path.join(paras['cached_feature_dir'], '{}_cached_examples_for_training.pt'.format(paras['mode'].split('-')[0]))
    taxonomy_cached_file = os.path.join(paras['cached_feature_dir'], '{}_cached_taxonomy.json'.format(paras['mode'].split('-')[0]))
    training_data = utils.load_and_cache_examples(
        train_file=train_file, taxonomy_file=taxonomy_file, seen_label_file=seen_label_file, 
        tokenizer=tokenizer, local_rank=paras['local_rank'], cached_features_file=cached_file, 
        taxonomy_cache_file=taxonomy_cached_file, logger=logger
    )

    # convert the categories to deduction paths
    training_data_with_path = utils.convert_category_to_path(training_data, local_rank=paras['local_rank'], logger=logger)
    # convert the rl examples to ids
    training_data_with_id = utils.convert_examples_to_id_for_rl(
        data=training_data_with_path, tokenizer=tokenizer, 
        local_rank=paras['local_rank'], logger=logger, shuffle=True
    )

    if paras['local_rank'] == 0:
        torch.distributed.barrier()

    do_rl_train(paras, training_data_with_id, model, tokenizer, logger)


def main():
    paras = get_args()

    # train orginal bert
    if paras['mode'] == 'bert' or paras['mode'] == 'distilbert':
        train_bert(paras)
    # train bert with rl
    elif paras['mode'] == 'bert-rl' or paras['mode'] == 'distilbert-rl':
        train_bert_rl(paras)



if __name__ == "__main__":
    main()