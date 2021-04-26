import sys
import os
import argparse
import random
import numpy as np 
import glob
import logging
import json
from tqdm import tqdm 
from time import strftime, gmtime
import apex

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import WEIGHTS_NAME
from transformers import AdamW

from sklearn.metrics import precision_score, recall_score, f1_score


class DatasetForBertEval(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_seq_len,
            cls_id, sep_id, pad_id, mask_id):
        self.features = features
        self.max_source_len = max_source_len
        self.max_seq_len = max_seq_len
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id

    def __len__(self):
        return len(self.features)

    def __trunk(self, ids, max_len):
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, idx):
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["category_ids"], self.max_seq_len - self.max_source_len)

        input_ids = source_ids + target_ids
        segment_ids = [0]*len(source_ids) + [1]*len(target_ids) + [0]*(self.max_seq_len-len(input_ids))
        input_mask = [1]*len(input_ids) + [0]*(self.max_seq_len-len(input_ids))
        input_ids = self.__pad(input_ids, self.max_seq_len)
        label = feature["label"]
        label_id = feature['label_id'],
        example_num = feature['example_num']
        return input_ids, input_mask, segment_ids, label, label_id, example_num


def set_random_seed(seed, n_gpu):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def evaluate(args, model, tokenizer, logger, prefix=""):
    if args.cache_dir == 'cache':
        eval_outputs_dir = os.path.join(args.output_dir, "result")
    elif args.cache_dir == 'rl_cache':
        eval_outputs_dir = os.path.join(args.output_dir, "rl_result")
    
    if args.fp16:
        from apex import amp
    else:
        amp = None
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6, eps=1e-8)

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    data_file = os.path.join(args.data_dir, args.dataset, 'split', '{}.json'.format(args.data_type))
    taxonomy_file = os.path.join(args.data_dir, args.dataset, 'split', 'taxonomy.json')
    seen_label_file = os.path.join(args.data_dir, args.dataset, 'split', 'seen_labels.txt')
    unseen_label_file = os.path.join(args.data_dir, args.dataset, 'split', 'unseen_labels.txt')
    cached_file = os.path.join(args.cached_feature_dir, args.dataset, '{}_cached_features_for_{}.pt'.format(args.mode, args.data_type))
    taxonomy_cached_file = os.path.join(args.cached_feature_dir, args.dataset, '{}_cached_taxonomy.json'.format(args.mode))
    eval_data = load_and_cache_examples(
        data_file, taxonomy_file, seen_label_file, unseen_label_file, tokenizer, args.local_rank, cached_file, taxonomy_cached_file, logger)
    cached_file = os.path.join(args.cached_feature_dir, args.dataset, '{}_cached_id_for_{}.pt'.format(args.mode, args.data_type))
    eval_data_id = convert_examples_to_id_for_bert(
        examples=eval_data['examples'], taxonomy=eval_data['taxonomy'], 
        seen_labels=eval_data['seen_labels'], unseen_labels=eval_data['unseen_labels'], 
        local_rank=args.local_rank, tokenizer=tokenizer, use_path=args.use_path,
        cached_features_file=cached_file, logger=logger)
    eval_features = eval_data_id['features']
    all_labels = eval_data_id['all_labels']

    eval_dataset = DatasetForBertEval(
        features=eval_features, max_source_len=args.max_source_seq_length,
        max_seq_len=args.max_seq_length,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, 
        pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id
    )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                                batch_size=args.eval_batch_size,
                                collate_fn=batch_list_to_batch_tensors)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_labels = None
    label_ids = None
    example_num = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.mode != 'distilbert':
                inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_labels = inputs["labels"].detach().cpu().numpy()
            label_ids = batch[4].cpu().numpy()
            example_num = batch[5].cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_labels = np.append(out_labels, inputs["labels"].detach().cpu().numpy(), axis=0) 
            label_ids = np.append(label_ids, batch[4].cpu().numpy(), axis=0)
            example_num = np.append(example_num, batch[5].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    preds_arg = np.argmax(preds, axis=1)
    prf, mAP = compute_metrics(preds_arg, out_labels, label_ids, example_num, all_labels)
    results = update_result(preds_arg, preds, out_labels, label_ids, example_num, all_labels)
    logger.info(json.dumps(prf))
    logger.info(json.dumps(mAP))
    write_result(eval_outputs_dir, results, args)


def write_result(eval_outputs_dir, results, args):
    if not os.path.exists(eval_outputs_dir):
        os.makedirs(eval_outputs_dir)
    filename = os.path.join(eval_outputs_dir, "{}_checkpoint_{}.json".format(args.data_type, args.checkpoint))
    with open(filename, 'w') as fout:
        fout.write(json.dumps(results, indent=4))


def update_result(preds, logits, out_labels, label_ids, example_num, all_labels):
    case_num = out_labels.shape[0]
    results = {}
    for i in range(case_num):
        if example_num[i].item() not in results:
            results[example_num[i].item()] = {'truth': [], 'pred': []}
        results[example_num[i].item()]['pred'].append([all_labels[label_ids[i].item()]['label'], logits[i].tolist()])
        if out_labels[i].item() == 1:
            results[example_num[i].item()]['truth'].append(all_labels[label_ids[i].item()]['label'])
    return results


def compute_metrics(preds, out_labels, label_ids, example_num, all_labels):
    def compute_prf(unseen_label_set, seen_label_set):
        f_score = 0.0
        precision = 0.0
        recall = 0.0
        for _, val in unseen_label_set.items():
            f_score += f1_score(val['truth'], val['pred'])
            precision += precision_score(val['truth'], val['pred'])
            recall += recall_score(val['truth'], val['pred'])
        unseen_f_score = f_score / len(unseen_label_set)
        unseen_precision = precision / len(unseen_label_set)
        unseen_recall = recall / len(unseen_label_set)

        for _, val in seen_label_set.items():
            f_score += f1_score(val['truth'], val['pred'])
            precision += precision_score(val['truth'], val['pred'])
            recall += recall_score(val['truth'], val['pred'])
        
        overall_f_score = f_score / (len(seen_label_set) + len(unseen_label_set))
        overall_precision = precision / (len(seen_label_set) + len(unseen_label_set))
        overall_recall = recall / (len(seen_label_set) + len(unseen_label_set))

        return {"unseen": {'f': unseen_f_score, 'p': unseen_precision, 'r': unseen_recall}, 
                "overall": {'f': overall_f_score, 'p': overall_precision, 'r': overall_recall}}

    def compute_mAP(unseen_label_set, seen_label_set):
        total_precision = 0.
        for _, val in unseen_label_set.items():
            pred = val['pred']
            truth = val['truth']
            new_val = list(zip(pred, truth))
            new_val = sorted(new_val, key=lambda d: d[0], reverse=True)
            precision = 0.
            correct = 0
            for i, item in enumerate(new_val):
                if item[1] == 0:
                    continue
                if item[0] == item[1]:
                    correct += 1
                precision += correct / (i + 1)
            precision /= sum(truth)
            total_precision += precision
        unseen_precision = total_precision / len(unseen_label_set)
        for _, val in seen_label_set.items():
            pred = val['pred']
            truth = val['truth']
            new_val = list(zip(pred, truth))
            new_val = sorted(new_val, key=lambda d: d[0], reverse=True)
            precision = 0.
            correct = 0
            for i, item in enumerate(new_val):
                if item[1] == 0:
                    continue
                if item[0] == item[1]:
                    correct += 1
                precision += correct / (i + 1)
            precision /= sum(truth)
            total_precision += precision
        overall_precision = total_precision / (len(unseen_label_set) + len(seen_label_set))
        return {"unseen": {'mAP': unseen_precision}, "overall": {'mAP': overall_precision}}

    case_num = out_labels.shape[0]
    seen_label_set = {}
    unseen_label_set = {}
    max_example_id = np.amax(example_num) + 1
    for i in range(case_num):
        label_id = label_ids[i].item()
        if all_labels[label_id]['seen'] and out_labels[i] == 1:
            seen_label_set[label_id] = {"pred": [0]*max_example_id, "truth": [0]*max_example_id}
        elif not all_labels[label_id]['seen'] and out_labels[i] == 1:
            unseen_label_set[label_id] = {"pred": [0]*max_example_id, "truth": [0]*max_example_id}
    for i in range(case_num):
        label_id = label_ids[i].item()
        if all_labels[label_id]['seen'] and label_id in seen_label_set:
            seen_label_set[label_id]['pred'][example_num[i].item()] = preds[i]
            seen_label_set[label_id]['truth'][example_num[i].item()] = out_labels[i]
        elif not all_labels[label_id]['seen'] and label_id in unseen_label_set:
            unseen_label_set[label_id]['pred'][example_num[i].item()] = preds[i]
            unseen_label_set[label_id]['truth'][example_num[i].item()] = out_labels[i]

    prf = compute_prf(unseen_label_set, seen_label_set)
    mAP = compute_mAP(unseen_label_set, seen_label_set)
    print(prf)
    print(mAP)
    return prf, mAP


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def load_taxonomy(taxonomy_file, tokenizer, taxonomy_cache_file):
    if os.path.exists(taxonomy_cache_file):
        with open(taxonomy_cache_file) as fin:
            taxonomy = json.load(fin)
    else:
        taxonomy = dict()
        taxonomy['root'] = dict()
        taxonomy['root']['id'] = 0
        index_ct = 1
        with open(taxonomy_file) as fin:
            for line in fin:
                item = json.loads(line)
                node = item['node']
                children = item['children']
                if node not in taxonomy:
                    taxonomy[node] = dict()
                taxonomy[node]['children'] = children
                taxonomy[node]['title'] = tokenizer.tokenize(node)
                taxonomy[node]['parent'] = ''
                for item in children:
                    if item not in taxonomy:
                        taxonomy[item] = dict()
                        taxonomy[item]['children'] = []
                        taxonomy[item]['title'] = tokenizer.tokenize(item)
                        taxonomy[item]['parent'] = ''
                        taxonomy[item]['id'] = index_ct
                        index_ct += 1
        for key, val in taxonomy.items():
            for item in val['children']:
                taxonomy[item]['parent'] = key
        for key in taxonomy.keys():
            ancestor_path = [key]
            current_node = key
            while taxonomy[current_node]['parent'] != 'root' and taxonomy[current_node]['parent'] != '':
                ancestor_path.append(taxonomy[current_node]['parent'])
                current_node = taxonomy[current_node]['parent']
            ancestor_path.reverse()
            taxonomy[key]['ancestor'] = ancestor_path
        
        taxonomy['root']['level'] = 0
        queue = ['root']
        while True:
            if len(queue) == 0:
                break
            item = queue.pop(0)
            for child in taxonomy[item]['children']:
                taxonomy[child]['level'] = taxonomy[item]['level'] + 1
            queue.extend(taxonomy[item]['children'])
        with open(taxonomy_cache_file, 'w') as fout:
            json.dump(taxonomy, fout, indent=4)
    return taxonomy

def load_and_cache_examples(
        train_file, taxonomy_file, seen_label_file, unseen_label_file, tokenizer, local_rank, cached_features_file, taxonomy_cache_file, logger):
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        training_data = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", train_file)
        seen_labels = []
        with open(seen_label_file) as fin:
            for line in fin:
                seen_labels.append(line.strip())
        unseen_labels = []
        with open(unseen_label_file) as fin:
            for line in fin:
                unseen_labels.append(line.strip())

        taxonomy = load_taxonomy(taxonomy_file, tokenizer, taxonomy_cache_file)

        examples = []
        with open(train_file) as fin:
            for line in tqdm(fin):
                item = json.loads(line)
                text = item['text']
                category = item['category']
                examples.append({'text': tokenizer.tokenize(text), 'category': category})
        training_data = {
            "examples": examples,
            "taxonomy": taxonomy,
            "seen_labels": seen_labels,
            "unseen_labels": unseen_labels
        }

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(training_data, cached_features_file)

    return training_data


def convert_examples_to_id_for_bert(
        examples, taxonomy, seen_labels, unseen_labels, local_rank, tokenizer, use_path, cached_features_file, logger):

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        eval_features = torch.load(cached_features_file)
    else:
        logger.info("Converting examples to id for bert")
        all_labels = {}
        for label in seen_labels:
            ancestor = taxonomy[label]['ancestor']
            ancestor_token = [taxonomy[node]['title'] for node in ancestor]
            path = []
            for i, node in enumerate(ancestor_token):
                if i != 0:
                    path += ['-']
                path += node
            all_labels[taxonomy[label]['id']] = {
                'label': label, 
                'label_id':tokenizer.convert_tokens_to_ids(taxonomy[label]['title']),  
                'path': tokenizer.convert_tokens_to_ids(path), 
                'seen': True
            }
        for label in unseen_labels:
            ancestor = taxonomy[label]['ancestor']
            ancestor_token = [taxonomy[node]['title'] for node in ancestor]
            path = []
            for i, node in enumerate(ancestor_token):
                if i != 0:
                    path += ['-']
                path += node
            all_labels[taxonomy[label]['id']] = {
                'label': label, 
                'label_id':tokenizer.convert_tokens_to_ids(taxonomy[label]['title']),
                'path': tokenizer.convert_tokens_to_ids(path), 
                'seen': False
            }
        
        features = []
        example_num = 0
        for case in tqdm(examples):
            text = case['text']
            text_id = tokenizer.convert_tokens_to_ids(text)
            category = case['category']
            for key, val in all_labels.items():
                label = 1 if val['label'] in category else 0
                target = val['path'] if use_path else val['label_id']
                features.append({
                        "source_ids": text_id, 
                        'category_ids': target, 
                        'label': label, 
                        'label_id': key,
                        'example_num': example_num}
                    )
            example_num += 1
        
        eval_features = {
            'features': features,
            'all_labels': all_labels
        }

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(eval_features, cached_features_file)
    
    return eval_features


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['bert', 'distilbert'], default='bert')
    parser.add_argument('--model_config', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, choices=['wos', 'yelp', 'qba'], default='wos')
    parser.add_argument('--data_dir', type=str, default="../../data/")
    parser.add_argument('--data_type', type=str, choices=['test', 'dev'], default="dev")
    parser.add_argument('--cached_feature_dir', type=str, default="cached_input/")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument('--checkpoint', type=str, default=None,
        help="The checkpoint of the model to be tested")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--do_lower_case", action="store_false")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=256,
         help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_source_seq_length", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=280)
    parser.add_argument("--pos_reward", type=float, default=1.0,
                        help="reward for positive cases")
    parser.add_argument("--reward_type", type=str, choices=['in_cate', 'path', 'hier'], default='hier',
                        help="type of reward")
    parser.add_argument("--use_path", action="store_true")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.fp16:
        try:
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    
    if args.cache_dir == 'cache':
        logger_name = os.path.join(args.output_dir, "{}_log.txt".format(args.data_type))
    elif args.cache_dir == 'rl_cache':
        logger_name = os.path.join(args.output_dir, "rl_{}_log.txt".format(args.data_type))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(
        format=LOG_FORMAT, 
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN, 
        filename=logger_name, 
        filemode='a')
    logger = logging.getLogger()

    log_info_msg = "========{}========".format(strftime("%Y-%b-%d-%H_%M_%S", gmtime()))
    logger.info(log_info_msg)
    logger.info(json.dumps(args.__dict__, sort_keys=True, indent=4))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    args.device = device

    set_random_seed(args.seed, args.n_gpu)

    if args.local_rank in [-1, 0]:
        chechpoint_dir = os.path.join(args.output_dir, args.cache_dir)
        if args.checkpoint is not None:
            checkpoints = [os.path.join(chechpoint_dir, "ckpt-{}".format(args.checkpoint))]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(chechpoint_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if args.mode == 'bert':
            tokenizer = BertTokenizer.from_pretrained(args.model_config, do_lower_case=args.do_lower_case)
        elif args.mode == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained(args.model_config, do_lower_case=args.do_lower_case)

        for checkpoint in tqdm(checkpoints):
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            
            if args.eval_all_checkpoints:
                checkpoint_num = checkpoint.split('ckpt-')[-1]
                args.checkpoint = checkpoint_num
            logger.info("Evaluating checkpoint: %s", checkpoint)
            if args.mode == 'bert':
                model = BertForSequenceClassification.from_pretrained(checkpoint)
            elif args.mode == 'distilbert':
                 model = DistilBertForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, logger, prefix=prefix)


if __name__ == "__main__":
    main()
