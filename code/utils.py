import os 
import sys 
import numpy as np 
from tqdm import tqdm 
import random
import json
import glob
import torch 
from collections import defaultdict


# read the taxonomy file and extract the ancestors
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


# read/load the training data
def load_and_cache_examples(
        train_file, taxonomy_file, seen_label_file, tokenizer, local_rank, cached_features_file, taxonomy_cache_file, logger):
    
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        training_data = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", train_file)
        seen_labels = []
        with open(seen_label_file) as fin:
            for line in fin:
                seen_labels.append(line.strip())

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
            "seen_labels": seen_labels
        }

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(training_data, cached_features_file)

    return training_data


# conver the tokens to ids
def convert_examples_to_id_for_bert(
        examples, taxonomy, seen_labels, local_rank, tokenizer, cached_features_file, negative_sample_num, use_path, logger, shuffle=True, rl_pretrain=False):

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        training_features = torch.load(cached_features_file)
    else:
        logger.info("Converting examples to id for bert")
        training_features = []
        for case in tqdm(examples):
            text = case['text']
            category = case['category']
            for cate in category:
                cate_token = taxonomy[cate]['title']
                cate_ancestor = taxonomy[cate]['ancestor']
                cate_ancestor_id = []
                for i, item in enumerate(cate_ancestor):
                    if i == 0:
                        cate_ancestor_id.extend(tokenizer.convert_tokens_to_ids(taxonomy[item]['title']))
                    else:
                        cate_ancestor_id.extend(tokenizer.convert_tokens_to_ids(['-'] + taxonomy[item]['title']))
                
                label = 0.8 if rl_pretrain else 1
                training_features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(text),
                    "category_ids": tokenizer.convert_tokens_to_ids(cate_token),
                    "category_ancestors": cate_ancestor_id,
                    "label": label
                })
            category_set = set(category)
            negative_count = 0
            while negative_count < negative_sample_num:
                sample_category = random.choice(seen_labels)
                if sample_category in category_set:
                    continue
                else:
                    negative_count += 1
                    sample_cate_token = taxonomy[sample_category]['title']
                    sample_cate_ancestor = taxonomy[sample_category]['ancestor']
                    sample_cate_ancestor_id = []
                    for i, item in enumerate(sample_cate_ancestor):
                        if i == 0:
                            sample_cate_ancestor_id.extend(tokenizer.convert_tokens_to_ids(taxonomy[item]['title']))
                        else:
                            sample_cate_ancestor_id.extend(tokenizer.convert_tokens_to_ids(['-'] + taxonomy[item]['title']))
                    label = 0.2 if rl_pretrain else 0
                    training_features.append({
                        "source_ids": tokenizer.convert_tokens_to_ids(text), 
                        'category_ids': tokenizer.convert_tokens_to_ids(sample_cate_token), 
                        "category_ancestors": sample_cate_ancestor_id,
                        'label': label})
                    category_set.add(sample_category)
        if shuffle:
            random.shuffle(training_features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(training_features, cached_features_file)
    
    return training_features


def convert_category_to_path(data, local_rank, logger):
    logger.info("Convert category to path")
    new_data = []
    example = data['examples']
    taxonomy = data['taxonomy']
    seen_labels = data['seen_labels']
    new_example = []
    for case in tqdm(example):
        text = case['text']
        category = case['category']
        category_record = {cate: False for cate in category}
        category_level = {i: [] for i in range(1, 5)}
        for cate in category:
            level = taxonomy[cate]['level']
            category_level[level].append(cate)
        path = []
        for i in range(4, 0, -1):
            for item in category_level[i]:
                if category_record[item]:
                    continue
                path.append(taxonomy[item]['ancestor'])
                for node in taxonomy[item]['ancestor']:
                    category_record[node] = True
        new_example.append({
            "text": text,
            "category": category,
            "path": path
        })
    new_data = {
        "examples": new_example,
        "taxonomy": taxonomy,
        "seen_labels": seen_labels
    }

    return new_data   


def build_taxonomy_id(taxonomy, seen_labels_id, tokenizer):
    taxonomy_id = {}
    for key, val in tqdm(taxonomy.items()):
        key_id = val['id']
        if key_id != taxonomy['root']['id'] and key_id not in seen_labels_id:
            continue
        parent_id = -1 if key=='root' else taxonomy[val['parent']]['id']
        level = val['level']
        text = key
        title_id = tokenizer.convert_tokens_to_ids(val['title'])
        children_id = [taxonomy[child]['id'] for child in val['children'] if taxonomy[child]['id'] in seen_labels_id]            
        ancester_id = [taxonomy[ances]['id'] for ances in val['ancestor']]
        taxonomy_id[key_id] = {
            'parent': parent_id, # index id
            'level': level,
            'text': text,
            'title': title_id, # input id
            'children': children_id, # index id
            'ancestor': ancester_id # index id
        }
    return taxonomy_id


def convert_examples_to_id_for_rl(
        data, tokenizer, local_rank, logger, shuffle=True):
    logger.info("Converting examples to id for bert")
    examples = data['examples']
    taxonomy = data['taxonomy']
    seen_labels = data['seen_labels']

    seen_labels_id = [taxonomy[key]['id'] for key in seen_labels]
    taxonomy_id = build_taxonomy_id(taxonomy, seen_labels_id, tokenizer)

    examples_id = []
    for case in tqdm(examples):
        text = case['text']
        category = case['category']
        path = case['path']
        text_id = tokenizer.convert_tokens_to_ids(text)
        category_id = [taxonomy[key]['id'] for key in category]
        path_id = [[taxonomy[node]['id'] for node in one_path] for one_path in path]
        examples_id.append({
            "text": text_id,
            "category": category_id,
            "path": path_id
        })

    new_data = {
        "examples": examples,
        "examples_id": examples_id,
        "taxonomy": taxonomy,
        "taxonomy_id": taxonomy_id, # remove the unseen labels from the taxonomy
        "seen_labels": seen_labels,
        "seen_labels_id": seen_labels_id
    }

    return new_data


# get the max epoch model if continue to train
def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    os.path.basename(output_dir)
    both_set = set([int(os.path.basename(fn).split('.')[1]) for fn in fn_model_list]
                   ) & set([int(os.path.basename(fn).split('.')[1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


def init_rl_batch(batch):
    batch_list = []
    for x in zip(*batch):
        batch_list.append(x)
    return batch_list


# build corresponding running dirs
def make_all_dirs(paras):
    if not os.path.exists(paras['cache_dir']):
        os.makedirs(paras['cache_dir'])
    if not os.path.exists(paras['cached_feature_dir']):
        os.makedirs(paras['cached_feature_dir'])