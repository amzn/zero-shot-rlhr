import os 
import sys 
import json 
from tqdm import tqdm 
import random

import torch 
import utils


class DatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_seq_len,
            cls_id, sep_id, pad_id, mask_id,
            offset, num_training_instances, use_path):
        self.features = features
        self.max_source_len = max_source_len
        self.max_seq_len = max_seq_len
        self.offset = offset
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.num_training_instances = num_training_instances
        self.use_path = use_path
    
    def __len__(self):
        return int(self.num_training_instances)

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
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len)
        if self.use_path:
            target_ids = self.__trunk(feature["category_ancestors"], self.max_seq_len - self.max_source_len)
        else:
            target_ids = self.__trunk(feature["category_ids"], self.max_seq_len - self.max_source_len)

        input_ids = source_ids + target_ids
        segment_ids = [0]*len(source_ids) + [1]*len(target_ids) + [0]*(self.max_seq_len-len(input_ids))
        input_mask = [1]*len(input_ids) + [0]*(self.max_seq_len-len(input_ids))
        input_ids = self.__pad(input_ids, self.max_seq_len)
        label_id = feature["label"]
        return input_ids, input_mask, segment_ids, label_id


class DatasetForRL(torch.utils.data.Dataset):
    def __init__(
            self, data, max_source_len, max_seq_len,
            cls_id, sep_id, pad_id, mask_id, node_sep_id,
            offset, num_training_instances, use_path):
        self.examples = data['examples']
        self.taxonomy = data['taxonomy']
        self.seen_labels = data['seen_labels']
        self.examples_id = data['examples_id']
        self.taxonomy_id = data['taxonomy_id']
        self.seen_labels_id = data['seen_labels_id']
        self.max_source_len = max_source_len
        self.max_seq_len = max_seq_len
        self.offset = offset
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.node_sep_id = node_sep_id
        self.num_training_instances = num_training_instances
        self.use_path = use_path

    def __len__(self):
        return int(self.num_training_instances)

    def __getitem__(self, idx):
        idx = (self.offset + idx) % len(self.examples)
        data = self.examples_id[idx]
        text = data['text']
        category = data['category']
        path = data['path']
        return text, category, path, self.taxonomy['root']['id'], 
    
    def __build_path(self, ancestor_id):
        ancestor_path = []
        for i, id_ in enumerate(ancestor_id):
            ancestor_tokens = self.taxonomy_id[id_]['title']
            if i != 0:
                ancestor_path += [self.node_sep_id]
            ancestor_path += ancestor_tokens
        return ancestor_path
    
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

    def build_batch(self, text, current_node):
        if self.use_path:
            target = [self.taxonomy_id[i]['ancestor'] for i in current_node]
        else:
            target = [self.taxonomy_id[i]['title'] for i in current_node]
        batch = []
        for i, one_text in enumerate(text):
            target_ids = self.__build_path(target[i]) if self.use_path else target[i]
            source_ids = self.__trunk([self.cls_id] + one_text, self.max_source_len)
            target_ids = self.__trunk(target_ids, self.max_seq_len - self.max_source_len)
            input_ids = source_ids + target_ids
            segment_ids = [0]*len(source_ids) + [1]*len(target_ids) + [0]*(self.max_seq_len-len(input_ids))
            input_mask = [1]*len(input_ids) + [0]*(self.max_seq_len-len(input_ids))
            input_ids = self.__pad(input_ids, self.max_seq_len)
            batch.append([input_ids, input_mask, segment_ids])
        batch_tensor = utils.batch_list_to_batch_tensors(batch)
        return batch_tensor
    


    