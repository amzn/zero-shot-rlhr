import os 
import sys 
import argparse
from tqdm import tqdm 
import json 
from sklearn.metrics import precision_score, recall_score, f1_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['a', 'm'], default='a', 
                        help='a - analysis, m - metrics')
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, default=None)
    parser.add_argument('--taxonomy', type=str, default='cached_taxonomy.json')
    parser.add_argument('--unseen_labels', type=str, default='unseen_labels.txt')
    parser.add_argument('--seen_labels', type=str, default='seen_labels.txt')
    args = parser.parse_args()
    file_dir = args.file1.split('/')[0]
    args.taxonomy = os.path.join(file_dir, args.taxonomy)
    args.unseen_labels = os.path.join(file_dir, args.unseen_labels)
    args.seen_labels = os.path.join(file_dir, args.seen_labels)
    return args


def read_taxonomy(taxonomy_file):
    print('Reading {} ...'.format(taxonomy_file))
    with open(taxonomy_file) as fin:
        taxonomy = json.load(fin)
    return taxonomy


def get_path(node, taxonomy):
    node_level = {}
    for one_node in node:
        level = taxonomy[one_node]['level']
        if level not in node_level:
            node_level[level] = []
        node_level[level].append(one_node)
    node_rec = {one_node: False for one_node in node}

    path = []
    sorted_level = sorted(node_level.keys(), reverse=True)
    for level in sorted_level:
        level_node = node_level[level]
        for one_node in level_node:
            if node_rec[one_node]:
                continue
            current_node = one_node
            one_path = []
            get_end = False
            while True:
                if current_node == "root":
                    get_end = True
                    break
                if current_node not in node:
                    break
                one_path.append(current_node)
                current_node = taxonomy[current_node]['parent']
            if get_end:
                one_path.reverse()
                path.append(one_path)
                for one_node in one_path:
                    node_rec[one_node] = True
    rest_node = []
    for level, level_node in node_level.items():
        for one_node in level_node:
            if not node_rec[one_node]:
                rest_node.append(one_node)
    return path, rest_node    


def analysis(taxonomy, res1, file1, res2=None, file2=None):
    for key, val in res1.items():
        truth = val['truth']
        pred1 = []
        for item in val['pred']:
            if item[1][1] > item[1][0]:
                pred1.append(item[0])
        pred2 = None
        if res2 != None:
            pred2 = []
            for item in res2[key]['pred']:
                if item[1][1] > item[1][0]:
                    pred2.append(item[0])
        truth_path, truth_rest_node = get_path(truth, taxonomy)
        pred1_path, pred1_rest_node = get_path(pred1, taxonomy)
        if res2 != None:
            pred2_path, pred2_rest_node = get_path(pred2, taxonomy)
        print("{} \npath: {}, rest node: {}".format('truth', truth_path, truth_rest_node))
        print("{} \npath: {}, rest node: {}".format(file1, pred1_path, pred1_rest_node))
        if res2 != None:
            print("{} \npath: {}, rest node: {}".format(file2, pred2_path, pred2_rest_node))
        input()


def ebf_score(res1, res2=None):
    def calculate_ebf(truth, pred):
        tp = len([x for x in pred if x in truth])
        fp = len([x for x in pred if x not in truth])
        fn = len([x for x in truth if x not in pred])
        if len(pred) == 0:
            p = 0.
        else:
            p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p == 0 and r == 0:
            f1 = 0.
        else:
            f1 = 2 * p * r / (p + r)
        return f1

    print("Calculating example-based f1...")
    ebf = [0., 0.]
    for key, val in tqdm(res1.items()):
        truth = val['truth']
        pred1 = []
        for item in val['pred']:
            if item[1][1] > item[1][0]:
                pred1.append(item[0])
        pred2 = None
        if res2 != None:
            pred2 = []
            for item in res2[key]['pred']:
                if item[1][1] > item[1][0]:
                    pred2.append(item[0])
        ebf1 = calculate_ebf(truth, pred1)
        ebf[0] += ebf1
        if res2 != None:
            ebf2 = calculate_ebf(truth, pred2)
            ebf[1] += ebf2
    ebf[0] /= len(res1)
    ebf[0] = round(ebf[0], 4)
    if res2 != None:
        ebf[1] /= len(res2)
        ebf[1] = round(ebf[1], 4)
    return ebf


def calculate_f1_score(labels, res1, res2=None):
    def calculate_macro_f1(label_dict, pred2_exists=False):
        precision = [0., 0.]
        recall = [0., 0.] 
        f_score = [0., 0.]
        cnt = 0
        for key, val in label_dict.items():
            truth = val['truth']
            if sum(truth) == 0:
                continue
            cnt += 1
            pred1 = val['pred_1']
            pred2 = val['pred_2']
            f_score[0] += f1_score(truth, pred1)
            precision[0] += precision_score(truth, pred1)
            recall[0] += recall_score(truth, pred1)
            if pred2_exists:
                f_score[1] += f1_score(truth, pred2)
                precision[1] += precision_score(truth, pred2)
                recall[1] += recall_score(truth, pred2)
        
        precision = [x /cnt for x in precision]
        precision = [round(x, 4) for x in precision]
        recall = [x /cnt for x in recall]
        recall = [round(x, 4) for x in recall]
        f_score = [x /cnt for x in f_score]
        f_score = [round(x, 4) for x in f_score]
        return precision, recall, f_score
    
    def calculate_micro_f1(label_dict, pred2_exists=False):
        precision = [0., 0.]
        recall = [0., 0.]
        f_score = [0., 0.]
        all_truth = []
        all_pred1 = []
        all_pred2 = []
        for key, val in label_dict.items():
            truth = val['truth']
            if sum(truth) == 0:
                continue
            pred1 = val['pred_1']
            pred2 = val['pred_2']
            all_truth.extend(truth)
            all_pred1.extend(pred1)
            all_pred2.extend(pred2)
        
        precision[0] = precision_score(all_truth, all_pred1)
        recall[0] = recall_score(all_truth, all_pred1)
        f_score[0] = f1_score(all_truth, all_pred1)
        if pred2_exists:
            precision[1] = precision_score(all_truth, all_pred2)
            recall[1] = recall_score(all_truth, all_pred2)
            f_score[1] = f1_score(all_truth, all_pred2)
        precision = [round(x, 4) for x in precision]
        recall = [round(x, 4) for x in recall]
        f_score = [round(x, 4) for x in f_score]
        return precision, recall, f_score

    print("Calculating f1 ...")
    label_dict = {key: {'truth': [0]*len(res1), 'pred_1': [0]*len(res1), 'pred_2': [0]*len(res1)} for key in labels}
    for key, val in tqdm(res1.items()):
        truth = val['truth']
        pred1 = []
        for item in val['pred']:
            if item[1][1] > item[1][0]:
                pred1.append(item[0])
        pred2 = None
        if res2 != None:
            pred2 = []
            for item in res2[key]['pred']:
                if item[1][1] > item[1][0]:
                    pred2.append(item[0])
        for one_truth in truth:
            if one_truth in labels:
                label_dict[one_truth]['truth'][int(key)] = 1
        for one_pred in pred1:
            if one_pred in labels:
                label_dict[one_pred]['pred_1'][int(key)] = 1
        if pred2 != None:
            for one_pred in pred2:
                if one_pred in labels:
                    label_dict[one_pred]['pred_2'][int(key)] = 1

    macro_precision, macro_recall, macro_f1 = calculate_macro_f1(label_dict, pred2_exists=res2 is not None)
    micro_precision, micro_recall, micro_f1 = calculate_micro_f1(label_dict, pred2_exists=res2 is not None)
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


def parent_child_violation(taxonomy, res1, file1, res2=None, file2=None):
    violated_nodes = [0, 0]
    print("Calculating path...")
    for key, val in tqdm(res1.items()):
        truth = val['truth']
        pred1 = []
        for item in val['pred']:
            if item[1][1] > item[1][0]:
                pred1.append(item[0])
        pred2 = None
        if res2 != None:
            pred2 = []
            for item in res2[key]['pred']:
                if item[1][1] > item[1][0]:
                    pred2.append(item[0])
        truth_path, truth_rest_node = get_path(truth, taxonomy)
        pred1_path, pred1_rest_node = get_path(pred1, taxonomy)
        if res2 != None:
            pred2_path, pred2_rest_node = get_path(pred2, taxonomy)
        violated_nodes[0] += len(pred1_rest_node)
        if res2 != None:
            violated_nodes[1] += len(pred2_rest_node)
    violated_nodes[0] /= len(res1)
    violated_nodes[0] = round(violated_nodes[0], 4)
    if res2 != None:
        violated_nodes[1] /= len(res2)
        violated_nodes[1] = round(violated_nodes[1], 4)
    return violated_nodes


def metric(taxonomy, unseen_label_file, seen_label_file, res1, file1, res2=None, file2=None):    
    seen_labels = []
    unseen_labels = []
    with open(unseen_label_file) as fin:
        for line in fin:
            unseen_labels.append(line.strip())
    with open(seen_label_file) as fin:
        for line in fin:
            seen_labels.append(line.strip())
    
    ebf = ebf_score(res1, res2)
    unseen_macro_precision, unseen_macro_recall, unseen_macro_f1, \
            unseen_micro_precision, unseen_micro_recall, unseen_micro_f1 = calculate_f1_score(unseen_labels, res1, res2)
    all_macro_precision, all_macro_recall, all_macro_f1, \
            all_micro_precision, all_micro_recall, all_micro_f1 = calculate_f1_score(unseen_labels+seen_labels, res1, res2)
    violated_nodes = parent_child_violation(taxonomy, res1, file1, res2, file2)

    print("****************************************")
    print(file1)
    print('parent-child violate rate: {}'.format(violated_nodes[0]))
    print('ebf score: {}'.format(ebf[0]))
    print("unseen macro - precision: {}, recall: {}, f1: {}".format(
                        unseen_macro_precision[0], unseen_macro_recall[0], unseen_macro_f1[0]))
    print("unseen micro - precision: {}, recall: {}, f1: {}".format(
                        unseen_micro_precision[0], unseen_micro_recall[0], unseen_micro_f1[0]))
    print("all macro - precision: {}, recall: {}, f1: {}".format(
                        all_macro_precision[0], all_macro_recall[0], all_macro_f1[0]))
    print("all micro - precision: {}, recall: {}, f1: {}".format(
                        all_micro_precision[0], all_micro_recall[0], all_micro_f1[0]))

    if res2 != None:
        print("****************************************")
        print(file2)
        print('parent-child violate rate: {}'.format(violated_nodes[1]))
        print('ebf score: {}'.format(ebf[1]))
        print("unseen macro - precision: {}, recall: {}, f1: {}".format(
                            unseen_macro_precision[1], unseen_macro_recall[1], unseen_macro_f1[1]))
        print("unseen micro - precision: {}, recall: {}, f1: {}".format(
                            unseen_micro_precision[1], unseen_micro_recall[1], unseen_micro_f1[1]))
        print("all macro - precision: {}, recall: {}, f1: {}".format(
                            all_macro_precision[1], all_macro_recall[1], all_macro_f1[1]))
        print("all micro - precision: {}, recall: {}, f1: {}".format(
                            all_micro_precision[1], all_micro_recall[1], all_micro_f1[1]))
    
    
def main():
    args = get_args()
    print("Reading {} ...".format(args.file1))
    with open(args.file1) as fin:
        res1 = json.load(fin)
    if args.file2 != None:
        print("Reading {} ...".format(args.file2))
        with open(args.file2) as fin:
            res2 = json.load(fin)
    else:
        res2 = None
    
    taxonomy = read_taxonomy(args.taxonomy)
    print(len(taxonomy))
    if args.mode == 'a':
        analysis(taxonomy, res1, args.file1, res2, args.file2)
    elif args.mode == 'm':
        metric(taxonomy, args.unseen_labels, args.seen_labels, res1, args.file1, res2, args.file2)


if __name__ == "__main__":
    main()