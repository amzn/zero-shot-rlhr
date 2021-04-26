import os
import sys 
import numpy as np
import random
import scipy

import torch 


class Policy(object):
    def __init__(self, taxonomy_id, path_max_length, right_node_reward):
        self.taxonomy_id = taxonomy_id
        self.right_node_reward = right_node_reward
        self.wrong_node_reward = -0.1
        self.path_max_length = path_max_length
    
    def init_state(self, current_node, mode):
        if mode == 'greedy':
            self.greedy_state = [ {} for x in current_node ]
            self.state = self.greedy_state
        elif mode == 'random':
            self.random_state = [ {} for x in current_node ]
            self.state = self.random_state
        self.level = 0
    
    def update_state(self, index, prob, current_node, batch_index):
        length = index.size()[0]
        for i in range(length):
            node = current_node[index[i]]
            node_state = prob[index[i]]
            batch_num = batch_index[index[i]]
            self.state[int(batch_num)][int(node)] = {'prob': node_state, 'level': self.level}
        self.level += 1
    
    def select_current_node(self, index, text, current_node, batch_index):
        text_ret = [text[i] for i in index]
        current_node_ret = [current_node[i] for i in index]
        batch_index_ret = [batch_index[i] for i in index]
        return text_ret, np.asarray(current_node_ret), np.asarray(batch_index_ret)

    def get_children(self, text, current_node, batch_index):
        new_text = []
        new_node = []
        new_batch_index = []
        for i, node in enumerate(current_node):
            children = self.taxonomy_id[node]['children']
            new_node.extend(children)
            new_text += [text[i] for x in children]
            new_batch_index += [batch_index[i]] * len(children)
        return new_text, np.asarray(new_node), np.asarray(new_batch_index)
    
    def get_greedy_action(self, prob):
        greedy_index = torch.nonzero(prob[:, 1] > prob[:, 0], as_tuple=False)
        return greedy_index.squeeze(1)
    
    def get_random_action(self, prob, batch_index, batch_size, K):
        prob = prob.detach().cpu().numpy()
        action = []
        batch_prob = [[] for i in range(batch_size)]
        batch_action = [[] for i in range(batch_size)]
        for i in range(len(prob)):
            index_ = batch_index[i]
            batch_prob[index_].append(prob[i])
            batch_action[index_].append(i)
        for i in range(batch_size):
            batch_selected_actions = []
            for j, one_weight in enumerate(batch_prob[i]):
                one_action = np.random.choice([0, 1], p=one_weight)
                if one_action == 1:
                    batch_selected_actions.append(batch_action[i][j])
            if len(batch_selected_actions) > K:
                batch_selected_actions = random.sample(batch_selected_actions, K)
            action.extend(batch_selected_actions)
        action = sorted(action)
        return torch.Tensor(action).to(torch.int64)

    def get_path(self, state):
        node_level = {}
        for key, val in state.items():
            level = val['level']
            if level not in node_level:
                node_level[level] = []
            node_level[level].append(key)
        node_rec = {one_node: False for one_node in state.keys()}
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
                    if current_node == 0:
                        get_end = True
                        break
                    if current_node not in state:
                        break
                    one_path.append(current_node)
                    current_node = self.taxonomy_id[current_node]['parent']
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
    
    def path_reward(self, state, category, path):  
        def judge_path(path, all_path):
            for one_path in all_path:
                flag = True
                for node in path:
                    if node not in one_path:
                        flag = False
                        break
                if flag:
                    return True
                else:
                    continue
            return False         
        reward = []
        for i, one_state in enumerate(state):
            predicted_path, rest_node = self.get_path(one_state)
            batch_reward = 0.
            for one_path in predicted_path:
                path_reward = 0.
                if judge_path(one_path, path[i]):
                    for one_node in one_path:
                        path_reward += self.right_node_reward * (-one_state[one_node]['prob'][1])
                else:
                    for one_node in one_path:
                        path_reward += self.wrong_node_reward * (-one_state[one_node]['prob'][1])
                batch_reward += path_reward
            for one_node in rest_node:
                batch_reward += -1 * (-one_state[one_node]['prob'][1])
            reward.append(batch_reward)
        return reward

    def get_reward(self, state, category, path):
        return self.path_reward(state, category, path)
        

