import multiprocessing as mp
import time
from queue import Queue
import sys
import os
import random
from tqdm import tqdm
import datetime
import json
import math
import pickle
import torch
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from recommendersystem.recsys import recsys
from recommendersystem.recsysconfig import recsysconfig
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

random.seed(2023)

user_num = 1801
item_num = 7432
attribute_num = 33
att_num = attribute_num

with open("./data/item_info.json", "r") as f:
    item_info = json.load(f)
new_item_info = {}
for item in item_info:
    new_item_info[int(item)] = set(item_info[item])
item_info = new_item_info

with open("./data/user_info.json", "r") as f:
    user_info = json.load(f)
new_user_info = {}
for user in user_info:
    new_user_info[int(user)] = set(user_info[user])
user_info = new_user_info

with open("./data/attribute_tree_dict.json", "r") as f:
    attribute_tree_dict = json.load(f)
attribute_tree = trans_index(attribute_tree_dict) 
for key in attribute_tree:
    attribute_tree[key] = set(attribute_tree[key])

adj_index = [[], []]
for user in user_info:
    for item in user_info[user]:
        adj_index[0].append(user)
        adj_index[1].append(item + user_num)
        adj_index[1].append(user)
        adj_index[0].append(item + user_num)        
for item in item_info:
    for att in item_info[item]:
        adj_index[0].append(item + user_num)
        adj_index[1].append(att + item_num + user_num)
        adj_index[1].append(item + user_num)
        adj_index[0].append(att + item_num + user_num)        
adj_index = torch.tensor(adj_index)

def consumer(q):
    state_entropy_list = []
    state_att_prefer_list = []
    state_item_prefer_list = []
    state_convhis_list = []
    state_len_list = []
    action_list = []
    exit_flag = False
    while not exit_flag:
        while not q.empty():
            res=q.get()
            if res is None:
                exit_flag = True
                break
            state_entropy, state_att_prefer, state_item_prefer, state_convhis, state_len, action = res

            state_entropy_list.append(state_entropy)
            state_att_prefer_list.append(state_att_prefer)
            state_item_prefer_list.append(state_item_prefer)
            state_convhis_list.append(state_convhis)
            state_len_list.append(state_len)
            action_list.append(action)

    with open('./data/agent/RL_pretrain_data_2.pkl','wb') as f:
        pickle.dump([state_entropy_list, state_att_prefer_list, state_item_prefer_list, \
                        state_convhis_list, state_len_list, action_list], f)

def job(train_pair,q):
    ch = ConvHis(ConvHisConfig())
    agent = AgentRule(AgentRuleConfig(), ch)
    rec = recsys(recsysconfig(), convhis=ch, use_gpu=False)
    model_name = "iter60-2020-06-09"
    rec.load_model(model_name, True)
    rec.init_eval(adj_index)
    usersim = UserSim(UserSimConfig())
    dm = DialogueManager(DialogueManagerConfig(), rec, agent, usersim, ch)

    train_index_list = [_ for _ in range(len(train_pair))]
    for index in tqdm(train_index_list, ncols=0):
        user, item = train_pair[index]
        pos_attribute_set = item_info[item]
        parent_att_list = []
        for parent_att in attribute_tree:
            if len(pos_attribute_set & attribute_tree[parent_att]) > 0:
                parent_att_list.append(parent_att)
        random.shuffle(parent_att_list)
        max_iter = 1

        for parent_att in parent_att_list[:max_iter]:
            dm.initialize_dialogue(user, item, True, parent_att)
            is_over = False
            while not is_over:
                state_entropy = ch.get_attribute_entropy().copy()
                state_att_prefer = rec.get_attribute_preference().copy()
                state_item_prefer = rec.get_item_prefer_state().copy()
                state_convhis = ch.get_convhis_vector().copy()
                state_len = ch.get_length_vector().copy()
                is_over, reward, _ = dm.next_turn()
                action = dm.get_current_agent_action()
                q.put([state_entropy, state_att_prefer, state_item_prefer, state_convhis, state_len, action])

if __name__=='__main__':
    q = mp.Queue()
    process = []
    process_num = 4

    train_pair = []
    with open("./data/train_user_item_pair.txt", "r") as f:
        for line in f.readlines():
            if len(line) <= 3:
                continue
            user, item = line.split(',')
            user = int(user.strip())
            item = int(item.strip())
            train_pair.append([user, item])
    train_pair_len = len(train_pair)
    random.shuffle(train_pair)

    batch_size = math.ceil(len(train_pair)/process_num)
    for index in range(process_num):
        t = mp.Process(target=job,args=(train_pair[batch_size*index:batch_size*(index+1)],q))
        t.start()
        process.append(t)
    cos = mp.Process(target=consumer,args=(q,))
    cos.start()
    for p in process:
        p.join()
    q.put(None)
    cos.join()
