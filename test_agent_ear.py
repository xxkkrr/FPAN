import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json
import random
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from utils.LogPrint import Logger
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentRule import AgentRule
from agents.AgentRuleConfig import AgentRuleConfig
from agents.AgentEAR import AgentEAR
from agents.AgentEARConfig import AgentEARConfig
from recommendersystem.recsys import recsys
from recommendersystem.recsysconfig import recsysconfig
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig

random.seed(1021)
eps = np.finfo(np.float32).eps.item()

def agent_eval(agent, load_model_type, dm):
    use_gpu = False
    env = dm
    PG_test_data_path = "./data/agent/PG_test_list.pkl"
    with open(PG_test_data_path, "rb") as f:
        PG_test_data_list = pickle.load(f)

    agent.set_env(env)
    if load_model_type == "pretrain":
        print("load pretrain model ...")
        agent.load_model(True)
    elif load_model_type == "PG":
        print("load PG model ...")
        agent.load_model(False)
    else:
        print("no pretrian model...")

    sum_reward = 0.
    sum_turn = 0
    sum_success = 0
    episode_num = 0
    for e_data in tqdm(PG_test_data_list,ncols=0):
        reward, turn, success = agent.PG_eva_one_episode(e_data)
        episode_num += 1
        sum_reward += reward
        sum_turn += turn
        sum_success += success  

    average_reward = float(sum_reward)/episode_num
    average_turn = float(sum_turn)/episode_num
    success_rate = float(sum_success)/episode_num

    time_str = datetime.datetime.now().isoformat()
    print("{}: average_reward {:g}, average_turn {:g}, success_rate {:g}"\
            .format(time_str, average_reward, average_turn, success_rate))


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

ch = ConvHis(ConvHisConfig())
agent = AgentEAR(AgentEARConfig(), ch)
rec = recsys(recsysconfig(), convhis=ch)
model_name = "iter60-2020-06-09"
rec.load_model(model_name, True)
rec.init_eval(adj_index)
usersim = UserSim(UserSimConfig())
dm = DialogueManager(DialogueManagerConfig(), rec, agent, usersim, ch)
agent_eval(agent, "PG", dm)