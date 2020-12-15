import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import random
import pickle
import datetime
import numpy as np
from agents.DeepPolicyNetwork import TwoLayersModel

class AgentEAR():
    def __init__(self, config, convhis):
        self.convhis = convhis
        self.use_gpu = config.use_gpu
        self.DPN = TwoLayersModel(config)
        self.DPN_model_path = config.DPN_model_path
        self.DPN_model_name = config.DPN_model_name
        self.aciton_len = config.output_dim

        self.rec = None
        self.env = None

    def set_rec_model(self, rec_model):
        self.rec = rec_model

    def set_env(self, env):
        self.env = env

    def init_episode(self):
        self.DPN.eval()
        pass

    def next_turn(self):
        state_entropy = self.convhis.get_attribute_entropy()
        state_prefer = self.rec.get_attribute_preference()
        state_convhis = self.convhis.get_convhis_vector()
        state_len = self.convhis.get_length_vector()

        state_entropy = torch.tensor(state_entropy)
        state_prefer = torch.tensor(state_prefer)
        state_convhis = torch.tensor(state_convhis)
        state_len = torch.tensor(state_len)
        if self.use_gpu:
            state_entropy = state_entropy.cuda()
            state_prefer = state_prefer.cuda()
            state_convhis = state_convhis.cuda()
            state_len = state_len.cuda()

        state = torch.cat([state_entropy, state_prefer, state_convhis, state_len], -1)
        attribute_distribution = self.DPN(state, True)

        return int(attribute_distribution.argmax())

    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        torch.save(self.DPN.state_dict(), "/".join([self.DPN_model_path, self.DPN_model_name + name_suffix]))

    def load_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))

    def pretrain_batch_data_input(self, batchdata, is_train):
        if is_train:
            self.DPN.train()
        else:
            self.DPN.eval()

        output = self.DPN(batchdata.state_list, not is_train)
        if not is_train:
            output = torch.argmax(output, -1)
        return output

    def PG_train_one_episode(self, t_data):
        self.DPN.train()
        state_pool = []
        action_pool = []
        reward_pool = [] 

        state = self.env.initialize_episode(t_data[0], t_data[1])
        IsOver = False
        while not IsOver:
            attribute_distribution = self.DPN(state, True)
            c = Categorical(probs = attribute_distribution)

            i = 0
            action = c.sample()
            asked_list = self.convhis.get_asked_list()
            while(i<1000):
                if int(action) == self.aciton_len - 1:
                    break
                if int(action) not in asked_list:
                    break
                action = c.sample()
                i += 1

            IsOver, next_state, reward = self.env.step(int(action))                
            state_pool.append(state)
            action_pool.append(c.log_prob(action))
            reward_pool.append(reward)
            if not IsOver:
                state = next_state

        return action_pool, reward_pool

    def PG_eva_one_episode(self, t_data, silence=True):
        self.DPN.eval()
        total_reward = 0.
        turn_count = 0
        success = 0

        state = self.env.initialize_episode(t_data[0], t_data[1], t_data[2], silence)
        IsOver = False
        while not IsOver:
            turn_count += 1
            attribute_distribution = self.DPN(state, True)

            asked_list = torch.tensor(self.convhis.get_asked_list())
            if self.use_gpu:
                asked_list = asked_list.cuda()
            attribute_distribution[asked_list] = 0.

            action = int(attribute_distribution.argmax())
            IsOver, next_state, reward = self.env.step(int(action)) 
            total_reward += reward
            if not IsOver:
                state = next_state
            else:
                if reward > 0.:
                    success = 1
                else:
                    success = 0

        return total_reward, turn_count, success

    def test_one_episode(self, t_data, silence=True):
        self.DPN.eval()
        total_reward = 0.
        turn_count = 0
        success = 0
        action_list = []

        state = self.env.initialize_episode(t_data[0], t_data[1], t_data[2], silence)
        IsOver = False
        while not IsOver:
            turn_count += 1
            attribute_distribution = self.DPN(state, True)

            asked_list = torch.tensor(self.convhis.get_asked_list())
            if self.use_gpu:
                asked_list = asked_list.cuda()
            attribute_distribution[asked_list] = 0.

            action = int(attribute_distribution.argmax())
            action_list.append(action)
            IsOver, next_state, reward = self.env.step(int(action)) 
            total_reward += reward
            if not IsOver:
                state = next_state
            else:
                if reward > 0.:
                    success = 1
                else:
                    success = 0

        return total_reward, turn_count, success, action_list   