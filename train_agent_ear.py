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

class PretrainBatchData():
    def __init__(self, state_list, label_list, use_gpu, output_check=False):
        batch_size = len(state_list)
        self.state_list = torch.tensor(state_list)
        self.label_list = torch.tensor(label_list)
        if use_gpu:
            self.state_list = self.state_list.cuda()
            self.label_list = self.label_list.cuda()
        if output_check:
            self.output()

    def output(self):
        print("--------------------------")
        print("state_list:", self.state_list)
        print("label_list:", self.label_list)


def load_pretrain_data(pretrain_data_path, pretrain_batch_size, use_gpu=False):
    with open(pretrain_data_path, 'rb') as f:
        state_entropy_list, state_att_prefer_list, state_item_prefer_list, \
                        state_convhis_list, state_len_list, label_list = pickle.load(f)

    dialogue_state_list = []
    for state_entropy, state_att, state_convhis, state_len \
        in zip(state_entropy_list, state_att_prefer_list, state_convhis_list, state_len_list):
        dialogue_state = state_entropy + state_att + state_convhis + state_len
        dialogue_state_list.append(dialogue_state)

    assert len(dialogue_state_list) == len(label_list)

    all_list = list(zip(dialogue_state_list, label_list)) 
    random.shuffle(all_list)   
    dialogue_state_list, label_list = zip(*all_list)


    data_num = len(dialogue_state_list)
    data_num = data_num // 20

    test_state_list = dialogue_state_list[:data_num:]
    train_state_list = dialogue_state_list[data_num:]
    test_label_list = label_list[:data_num]
    train_label_list = label_list[data_num:]     

    print("train_state_list: {}, test_state_list: {}".format(len(train_state_list), len(test_state_list)))
    return train_state_list, train_label_list, test_state_list, test_label_list

def make_batch_data(state_list, label_list, batch_size, use_gpu):
    all_list = list(zip(state_list, label_list)) 
    random.shuffle(all_list)   
    state_list, label_list = zip(*all_list)

    max_iter = len(state_list)//batch_size
    if max_iter * batch_size < len(state_list):
        max_iter += 1

    batch_data_list = []
    for index in range(max_iter):
        left_index = index * batch_size
        right_index = (index+1) * batch_size
        batch_data = PretrainBatchData(state_list[left_index:right_index], label_list[left_index:right_index], use_gpu)
        batch_data_list.append(batch_data)
    return batch_data_list

def pretrain(agent):
    use_gpu = False
    pretrain_data_path = "./data/agent/RL_pretrain_data_2.pkl"
    pretrain_epoch_num = 100
    pretrain_weight_decay = 1e-5
    pretrain_lr = 1e-2
    pretrain_optimizer = optim.Adam(agent.DPN.parameters(), lr=pretrain_lr, weight_decay=pretrain_weight_decay)
    label_weight = torch.tensor([1.] * 33 + [1.])
    pretrain_criterion = nn.CrossEntropyLoss(weight = label_weight)
    pretrain_batch_size = 512
    pretrain_save_step = 1

    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("pretrain-agentear-{}-lr-{}-reg-{}-bs-{}.log"\
                        .format(date_str,str(pretrain_lr),str(pretrain_weight_decay),str(pretrain_batch_size)))

    print("prepare pretrain data...")
    train_state_list, train_label_list, test_state_list, test_label_list \
        = load_pretrain_data(pretrain_data_path, pretrain_batch_size, use_gpu)
    pretrain_test_data_list = make_batch_data(test_state_list, test_label_list, pretrain_batch_size, use_gpu)
    time_str = datetime.datetime.now().isoformat()
    print("{} start pretraining ...".format(time_str))
    print("lr: {:g}, batch_size: {}".format(pretrain_lr, pretrain_batch_size))
    best_acc = 0.
    best_acc_count = 0

    for _ in range(pretrain_epoch_num):
        print("epoch: ", _)
        loss_list = []
        pretrain_train_data_list = make_batch_data(train_state_list, train_label_list, pretrain_batch_size, use_gpu)
        pretrain_data_index_list = [_ for _ in range(len(pretrain_train_data_list))]
        # random.shuffle(pretrain_data_index_list)
        for pretrain_data_index in tqdm(pretrain_data_index_list, ncols=0):
            t_batch_data = pretrain_train_data_list[pretrain_data_index]
            output = agent.pretrain_batch_data_input(t_batch_data, True)
            loss = pretrain_criterion(output, t_batch_data.label_list)
            loss_list.append(loss.item())
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            time_str = datetime.datetime.now().isoformat()
        epoch_loss = np.mean(np.array(loss_list))
        print("{}: epoch {}, loss {:g}".format(time_str, _, epoch_loss))

        if _ % pretrain_save_step == 0:
            print("start evaluation")
            pre_label_list = []
            gt_label_list = []
            for e_batch_data in tqdm(pretrain_test_data_list, ncols=0):
                output = agent.pretrain_batch_data_input(e_batch_data, False)
                pre_label_list.extend(output.tolist())
                gt_label_list.extend(e_batch_data.label_list.tolist())
            cur_acc = accuracy_score(gt_label_list, pre_label_list)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_acc_count = 0
                agent.save_model(True)
            else:
                best_acc_count += 1
            print("{}: epoch {}, accuracy {:g}, best accuracy {:g}".format(time_str, str(_), cur_acc, best_acc))
            print(classification_report(gt_label_list, pre_label_list))

            if best_acc_count == 5:
                break

def standard_reward(a):
    return (a - np.mean(a)) / (np.std(a) + eps)

def PG_train(agent, load_model_type, dm):
    use_gpu = False
    env = dm
    PG_lr = 0.001
    PG_discount_rate = 0.7
    PG_train_data_path = "./data/agent/PG_eval_list.pkl"
    PG_test_data_path = "./data/agent/PG_test_list.pkl"
    PG_epoch_num = 100
    # PG_optimizer = optim.RMSprop(agent.DPN.parameters(), lr=PG_lr,)
    PG_optimizer = optim.SGD(agent.DPN.parameters(), lr=PG_lr,)
    PG_save_step = 1

    with open(PG_train_data_path, "rb") as f:
        PG_train_data_list = pickle.load(f)
    with open(PG_test_data_path, "rb") as f:
        PG_test_data_list = pickle.load(f)

    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("PG-agentear-{}-lr-{}.log"\
                        .format(date_str,str(PG_lr)))
    print("PG_train_data_list: {}, PG_test_data_list:{}".format(len(PG_train_data_list), len(PG_test_data_list)))

    agent.set_env(env)
    if load_model_type == "pretrain":
        print("load pretrain model ...")
        agent.load_model(True)
    elif load_model_type == "PG":
        print("load PG model ...")
        agent.load_model(False)
    else:
        print("no pretrian model...")

    time_str = datetime.datetime.now().isoformat()
    print("{} start PG ...".format(time_str))
    print("lr: {:g}".format(PG_lr))

    best_average_reward = 0.
    best_average_turn = 100.
    best_success_rate = 0.
    best_count = 0
    each_epoch_len = len(PG_train_data_list)

    for _ in range(PG_epoch_num):
        print("epoch: ", _)
        epoch_reward_sum = 0.
        epoch_turn_sum = 0.
        epoch_success_sum = 0.
        PG_data_index_list = [_ for _ in range(len(PG_train_data_list))]
        random.shuffle(PG_data_index_list)
        PG_data_index_list = PG_data_index_list[:each_epoch_len]
        for PG_data_index in tqdm(PG_data_index_list,ncols=0):
            t_data = PG_train_data_list[PG_data_index]
            action_pool, reward_pool = agent.PG_train_one_episode(t_data)
            epoch_reward_sum += sum(reward_pool)
            epoch_turn_sum += len(reward_pool)
            epoch_success_sum += (reward_pool[-1] > 0.)

            total_reward = 0.
            for index in reversed(range(len(reward_pool))):
                total_reward = total_reward * PG_discount_rate + reward_pool[index]
                reward_pool[index] = total_reward

            reward_pool = np.array(reward_pool)
            # reward_pool = standard_reward(reward_pool)

            reward_pool_tensor = torch.from_numpy(reward_pool)
            action_pool_tensor = torch.stack(action_pool, 0)
            if use_gpu:
                reward_pool_tensor = reward_pool_tensor.cuda()
                action_pool_tensor = action_pool_tensor.cuda()

            loss = torch.sum(torch.mul(action_pool_tensor, reward_pool_tensor).mul(-1))
            PG_optimizer.zero_grad()
            loss.backward()
            PG_optimizer.step()

        time_str = datetime.datetime.now().isoformat()
        print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}".\
                format(time_str, _, epoch_reward_sum/len(PG_train_data_list), \
                    epoch_turn_sum/len(PG_train_data_list), epoch_success_sum/len(PG_train_data_list)))

        if (_+1) % PG_save_step == 0:
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
            if average_turn < best_average_turn:
                best_average_reward = average_reward
                best_average_turn = average_turn
                best_success_rate = success_rate
                agent.save_model(False) 
                best_count = 0
            else:
                best_count += 1

            time_str = datetime.datetime.now().isoformat()
            print("{}: test epoch {}, average_reward {:g}, average_turn {:g}, success_rate {:g}, \
                    best_average_reward: {:g}, best_average_turn {:g}, best_success_rate {:g}"\
                    .format(time_str, _, average_reward, average_turn, success_rate, \
                        best_average_reward, best_average_turn, best_success_rate))

            if best_count == 10:
                break


parser = argparse.ArgumentParser(description='train ear agent')
parser.add_argument('--mode', type=str, 
                    help='choose from pretrain or PG')
args = parser.parse_args()

if args.mode == "pretrain":
    # preatrain
    agent = AgentEAR(AgentEARConfig(), None)
    pretrain(agent)
elif args.mode == "PG":
    # PG
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
    PG_train(agent, "pretrain", dm)
else:
    print("Not support {}. Choose from pretrain or PG".format(args.mode))