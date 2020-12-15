import sys
import os
import random
from tqdm import tqdm
import datetime
import json
import math
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import argparse
from recommendersystem.recsys import recsys
from recommendersystem.recsysconfig import recsysconfig
from utils.LogPrint import Logger
from utils.build_train_rec_data_loader_new import *
from utils.build_test_rec_data_loader_new import *

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

def fileter_test_info(test_info):
    new_test_info = []
    for test in test_info:
        candidate_list = test['candidate_list']
        if len(candidate_list) > 1:
            new_test_info.append(test)
    return new_test_info

use_gpu = False

rec = recsys(recsysconfig(), use_gpu=use_gpu)

with open("./data/user_info.json", "r") as f:
    user_info = json.load(f)
new_user_info = {}
for user in user_info:
    new_user_info[int(user)] = set(user_info[user])
user_info = new_user_info

with open("./data/item_info.json", "r") as f:
    item_info = json.load(f)
new_item_info = {}
for item in item_info:
    new_item_info[int(item)] = set(item_info[item])
item_info = new_item_info

with open("./data/attribute_tree_dict.json", "r") as f:
    attribute_tree_dict = json.load(f)
att_tree_dict = trans_index(attribute_tree_dict)

train_info_path = "./data/rec/rec_train_data_with_rule_multi_2.txt"

with open("./data/rec/rec_test_data_with_rule_2.txt", "r") as f:
    test_info = json.load(f)
test_info = fileter_test_info(test_info)

user_num = 1801
item_num = 7432
attribute_num = 33

att_num = attribute_num

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

if use_gpu:
    adj_index = adj_index.cuda()

def rec_test(test_model_path = None):
    if test_model_path is not None:
        if use_gpu:
            rec.rec.load_state_dict(torch.load(test_model_path))
        else:
            rec.rec.load_state_dict(torch.load(test_model_path,  map_location='cpu'))
    rec.init_eval(adj_index)

    test_batch_size = 512
    test_shuffle = False
    test_num_threads = 2

    att_auc_list = []
    item_auc_list = []
    att_auc_list_2 = []
    item_auc_list_2 = []

    test_loader = build_test_item_att_loader(test_info, user_info, item_info, att_tree_dict, \
                                                att_num, item_num, use_gpu=use_gpu, \
                                                batch_size=test_batch_size, shuffle=test_shuffle, num_threads=test_num_threads)
    for batch_data in tqdm(test_loader, ncols=0):
        user_list, item_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, \
            neg_item_list, neg_item_mask_1, neg_item_mask_2, \
            test_attribute_list, test_attribute_mask, test_attribute_len, test_attribute_label, \
            test_item_list, test_item_mask, test_item_len, test_item_label = batch_data

        if use_gpu:
            user_list = user_list.cuda()
            item_list = item_list.cuda()
            pos_att_list = pos_att_list.cuda()
            pos_att_mask = pos_att_mask.cuda()
            neg_att_list = neg_att_list.cuda()
            neg_att_mask = neg_att_mask.cuda()
            neg_item_list = neg_item_list.cuda()
            neg_item_mask_1 = neg_item_mask_1.cuda()
            neg_item_mask_2 = neg_item_mask_2.cuda()
            test_attribute_list = test_attribute_list.cuda()
            test_attribute_mask = test_attribute_mask.cuda()
            test_item_list = test_item_list.cuda()
            test_item_mask = test_item_mask.cuda()

        rec.rec.get_current_user_embedding(user_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, neg_item_list, neg_item_mask_1)
        attribute_score = rec.rec.get_current_att_score(user_list, test_attribute_list)
        attribute_score = attribute_score.masked_fill(test_attribute_mask==False, -1e9)
        item_score = rec.rec.get_current_item_score(user_list, test_item_list)
        item_score = item_score.masked_fill(test_item_mask==False, -1e9)

        attribute_score_list = attribute_score.cpu().detach().numpy()
        item_score_list = item_score.cpu().detach().numpy()

        for each_att_score, att_len, att_label in zip(attribute_score_list, test_attribute_len, test_attribute_label):
            if sum(att_label) == 0:
                continue
            each_att_score_ = each_att_score[:att_len]
            auc = roc_auc_score(att_label, each_att_score_)
            att_auc_list.append(auc)

        for each_item_score, item_len, item_label in zip(item_score_list, test_item_len, test_item_label):
            each_item_score_ = each_item_score[:item_len]
            auc = roc_auc_score(item_label, each_item_score_)
            item_auc_list.append(auc)

        # with item neg
        rec.rec.get_current_user_embedding(user_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, neg_item_list, neg_item_mask_2)
        attribute_score = rec.rec.get_current_att_score(user_list, test_attribute_list)
        attribute_score = attribute_score.masked_fill(test_attribute_mask==False, -1e9)
        item_score = rec.rec.get_current_item_score(user_list, test_item_list)
        item_score = item_score.masked_fill(test_item_mask==False, -1e9)

        attribute_score_list = attribute_score.cpu().detach().numpy()
        item_score_list = item_score.cpu().detach().numpy()

        for each_att_score, att_len, att_label in zip(attribute_score_list, test_attribute_len, test_attribute_label):
            if sum(att_label) == 0:
                continue
            each_att_score_ = each_att_score[:att_len]
            auc = roc_auc_score(att_label, each_att_score_)
            att_auc_list_2.append(auc)

        for each_item_score, item_len, item_label in zip(item_score_list, test_item_len, test_item_label):
            each_item_score_ = each_item_score[:item_len]
            auc = roc_auc_score(item_label, each_item_score_)
            item_auc_list_2.append(auc)

    mean_att_auc = np.mean(np.array(att_auc_list))
    mean_item_auc = np.mean(np.array(item_auc_list))
    mean_att_auc_2 = np.mean(np.array(att_auc_list_2))
    mean_item_auc_2 = np.mean(np.array(item_auc_list_2))

    print("---------test-----------")
    print("att_auc: {}, item_auc: {}".format(str(mean_att_auc), str(mean_item_auc)))
    print("with item neg: att_auc: {}, item_auc: {}".format(str(mean_att_auc_2), str(mean_item_auc_2)))
    return mean_att_auc, mean_item_auc, mean_att_auc_2, mean_item_auc_2

model_name = "iter60-2020-06-09"
rec.load_model(model_name, False)
rec.init_eval(adj_index)
with torch.no_grad():
    rec_test()