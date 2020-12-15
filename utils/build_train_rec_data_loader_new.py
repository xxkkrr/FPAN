import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
import json

random.seed(1640)

def pad_list_of_list(list_of_list, pad_idx=0):
    maxlen = max([len(_) for _ in list_of_list])
    if maxlen == 0:
        maxlen = 1
    padded_list_of_list = np.full((len(list_of_list), maxlen), pad_idx)
    mask_list_of_list = np.full((len(list_of_list), maxlen), False)
    for i in range(len(list_of_list)):
        padded_list_of_list[i][:len(list_of_list[i])] = list_of_list[i]
        mask_list_of_list[i][:len(list_of_list[i])] = True
    return torch.from_numpy(padded_list_of_list), torch.from_numpy(mask_list_of_list)

def item_att_collate_fn(batch):
    user, item, item_pos_att, item_neg_att, item_neg_item, item_neg_train1, item_neg_train2, \
                att_pos_att, att_neg_att, att_neg_item, att_pos_train, att_neg_train = zip(*batch)
    user_list = torch.tensor(user)
    pos_item_list = torch.tensor(item)
    item_pos_att_list, item_pos_att_mask = pad_list_of_list(item_pos_att)
    item_neg_att_list, item_neg_att_mask = pad_list_of_list(item_neg_att)
    item_neg_item_list, item_neg_item_mask = pad_list_of_list(item_neg_item)
    neg_item_list1, neg_item_mask1 = pad_list_of_list(item_neg_train1)
    neg_item_list2, neg_item_mask2 = pad_list_of_list(item_neg_train2)

    pos_att_list = torch.tensor(att_pos_train)
    att_pos_att_list, att_pos_att_mask = pad_list_of_list(att_pos_att)
    att_neg_att_list, att_neg_att_mask = pad_list_of_list(att_neg_att)
    att_neg_item_list, att_neg_item_mask = pad_list_of_list(att_neg_item)
    neg_att_list, neg_att_mask = pad_list_of_list(att_neg_train)    

    return user_list, pos_item_list, \
            item_pos_att_list, item_pos_att_mask, item_neg_att_list, item_neg_att_mask, item_neg_item_list, item_neg_item_mask,\
            neg_item_list1, neg_item_mask1, neg_item_list2, neg_item_mask2, \
            pos_att_list, \
            att_pos_att_list, att_pos_att_mask, att_neg_att_list, att_neg_att_mask, att_neg_item_list, att_neg_item_mask,\
            neg_att_list, neg_att_mask

class ItemAttTrainGenerator(Dataset):
    def __init__(self, train_info_path, user_info, item_info, att_tree_dict, \
                    att_num, item_num, neg_item_num_selected, neg_att_num_selected, \
                    use_gpu=False, add_neg_item_prob=0.5, add_neg_item_num_max=100):
        with open(train_info_path, "r") as f:
            self.train_info = json.load(f)
        self.user_info = user_info
        for key in self.user_info:
            self.user_info[key] = set(self.user_info[key])
        self.item_info = item_info
        for key in self.item_info:
            self.item_info[key] = set(self.item_info[key])
        self.att_tree_dict = att_tree_dict
        for key in self.att_tree_dict:
            self.att_tree_dict[key] = set(self.att_tree_dict[key])
        self.neg_item_num_selected = neg_item_num_selected
        self.neg_att_num_selected = neg_att_num_selected
        self.att_num = att_num
        self.item_num = item_num
        self.len = len(self.train_info)
        self.add_neg_item_prob = add_neg_item_prob
        self.add_neg_item_num_max = add_neg_item_num_max

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        out_dict = {}

        index2 = random.randint(0, 2)

        user = self.train_info[index][index2]['user']
        item = self.train_info[index][index2]['item']
        asked_list = self.train_info[index][index2]['asked_list']
        candidate_list = self.train_info[index][index2]['candidate_list']

        #-------------att----------------
        if len(asked_list) == 1:
            split_index = 1
        else:
            split_index = random.choice([_ for _ in range(1, len(asked_list))])
        att_pos_attribute = set()
        att_neg_attribute = set()
        for parent_att in asked_list[:split_index]:
            att_set = self.att_tree_dict[parent_att]
            pos_att_set = att_set & self.item_info[item]
            neg_att_set = att_set - pos_att_set
            att_pos_attribute = att_pos_attribute.union(pos_att_set)
            att_neg_attribute = att_neg_attribute.union(neg_att_set)

        if len(self.item_info[item] - set(att_pos_attribute)) == 0:
            att_pos_train = random.sample(self.item_info[item], 1)[0]
            att_pos_attribute.remove(att_pos_train)
        else:
            att_pos_train = random.sample(self.item_info[item] - set(att_pos_attribute), 1)[0]

        att_pos_att = []
        att_neg_att = []
        att_neg_item = []
        for _ in att_pos_attribute:
            att_pos_att.append(_)
        for _ in att_neg_attribute:
            att_neg_att.append(_)
        add_neg_item = random.random()
        if add_neg_item < self.add_neg_item_prob and len(candidate_list) >= 3:
            not_target_set = set(candidate_list) - set([item])
            max_neg_num = max(1, min(self.add_neg_item_num_max, len(not_target_set) // 3))
            rand_neg_num = random.randint(1, max_neg_num)
            add_neg_item_list = random.sample(not_target_set, rand_neg_num)
            for _ in add_neg_item_list:
                att_neg_item.append(_)            

        neg_att_set = set(range(self.att_num)) - self.item_info[item] - set(att_neg_attribute)
        if len(neg_att_set) == 0:
            neg_att_set = set(range(self.att_num)) - self.item_info[item]          
        if len(neg_att_set) <= self.neg_att_num_selected:
            att_neg_train = list(neg_att_set)
        else:
            att_neg_train = random.sample(neg_att_set, self.neg_att_num_selected)

        #-----------item------------
        neg_item_set1 = set(range(self.item_num)) - self.user_info[user]
        neg_item_set2 = set(candidate_list) - self.user_info[user]

        if len(neg_item_set1) <= self.neg_item_num_selected:
            item_neg_train1 = list(neg_item_set1)
        else:
            item_neg_train1 = random.sample(neg_item_set1, self.neg_item_num_selected)

        if len(neg_item_set2) <= self.neg_item_num_selected:
            if len(neg_item_set2) == 0:
                item_neg_train2 = random.sample(neg_item_set1, self.neg_item_num_selected)
            else:
                item_neg_train2 = list(neg_item_set2)
        else:
            item_neg_train2 = random.sample(neg_item_set2, self.neg_item_num_selected)

        att_pos_attribute = set()
        att_neg_attribute = set()
        for parent_att in asked_list:
            att_set = self.att_tree_dict[parent_att]
            pos_att_set = att_set & self.item_info[item]
            neg_att_set = att_set - pos_att_set
            att_pos_attribute = att_pos_attribute.union(pos_att_set)
            att_neg_attribute = att_neg_attribute.union(neg_att_set)
        item_pos_att = []
        item_neg_att = []
        item_neg_item = []
        for _ in att_pos_attribute:
            item_pos_att.append(_)
        for _ in att_neg_attribute:
            item_neg_att.append(_)
        add_neg_item = random.random()
        if add_neg_item < self.add_neg_item_prob:
            not_target_set = set(candidate_list) - set([item]) - set(item_neg_train2)
            if len(not_target_set) > 0:
                max_neg_num = max(1, min(self.add_neg_item_num_max, len(not_target_set) // 2))
                rand_neg_num = random.randint(1, max_neg_num)
                add_neg_item_list = random.sample(not_target_set, rand_neg_num)
                for _ in add_neg_item_list:
                    item_neg_item.append(_)   

        return user, item, item_pos_att, item_neg_att, item_neg_item, item_neg_train1, item_neg_train2, \
                att_pos_att, att_neg_att, att_neg_item, att_pos_train, att_neg_train


def build_item_att_loader(train_info_path, user_info, item_info, att_tree_dict, \
                            att_num, item_num, neg_item_num_selected, neg_att_num_selected, \
                            use_gpu=False, add_neg_item_prob=0.5, add_neg_item_num_max=100, \
                            batch_size=1, shuffle=True, num_threads=4):
    train_generator = ItemAttTrainGenerator(train_info_path, user_info, item_info, att_tree_dict, \
                                            att_num, item_num, neg_item_num_selected, neg_att_num_selected, \
                                            use_gpu=use_gpu, add_neg_item_prob=add_neg_item_prob, \
                                            add_neg_item_num_max=add_neg_item_num_max)
    return DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=item_att_collate_fn
    )
