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

def test_item_att_collate_fn(batch):
    user, item, pos_att, neg_att, neg_item, \
    test_attribute, test_attribute_len, test_attribute_label, \
    test_item, test_item_len, test_item_label = zip(*batch)

    user_list = torch.tensor(user)
    item_list = torch.tensor(item)
    pos_att_list, pos_att_mask = pad_list_of_list(pos_att)
    neg_att_list, neg_att_mask = pad_list_of_list(neg_att)
    neg_item_list, neg_item_mask_2 = pad_list_of_list(neg_item)
    neg_item_mask_1 = torch.full(neg_item_mask_2.size(), False)
    test_attribute_list, test_attribute_mask = pad_list_of_list(test_attribute)
    test_item_list, test_item_mask = pad_list_of_list(test_item)

    return user_list, item_list, pos_att_list, pos_att_mask, neg_att_list, neg_att_mask, \
            neg_item_list, neg_item_mask_1, neg_item_mask_2, \
            test_attribute_list, test_attribute_mask, test_attribute_len, test_attribute_label, \
            test_item_list, test_item_mask, test_item_len, test_item_label

class ItemAttTestGenerator(Dataset):
    def __init__(self, test_info, user_info, item_info, att_tree_dict, \
                    att_num, item_num, use_gpu=False):
        self.test_info = test_info
        self.user_info = user_info
        for key in self.user_info:
            self.user_info[key] = set(self.user_info[key])
        self.item_info = item_info
        for key in self.item_info:
            self.item_info[key] = set(self.item_info[key])
        self.att_tree_dict = att_tree_dict
        for key in self.att_tree_dict:
            self.att_tree_dict[key] = set(self.att_tree_dict[key])
        self.att_num = att_num
        self.item_num = item_num
        self.len = len(self.test_info)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        out_dict = {}
        user = self.test_info[index]['user']
        item = self.test_info[index]['item']
        asked_list = self.test_info[index]['asked_list']
        candidate_list = self.test_info[index]['candidate_list']
        neg_items = self.test_info[index]['neg_items']

        att_pos_attribute = set()
        att_neg_attribute = set()
        for parent_att in asked_list:
            att_set = self.att_tree_dict[parent_att]
            pos_att_set = att_set & self.item_info[item]
            neg_att_set = att_set - pos_att_set
            att_pos_attribute = att_pos_attribute.union(pos_att_set)
            att_neg_attribute = att_neg_attribute.union(neg_att_set)
        pos_att = []
        neg_att = []
        neg_item = []
        for _ in att_pos_attribute:
            pos_att.append(_)
        for _ in att_neg_attribute:
            neg_att.append(_)
        for _ in neg_items:
            neg_item.append(_)

        #-------------att---------------
        full_attribute = self.item_info[item]
        test_pos_attribute = list(full_attribute - att_pos_attribute)
        test_neg_attribute = list(set(range(self.att_num)) - full_attribute)
        test_attribute = test_pos_attribute + test_neg_attribute
        test_attribute_len = len(test_attribute)
        test_attribute_label = [1] * len(test_pos_attribute) + [0] * len(test_neg_attribute)

        #-------------item--------------
        candidate_list.remove(item)
        test_item = [item] + candidate_list[:1000]
        test_item_len = len(test_item)
        test_item_label = [1] + [0] * len(candidate_list[:1000])

        return user, item, pos_att, neg_att, neg_item, \
                test_attribute, test_attribute_len, test_attribute_label, \
                test_item, test_item_len, test_item_label

def build_test_item_att_loader(test_info, user_info, item_info, att_tree_dict, \
                            att_num, item_num, use_gpu=False, \
                            batch_size=1, shuffle=True, num_threads=4):
    test_generator = ItemAttTestGenerator(test_info, user_info, item_info, att_tree_dict, \
                                            att_num, item_num, use_gpu=use_gpu)
    return DataLoader(
        test_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=test_item_att_collate_fn
    )
