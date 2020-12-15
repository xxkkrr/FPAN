import json
import random

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

class UserSim():
    def __init__(self, config):
        with open(config.item_info_path, 'r') as f:
            self.item_info = json.load(f)
        for item in self.item_info:
            self.item_info[item] = set(self.item_info[item])
        self.item_info = trans_index(self.item_info)
        
        with open(config.attribute_tree_path, 'r') as f:
            self.attribute_tree = json.load(f)
        self.attribute_tree = trans_index(self.attribute_tree)
        for key in self.attribute_tree:
            self.attribute_tree[key] = set(self.attribute_tree[key])

        self.user = None
        self.target_item = None
        self.pos_attribute_set = None

    def init_episode(self, user, target_item, given_init_parent_attribute):
        self.user = user
        self.target_item = target_item
        self.pos_attribute_set = self.item_info[target_item]

        if given_init_parent_attribute is None:
            parent_att_list = []
            for parent_att in self.attribute_tree:
                if len(self.pos_attribute_set & self.attribute_tree[parent_att]) > 0:
                    parent_att_list.append(parent_att)
            init_parent_att = random.choice(parent_att_list)
        else:
            init_parent_att = given_init_parent_attribute
        att_in_parent = self.attribute_tree[init_parent_att]
        init_pos_attribute_set = self.pos_attribute_set & att_in_parent
        init_neg_attribute_set = att_in_parent - init_pos_attribute_set

        return init_pos_attribute_set, init_neg_attribute_set, init_parent_att

    def next_turn(self, asked_attribute_list):
        pos_attribute_list = set(asked_attribute_list) & self.pos_attribute_set
        return list(pos_attribute_list)
