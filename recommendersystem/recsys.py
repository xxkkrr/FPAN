import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recommendersystem.myrecmodel import MyRec

def pad_list_of_list(list_of_list, pad_idx=0):
    maxlen = max([len(_) for _ in list_of_list])
    padded_list_of_list = np.full((len(list_of_list), maxlen), pad_idx)
    mask_list_of_list = np.full((len(list_of_list), maxlen), False)
    for i in range(len(list_of_list)):
        padded_list_of_list[i][:len(list_of_list[i])] = list_of_list[i]
        mask_list_of_list[i][:len(list_of_list[i])] = True
    return torch.from_numpy(padded_list_of_list), torch.from_numpy(mask_list_of_list)

class recsys():
    def __init__(self, config, convhis=None, use_gpu=None):
        if use_gpu is not None:
            config.use_gpu = use_gpu

        self.model_info_str = 'dim-{}_layer-{}_conv-{}_fa-{}_la-{}'.\
                            format(str(config.hidden_dim), str(config.nlayer), str(config.conv_name), \
                                    str(config.feedback_aggregate), str(config.layer_aggregate))

        self.top_taxo = config.top_taxo
        self.att_score_norm = config.att_score_norm
        self.item_num = config.item_num
        self.attribute_num = config.attribute_num
        self.parent_attribute_num = config.parent_attribute_num
        self.use_gpu = config.use_gpu

        self.item_offset = torch.tensor(0)
        self.att_offset = torch.tensor(self.item_num)

        self.max_rec_item_num = config.max_rec_item_num
        self.item_state_num = config.item_state_num
        self.convhis = convhis

        with open(config.attribute_tree_path, 'r') as f:
            self.attribute_tree = json.load(f)
        new_attribute_tree = {}
        for parent in self.attribute_tree:
            new_attribute_tree[int(parent)] = set(self.attribute_tree[parent])
        self.attribute_tree = new_attribute_tree

        self.attribute_parent_mat = np.zeros([self.parent_attribute_num, self.attribute_num])
        for attribute_parent, attribute_list in self.attribute_tree.items():
            for attribute in attribute_list:
                self.attribute_parent_mat[attribute_parent, attribute] = 1.  
        
        self.logsigmoid = nn.LogSigmoid()

        self.rec = MyRec(config)
        self.rec_model_path = config.rec_model_path

        if self.use_gpu:
            self.item_offset = self.item_offset.cuda()
            self.att_offset = self.att_offset.cuda()

    def save_model(self, extra_name=None):
        name = "rec_model" + "_" + self.model_info_str
        if extra_name is not None:
            name = name + '_' + extra_name
        torch.save(self.rec.state_dict(), "/".join([self.rec_model_path, name]))

    def load_model(self, extra_name=None, transfer_to_cpu=False):
        name = "rec_model" + "_" + self.model_info_str
        if extra_name is not None:
            name = name + '_' + extra_name
        if transfer_to_cpu:
            self.rec.load_state_dict(torch.load("/".join([self.rec_model_path, name]), map_location='cpu'))
        else:
            self.rec.load_state_dict(torch.load("/".join([self.rec_model_path, name])))

    def init_train(self):
        self.rec.train()

    def init_eval(self, edge_index):
        self.rec.eval()
        self.rec.graph_prop(edge_index)

    def get_item_preference(self, user, pos_attribute, neg_attribute, neg_item, candidate_list=None):
        pos_att_list = []
        neg_att_list = []
        neg_item_list = []

        for _ in pos_attribute:
            pos_att_list.append(_)
        pos_att_mask = [True] * len(pos_att_list)

        for _ in neg_attribute:
            neg_att_list.append(_)
        if len(neg_att_list) == 0:
            neg_att_list = [0]
            neg_att_mask = [False]
        else:
            neg_att_mask = [True] * len(neg_att_list)

        for _ in neg_item:
            neg_item_list.append(_)
        if len(neg_item_list) == 0:
            neg_item_list = [0]
            neg_item_mask = [False]
        else:
            neg_item_mask = [True] * len(neg_item_list)

        user = torch.tensor(user)
        pos_att_list = torch.tensor(pos_att_list)
        pos_att_mask = torch.tensor(pos_att_mask)
        neg_att_list = torch.tensor(neg_att_list)
        neg_att_mask = torch.tensor(neg_att_mask)
        neg_item_list = torch.tensor(neg_item_list)
        neg_item_mask = torch.tensor(neg_item_mask)
        if candidate_list is not None:
            candidate_list = torch.tensor(candidate_list)

        if self.use_gpu:
            user = user.cuda()
            pos_att_list = pos_att_list.cuda()
            pos_att_mask = pos_att_mask.cuda()
            neg_att_list = neg_att_list.cuda()
            neg_att_mask = neg_att_mask.cuda()
            neg_item_list = neg_item_list.cuda()
            neg_item_mask = neg_item_mask.cuda()   
            if candidate_list is not None:
                candidate_list = candidate_list.cuda()

        self.rec.get_current_user_embedding(user, pos_att_list, pos_att_mask, \
                                            neg_att_list, neg_att_mask, neg_item_list, neg_item_mask)
        item_score = self.rec.get_current_item_score(user, candidate_list)
        item_score = item_score.detach()
        if self.use_gpu:
            item_score = item_score.cpu()
        return item_score

    def get_att_preference(self, user, pos_attribute, neg_attribute, neg_item, candidate_list=None, return_parent=True):
        pos_att_list = []
        neg_att_list = []
        neg_item_list = []

        for _ in pos_attribute:
            pos_att_list.append(_)
        pos_att_mask = [True] * len(pos_att_list)

        for _ in neg_attribute:
            neg_att_list.append(_)
        if len(neg_att_list) == 0:
            neg_att_list = [0]
            neg_att_mask = [False]
        else:
            neg_att_mask = [True] * len(neg_att_list)

        for _ in neg_item:
            neg_item_list.append(_)
        if len(neg_item_list) == 0:
            neg_item_list = [0]
            neg_item_mask = [False]
        else:
            neg_item_mask = [True] * len(neg_item_list)

        user = torch.tensor(user)
        pos_att_list = torch.tensor(pos_att_list)
        pos_att_mask = torch.tensor(pos_att_mask)
        neg_att_list = torch.tensor(neg_att_list)
        neg_att_mask = torch.tensor(neg_att_mask)
        neg_item_list = torch.tensor(neg_item_list)
        neg_item_mask = torch.tensor(neg_item_mask)
        if candidate_list is not None:
            candidate_list = torch.tensor(candidate_list)

        if self.use_gpu:
            user = user.cuda()
            pos_att_list = pos_att_list.cuda()
            pos_att_mask = pos_att_mask.cuda()
            neg_att_list = neg_att_list.cuda()
            neg_att_mask = neg_att_mask.cuda()
            neg_item_list = neg_item_list.cuda()
            neg_item_mask = neg_item_mask.cuda()   
            if candidate_list is not None:
                candidate_list = candidate_list.cuda()

        self.rec.get_current_user_embedding(user, pos_att_list, pos_att_mask, \
                                            neg_att_list, neg_att_mask, neg_item_list, neg_item_mask)
        att_score = self.rec.get_current_att_score(user, candidate_list)
        if self.use_gpu:
            att_score = att_score.cpu()
        
        att_score = att_score.detach().numpy()
        if return_parent: 
            parent_att_score_list = []
            for parent_att in range(self.parent_attribute_num):
                parent_att_score = att_score[list(self.attribute_tree[parent_att])]
                parent_att_score = - np.sort(- parent_att_score)
                parent_att_score_sum = sum(parent_att_score[:self.top_taxo]) / self.att_score_norm
                parent_att_score_list.append(parent_att_score_sum)
            return parent_att_score_list
        else:
            att_score = att_score.tolist()
            return att_score

    def item_one_step_train(self, user_list, edge_index, known_pos_att_list, known_pos_att_mask, \
                            known_neg_att_list, known_neg_att_mask, known_neg_item_list, known_neg_item_mask, \
                        pos_item_list, neg_item_list1, neg_item_mask1, neg_item_list2, neg_item_mask2):

        self.rec.graph_prop(edge_index)
        self.rec.get_current_user_embedding(user_list, known_pos_att_list, known_pos_att_mask, \
                                            known_neg_att_list, known_neg_att_mask, known_neg_item_list, known_neg_item_mask)

        pos_item_score = self.rec.get_current_item_score(user_list, pos_item_list.unsqueeze(-1)).squeeze(-1)
        neg_item_score1 = self.rec.get_current_item_score(user_list, neg_item_list1)
        neg_item_score1 = neg_item_score1.masked_fill(neg_item_mask1==False, -1e9)
        neg_item_score1, _ = neg_item_score1.max(dim=-1)
        neg_item_score2 = self.rec.get_current_item_score(user_list, neg_item_list2)
        neg_item_score2 = neg_item_score2.masked_fill(neg_item_mask2==False, -1e9)
        neg_item_score2, _ = neg_item_score2.max(dim=-1)
        item_loss1 = - self.logsigmoid(pos_item_score - neg_item_score1)
        item_loss2 = - self.logsigmoid(pos_item_score - neg_item_score2)
        item_loss1 = item_loss1.mean()
        item_loss2 = item_loss2.mean()
        return item_loss1, item_loss2

    def att_one_step_train(self, user_list, edge_index, known_pos_att_list, known_pos_att_mask, \
                            known_neg_att_list, known_neg_att_mask, known_neg_item_list, known_neg_item_mask, \
                            pos_att_list, neg_att_list, neg_att_mask):
        self.rec.graph_prop(edge_index)
        self.rec.get_current_user_embedding(user_list, known_pos_att_list, known_pos_att_mask, \
                                            known_neg_att_list, known_neg_att_mask, known_neg_item_list, known_neg_item_mask)

        pos_att_socre = self.rec.get_current_att_score(user_list, pos_att_list.unsqueeze(-1)).squeeze(-1)
        neg_att_score = self.rec.get_current_att_score(user_list, neg_att_list)
        neg_att_score = neg_att_score.masked_fill(neg_att_mask==False, -1e9)
        neg_att_score, _ = neg_att_score.max(dim=-1)
        att_loss = - (self.logsigmoid(pos_att_socre - neg_att_score)).mean()
        return att_loss

    def get_recommend_item_list(self, candidate_list=None):
        user = self.convhis.get_user()
        pos_attribute = self.convhis.get_pos_attribute()
        neg_attribute = self.convhis.get_neg_attribute()
        neg_item = self.convhis.get_conv_neg_item_list()

        item_score_list = self.get_item_preference(user, pos_attribute, neg_attribute, neg_item, candidate_list=candidate_list)
        values, indices = item_score_list.sort(descending=True)
        if candidate_list == None:
            return indices.tolist()[:self.max_rec_item_num]
        else:
            indices = indices.tolist()[:self.max_rec_item_num]
            item_list = []
            for i in indices:
                item_list.append(candidate_list[i])
            return item_list

    def get_item_prefer_state(self):
        user = self.convhis.get_user()
        pos_attribute = self.convhis.get_pos_attribute()
        neg_attribute = self.convhis.get_neg_attribute()
        neg_item = self.convhis.get_conv_neg_item_list()
        candidate_list = self.convhis.get_candidate_list()
        item_score_list = self.get_item_preference(user, pos_attribute, neg_attribute, neg_item, candidate_list=candidate_list)
        values, indices = item_score_list.sort(descending=True)
        score_list = values.tolist()[:self.item_state_num]
        if len(score_list) < self.item_state_num:
            score_list = score_list + [0.] * (self.item_state_num - len(score_list))
        return score_list

    def get_attribute_preference(self, candidate_list=None, return_parent=True):
        user = self.convhis.get_user()
        pos_attribute = self.convhis.get_pos_attribute()
        neg_attribute = self.convhis.get_neg_attribute()
        neg_item = self.convhis.get_conv_neg_item_list()

        att_score_list = self.get_att_preference(user, pos_attribute, neg_attribute, neg_item, \
                                                candidate_list=candidate_list, return_parent=return_parent)

        asked_list = self.convhis.get_asked_list()
        if return_parent:
            for att in asked_list:
                att_score_list[att] = -1.
        else:
            for parent_att in asked_list:
                for att in self.attribute_tree[parent_att]:
                    att_score_list[att] = -1.

        return att_score_list