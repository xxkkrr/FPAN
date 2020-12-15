import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from recommendersystem.conv import GeneralConv

class MyRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feedback_aggregate = config.feedback_aggregate
        self.layer_aggregate = config.layer_aggregate
        self.gpu = config.use_gpu
        self.hidden_dim = config.hidden_dim
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.n_layers = config.nlayer
        self.attribute_num = config.attribute_num
        self.user_embed = nn.Embedding(self.user_num, self.hidden_dim)
        self.item_embed = nn.Embedding(self.item_num, self.hidden_dim)
        self.attribute_embed = nn.Embedding(self.attribute_num, self.hidden_dim)
        self.init_para()

        self.user_index = torch.tensor([_ for _ in range(self.user_num)])
        self.item_index = torch.tensor([_ for _ in range(self.item_num)])
        self.attribute_index = torch.tensor([_ for _ in range(self.attribute_num)])     
        self.user_graph_index = torch.tensor([_ for _ in range(self.user_num)])   
        self.item_graph_index = torch.tensor([_ for _ in range(self.user_num, self.user_num+self.item_num)])
        self.attribute_graph_index = torch.tensor([_ for _ in range(self.user_num+self.item_num, self.user_num+self.item_num+self.attribute_num)])

        self.gcs = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        self.nlayer = config.nlayer
        self.conv_name = config.conv_name
        self.n_heads = config.n_heads
        for l in range(self.n_layers):
            self.gcs.append(GeneralConv(self.conv_name, self.hidden_dim, self.hidden_dim, self.n_heads))

        self.graph_rep = None 
        self.eps = torch.tensor(1e-9)
        self.drop = nn.Dropout(config.drop)

        if self.layer_aggregate == 'last_layer' or self.layer_aggregate == 'mean':
            self.graph_dim = self.hidden_dim
        elif self.layer_aggregate == 'concat':
            self.graph_dim = (self.n_layers + 1) * self.hidden_dim
        else:
            print("not support layer_aggregate type : {} !!!".format(self.layer_aggregate))

        if self.feedback_aggregate == 'mean':
            pass
        elif self.feedback_aggregate == 'gating':
            self.gu_linear = nn.Linear(3 * self.graph_dim, self.graph_dim)
            self.gni_linear = nn.Linear(3 * self.graph_dim, self.graph_dim)                             
        else:
            print("not support feedback_aggregate type : {} !!!".format(self.feedback_aggregate))

        if self.gpu:
            self.user_embed = self.user_embed.cuda()
            self.item_embed = self.item_embed.cuda()
            self.attribute_embed = self.attribute_embed.cuda()
            self.gcs = self.gcs.cuda()
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()
            self.eps = self.eps.cuda()
            self.drop = self.drop.cuda()
            if self.feedback_aggregate == 'gating':
                self.gu_linear = self.gu_linear.cuda()
                self.gni_linear = self.gni_linear.cuda()               

            self.user_index = self.user_index.cuda()
            self.item_index = self.item_index.cuda()
            self.attribute_index = self.attribute_index.cuda()
            self.user_graph_index = self.user_graph_index.cuda()
            self.item_graph_index = self.item_graph_index.cuda()
            self.attribute_graph_index = self.attribute_graph_index.cuda()

    def init_para(self):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.user_num + self.item_num + self.attribute_num, self.hidden_dim)
        )
        nn.init.xavier_uniform_(all_embed)
        self.user_embed.weight.data = all_embed[:self.user_num].data
        self.item_embed.weight.data = all_embed[self.user_num:self.user_num+self.item_num].data
        self.attribute_embed.weight.data = all_embed[self.user_num+self.item_num:self.user_num+self.item_num+self.attribute_num].data

    def graph_prop(self, edge_index):
        node_input_user = self.user_embed(self.user_index)
        node_input_item = self.item_embed(self.item_index)
        node_input_att = self.attribute_embed(self.attribute_index)
        node_input = torch.cat([node_input_user, node_input_item, node_input_att], dim=0)
        self.graph_layer_rep = [node_input]
        for gc in self.gcs:
            x = gc(self.graph_layer_rep[-1], edge_index)
            x = F.leaky_relu(x)
            x = self.drop(x)
            self.graph_layer_rep.append(x)

        if self.layer_aggregate == 'last_layer':
            self.graph_rep = self.graph_layer_rep[-1]
        if self.layer_aggregate == 'mean':
            graph_layer_rep_tensor = torch.stack(self.graph_layer_rep, dim=1)
            self.graph_rep = torch.mean(graph_layer_rep_tensor, dim=1)
        if self.layer_aggregate == 'concat':
            self.graph_rep = torch.cat(self.graph_layer_rep, dim=1)

        return self.graph_rep

    def get_current_user_embedding(self, user_list, pos_att_list, pos_att_mask, \
                                    neg_att_list, neg_att_mask, neg_item_list, neg_item_mask):

        e_u = self.graph_rep[self.user_graph_index[user_list]]

        pos_att_rep = self.graph_rep[self.attribute_graph_index[pos_att_list]]
        pos_att_rep_mask = pos_att_mask.unsqueeze(-1).expand(pos_att_rep.size())
        pos_att_rep = pos_att_rep.masked_fill(pos_att_rep_mask==False, 0.)
        e_pa = torch.sum(pos_att_rep,dim=-2) / (torch.sum(pos_att_mask,dim=-1,keepdim=True).type(pos_att_rep.type()) + self.eps)

        neg_att_rep = self.graph_rep[self.attribute_graph_index[neg_att_list]]
        neg_att_rep_mask = neg_att_mask.unsqueeze(-1).expand(neg_att_rep.size())
        neg_att_rep = neg_att_rep.masked_fill(neg_att_rep_mask==False, 0.)
        e_na = torch.sum(neg_att_rep,dim=-2) / (torch.sum(neg_att_mask,dim=-1,keepdim=True).type(neg_att_rep.type()) + self.eps)

        neg_item_rep = self.graph_rep[self.item_graph_index[neg_item_list]]
        neg_item_rep_mask = neg_item_mask.unsqueeze(-1).expand(neg_item_rep.size())
        neg_item_rep = neg_item_rep.masked_fill(neg_item_rep_mask==False, 0.)
        e_ni = torch.sum(neg_item_rep,dim=-2) / (torch.sum(neg_item_mask,dim=-1,keepdim=True).type(neg_item_rep.type()) + self.eps)

        if self.feedback_aggregate == 'mean':
            self.current_user = e_u + e_pa - e_na - e_ni         

        if self.feedback_aggregate == 'gating':
            neg_item_rep = self.graph_rep[self.item_graph_index[neg_item_list]]
            neg_item_rep_mask = neg_item_mask.unsqueeze(-1).expand(neg_item_rep.size())
            neg_item_rep = neg_item_rep.masked_fill(neg_item_rep_mask==False, 0.)
            e_pa_ex = e_pa.unsqueeze(-2).expand(neg_item_rep.size())

            input_tensor2 = torch.cat([e_pa_ex, neg_item_rep, e_pa_ex * neg_item_rep], dim=-1)
            input_tensor2 = self.drop(input_tensor2)
            g_ni = torch.sigmoid(self.gni_linear(input_tensor2))
            neg_item_rep = neg_item_rep * g_ni
            e_ni = torch.sum(neg_item_rep,dim=-2) / (torch.sum(neg_item_mask,dim=-1,keepdim=True).type(neg_item_rep.type()) + self.eps)

            input_tensor3 = torch.cat([e_u, e_na, e_u * e_na], dim=-1)
            input_tensor3 = self.drop(input_tensor3)
            g_u = torch.sigmoid(self.gu_linear(input_tensor3))
            e_u = e_u * g_u
            u = e_u + e_pa - e_na - e_ni
            self.current_user = u


    def get_current_att_score(self, user_list, att_list=None):
        user_feature = self.current_user
        if att_list is None:
            att_feature = self.graph_rep[self.attribute_graph_index]
            att_score = torch.matmul(user_feature, att_feature.T)
        else:
            att_feature = self.graph_rep[self.attribute_graph_index[att_list]]
            att_score = torch.sum(user_feature.unsqueeze(-2)*att_feature, dim=-1) 
        return att_score

    def get_current_item_score(self, user_list, item_list=None):
        user_feature = self.current_user
        if item_list is None:
            item_feature = self.graph_rep[self.item_graph_index]
            item_score = torch.matmul(user_feature, item_feature.T)
        else:
            item_feature = self.graph_rep[self.item_graph_index[item_list]]
            item_score = torch.sum(user_feature.unsqueeze(-2)*item_feature, dim=-1)
        return item_score