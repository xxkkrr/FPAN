import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, n_heads):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
        elif self.conv_name == 'sage':
            self.base_conv = SAGEConv(in_hid, out_hid, concat=True)
        else:
            print("no predefined conv layer {} !".format(conv_name))
    def forward(self, input_x, edge_index):
        if self.conv_name == 'gcn':
            return self.base_conv(input_x, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(input_x, edge_index)
        elif self.conv_name == 'sage':
        	return self.base_conv(input_x, edge_index)