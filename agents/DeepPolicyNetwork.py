import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayersModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpu = config.use_gpu
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(config.dp)

        if self.gpu:
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
            self.softmax = self.softmax.cuda()
            self.drop = self.drop.cuda()
        
    def forward(self, x, return_softmax):
        h1 = self.fc1(x)
        h1 = F.relu(h1)
        h1 = self.drop(h1)
        h2 = self.fc2(h1)    

        if return_softmax:
            out = self.softmax(h2)
        else:
            out = h2
        return out