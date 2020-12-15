import random
import torch
import json
import numpy as np

class AgentRule():
    def __init__(self, config, convhis):
        self.rec_prob_para = config.rec_prob_para * 1.
        self.attribute_num = config.attribute_num
        self.convhis = convhis

    def init_episode(self):
        pass

    def next_turn(self):
        candidate_list_len = self.convhis.get_candidate_list_len()
        attribute_entropy = self.convhis.get_attribute_entropy()

        rec_prob = self.rec_prob_para / max(candidate_list_len * 1., self.rec_prob_para)
        if random.random() < rec_prob:
            return self.attribute_num # recommend
        else:
            return np.argmax(attribute_entropy).item() # ask attribute index
