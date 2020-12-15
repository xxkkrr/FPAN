import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentRuleConfig():
    def __init__(self):
        self.rec_prob_para = 10 
        self.attribute_num = 29