import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class AgentEARConfig():
    def __init__(self):
    	self.use_gpu = False
    	self.input_dim = 33 + 33 + 15 + 8
    	self.hidden_dim = 64
    	self.output_dim = 33 + 1
    	self.dp = 0.2
    	self.DPN_model_path = root_path + "/agents/agent_ear/"
    	self.DPN_model_name = "TwoLayer"
    	self.PG_discount_rate = 0.7