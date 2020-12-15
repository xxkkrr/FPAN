import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class recsysconfig():
    def __init__(self):
        self.use_gpu = False
        self.rec_model_path = root_path + "/recommendersystem/recmodel"
        self.attribute_tree_path = root_path + "/data/attribute_tree_dict.json"
        self.hidden_dim = 64
        self.user_num = 27675
        self.item_num = 70311
        self.attribute_num = 590
        self.parent_attribute_num = 29
        self.nlayer = 2
        self.conv_name = 'sage' # 'sage', 'gcn', 'gat'
        self.n_heads = 1
        self.drop = 0.1
        self.max_rec_item_num = 10
        self.item_state_num = 30
        self.top_taxo = 3
        self.feedback_aggregate = "gating" # 'mean', 'gating'
        self.layer_aggregate = "mean" # 'mean', 'last_layer', 'concat'
        self.att_score_norm = 5.