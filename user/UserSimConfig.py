import os

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class UserSimConfig():
    def __init__(self):
        self.item_info_path = root_path + "/data/item_info.json"
        self.attribute_tree_path = root_path + "/data/attribute_tree_dict.json"