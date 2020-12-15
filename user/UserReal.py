import json
import random

class UserReal():
    def __init__(self, config):
        with open(config.item_info_path, 'r') as f:
            self.item_info = json.load(f)
        for item in self.item_info:
            self.item_info = set(self.item_info[item])
        self.user = None
        self.target_item = None
        self.pos_attribute_set = None
        
    def init_episode(self, user, target_item):
        self.user = user
        self.target_item = target_item
        self.pos_attribute_set = self.item_info[target_item]
        print("Now you are user {}".format(self.user))
        print("Target restaurant info: {}".format(str(self.pos_attribute_set)))
        att_content = input("choose one attribute as initialization:")
        return int(att_content)

    def next_turn(self, request_facet):
        att_content = input("choose attribute you prefer, use comma to seperate:")
        return list(eval(att_content))