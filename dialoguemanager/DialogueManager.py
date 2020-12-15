import json
import torch

def trans_index(origin_dict):
    new_dict = {}
    for key in origin_dict:
        new_dict[int(key)] = origin_dict[key]
    return new_dict

class DialogueManager:
    def __init__(self, config, rec, agent, user, convhis):      
        with open(config.index2attribute_path, 'r') as f:
            self.index2attribute = json.load(f)
        self.index2attribute = trans_index(self.index2attribute)

        with open(config.index2item_path, 'r') as f:
            self.index2item = json.load(f)      
        self.index2item = trans_index(self.index2item)

        with open(config.attribute_tree_path, 'r') as f:
            self.attribute_tree = json.load(f)
        self.attribute_tree = trans_index(self.attribute_tree)

        self.rec_action_index = config.rec_action_index
        self.rec_success_reward = config.rec_success_reward
        self.pos_attribute_reward = config.pos_attribute_reward
        self.user_quit_reward = config.user_quit_reward
        self.every_turn_reward = config.every_turn_reward
        self.turn_limit = config.turn_limit

        self.rec = rec
        self.agent = agent
        self.user = user
        self.convhis = convhis

        self.target_user = None
        self.target_item = None
        self.silence = True
        self.turn_num = None
        self.current_agent_action = None

    def output_user_attribute(self, attribute_list):
        if len(attribute_list) == 0:
            user_utt = "not like"
        else:
            output_attribute_list = list(map(lambda x: self.index2attribute[x], attribute_list))
            user_utt = "like " + ",".join(output_attribute_list)
        print("turn {} user: {}".format(str(self.turn_num), user_utt))

    def output_user_item(self, like):
        if like:
            user_utt = "like"
        else:
            user_utt = "not like"
        print("turn {} user: {}".format(str(self.turn_num), user_utt))

    def output_agent_attribute(self, action_index):
        attribute_list = self.attribute_tree[action_index]
        attribute_list = map(lambda x: self.index2attribute[x], attribute_list)
        agent_utt = "choose attribute you like: " + ','.join(attribute_list)
        print("turn {} agent: {}".format(str(self.turn_num), agent_utt))

    def output_agent_item(self, rec_item_list):
        item_list = map(lambda x: self.index2item[x], rec_item_list)
        agent_utt = "recommend items: " + ','.join(item_list)
        print("turn {} agent: {}".format(str(self.turn_num), agent_utt))

    def initialize_dialogue(self, target_user, target_item, silence, given_init_parent_attribute=None):
        self.target_user = target_user
        self.target_item = target_item
        self.silence = silence
        self.turn_num = 1
        self.current_agent_action = None

        self.agent.init_episode()
        init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute \
            = self.user.init_episode(target_user, target_item, given_init_parent_attribute)
        self.convhis.init_conv(target_user, target_item, init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute)

        if not self.silence:
            self.output_user_attribute(list(init_pos_attribute_set))
        return list(init_pos_attribute_set)

    def agent_turn(self):
        action_index = self.agent.next_turn()
        self.current_agent_action = action_index
        if action_index == self.rec_action_index:
            candidate_list = self.convhis.get_candidate_list()
            if self.rec is None:
                rec_item_list = candidate_list[:1]
            else:
                rec_item_list = self.rec.get_recommend_item_list(candidate_list)
            if not self.silence:
                self.output_agent_item(rec_item_list)
            return None, rec_item_list
        else:
            ask_attribute_list = self.attribute_tree[action_index]
            if not self.silence:
                self.output_agent_attribute(action_index)
            return action_index, None

    def user_turn(self, ask_attribute_list, ask_item_list):
        self.turn_num += 1
        if ask_attribute_list != None:
            attribute_list = self.user.next_turn(ask_attribute_list)
            if not self.silence:
                self.output_user_attribute(attribute_list)
            return attribute_list
        if ask_item_list != None:
            item_list = []
            for item in ask_item_list:
                if item == self.target_item:
                    item_list.append(item)
                    break
            if not self.silence:
                self.output_user_item(len(item_list)>0)
            return item_list

    def next_turn(self):
        if self.turn_num == self.turn_limit + 1:
            return True, self.user_quit_reward, None
        action_index, ask_item_list = self.agent_turn()
        if action_index != None:
            ask_attribute_list = self.attribute_tree[action_index]
            attribute_list = self.user_turn(ask_attribute_list, ask_item_list)
            pos_attribute_set = set(attribute_list)
            neg_attribute_set = set(ask_attribute_list) - pos_attribute_set
            self.convhis.add_new_attribute(pos_attribute_set, action_index)
            self.convhis.update_conv_his(len(pos_attribute_set)>0, action_index)
            if len(attribute_list) > 0:
                return False, self.every_turn_reward + self.pos_attribute_reward, None
            else:
                return False, self.every_turn_reward, None
        if ask_item_list != None:
            item_list = self.user_turn(action_index, ask_item_list)
            if len(item_list) > 0:
                return True, self.rec_success_reward, ask_item_list
            else:
                self.convhis.add_conv_neg_item_list(ask_item_list)
                return False, self.every_turn_reward, ask_item_list

    def get_current_agent_action(self):
        return self.current_agent_action

    def initialize_episode(self, target_user, target_item, given_init_parent_attribute=None, silence=True):
        self.target_user = target_user
        self.target_item = target_item
        self.turn_num = 1
        self.current_agent_action = None
        self.silence = silence

        init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute \
            = self.user.init_episode(target_user, target_item, given_init_parent_attribute)
        self.convhis.init_conv(target_user, target_item, init_pos_attribute_set, init_neg_attribute_set, init_parent_attribute)

        state_entropy = self.convhis.get_attribute_entropy().copy()
        state_att_prefer = self.rec.get_attribute_preference().copy()
        state_convhis = self.convhis.get_convhis_vector().copy()
        state_len = self.convhis.get_length_vector().copy()
        dialogue_state = state_entropy + state_att_prefer + state_convhis + state_len
        return torch.tensor(dialogue_state)

    def step(self, action_index):
        def transfer_agent_action(action_index):
            if action_index == self.rec_action_index:
                candidate_list = self.convhis.get_candidate_list()
                if self.rec is None:
                    rec_item_list = candidate_list[:1]
                else:
                    rec_item_list = self.rec.get_recommend_item_list(candidate_list)
                if not self.silence:
                    self.output_agent_item(rec_item_list)
                return None, rec_item_list
            else:
                ask_attribute_list = self.attribute_tree[action_index]
                if not self.silence:
                    self.output_agent_attribute(action_index)
                return action_index, None        

        action_index, ask_item_list = transfer_agent_action(action_index)
        IsOver, next_state, reward = None, None, None

        if action_index != None:
            ask_attribute_list = self.attribute_tree[action_index]
            attribute_list = self.user_turn(ask_attribute_list, ask_item_list)
            pos_attribute_set = set(attribute_list)
            neg_attribute_set = set(ask_attribute_list) - pos_attribute_set
            self.convhis.add_new_attribute(pos_attribute_set, action_index)
            self.convhis.update_conv_his(len(pos_attribute_set)>0, action_index)
            if len(attribute_list) > 0:
                IsOver = False
                reward = self.every_turn_reward + self.pos_attribute_reward
            else:
                IsOver = False
                reward = self.every_turn_reward
        if ask_item_list != None:
            item_list = self.user_turn(action_index, ask_item_list)
            if len(item_list) > 0:
                IsOver = True
                reward = self.rec_success_reward
            else:
                self.convhis.add_conv_neg_item_list(ask_item_list)
                IsOver = False
                reward = self.every_turn_reward                

        if self.turn_num == self.turn_limit + 1:
            if not IsOver:
                IsOver = True
                reward = self.user_quit_reward

        state_entropy = self.convhis.get_attribute_entropy().copy()
        state_att_prefer = self.rec.get_attribute_preference().copy()
        state_convhis = self.convhis.get_convhis_vector().copy()
        state_len = self.convhis.get_length_vector().copy()
        dialogue_state = state_entropy + state_att_prefer + state_convhis + state_len
        dialogue_state = torch.tensor(dialogue_state)
        return IsOver, dialogue_state, reward