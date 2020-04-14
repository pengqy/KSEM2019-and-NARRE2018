
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModule import BasicModule
from .prediction import PredictionNet

class Model(BasicModule):

    def __init__(self, opt, Net):
        super(Model, self).__init__()
        self.opt = opt

        self.model_name = self.opt.model

        self.user_net = Net(opt, 'user')
        self.item_net = Net(opt, 'item')

        if self.opt.ui_merge == 'cat' and self.opt.r_id_method == 'cat':
            feature_dim = self.opt.attention_size * 4
        elif self.opt.ui_merge != 'cat' and self.opt.r_id_method == 'add':
            feature_dim = self.opt.attention_size
        elif (self.opt.ui_merge == 'cat' and self.opt.r_id_method == 'add') or \
                (self.opt.ui_merge != 'cat' and self.opt.r_id_method == 'cat'):
            feature_dim = self.opt.attention_size * 2

        self.opt.feature_dim = feature_dim
        self.predict_net = PredictionNet(opt)
        self.dropout = nn.Dropout(self.opt.drop_out)

    def unpack_input(self, x):

        uids, iids = list(zip(*x))
        uids = list(uids)
        iids = list(iids)
        user_reviews = self.opt.user_list[uids]
        user_summary = self.opt.user_summary[uids]
        user_item2id = self.opt.user2itemid_dict[uids]  # 检索出该user对应的item id

        item_reviews = self.opt.item_list[iids]
        item_summary = self.opt.item_summary[iids]
        item_user2id = self.opt.item2userid_dict[iids]  # 检索出该item对应的user id

        train_data = [user_reviews, user_summary, item_reviews, item_summary, user_item2id, item_user2id, uids, iids]
        train_data = list(map(lambda x: torch.LongTensor(x).cuda(), train_data))
        return train_data

    def forward(self, datas):
        user_reviews, user_summary, user_review_char, user_summary_char, item_reviews, item_summary, \
            item_review_char, item_summary_char, user_item2id, item_user2id, uids, iids = datas  # self.unpack_input(datas)

        user_all_feature = self.user_net(user_reviews, user_summary, user_review_char, user_summary_char, uids, user_item2id, self.opt.u_max_r)
        item_all_feature = self.item_net(item_reviews, item_summary, item_review_char, item_summary_char, iids, item_user2id, self.opt.i_max_r)
        # -------------- the method for merge the user feature and item feature --------------
        if self.opt.ui_merge == 'cat':
            ui_feature = torch.cat([user_all_feature, item_all_feature], 1)
        elif self.opt.ui_merge == 'add':
            ui_feature = user_all_feature + item_all_feature
        elif self.opt.ui_merge == 'dot':
            ui_feature = user_all_feature * item_all_feature

        # -------------- the method for output layer  --------------
        ui_feature = F.relu(ui_feature)
        ui_feature = self.dropout(ui_feature)
        output = self.predict_net(ui_feature, uids, iids).squeeze(1)
        return output
