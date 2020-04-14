# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFM_net(nn.Module):

    def __init__(self, dim, user_num, item_num):
        super(LFM_net, self).__init__()

        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # -------------------------LFM-user/item-bias-----------------------
        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))

        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.fc.weight, a=-0.1, b=0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.constant_(self.b_users, 0.1)
        nn.init.constant_(self.b_items, 0.1)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]

class FM_net(nn.Module):

    def __init__(self, dim):
        super(FM_net, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)
        # ------------------------------FM----------------------------------
        self.fm_V = nn.Parameter(torch.randn(5, dim))

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias, a=0, b=1)

        nn.init.uniform_(self.fm_V)

    def build_fm(self, input_vec):
        '''
        y = w_0 + \sum {w_ix_i} + \sum_{i=1}\sum_{j=i+1}<v_i, v_j>x_ix_j
        factorization machine layer
        refer: https://github.com/vanzytay/KDD2018_MPCN/blob/master/tylib/lib
                      /compose_op.py#L13
        '''
        # linear part: first two items
        fm_linear_part = self.fc(input_vec)

        fm_interactions_1 = torch.mm(input_vec, self.fm_V.t())
        fm_interactions_1 = torch.pow(fm_interactions_1, 2)

        fm_interactions_2 = torch.mm(torch.pow(input_vec, 2),
                                     torch.pow(self.fm_V, 2).t())
        fm_output = 0.5 * torch.sum(fm_interactions_1 - fm_interactions_2) + fm_linear_part
        return fm_output

    def forward(self, feature):
        return self.build_fm(feature)

class MLP_net(nn.Module):

    def __init__(self, dim):
        super(MLP_net, self).__init__()
        self.dim = dim
        # ---------------------------fc_linear------------------------------
        self.fc = nn.Linear(dim, 1)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias, a=0, b=1)

    def forward(self, feature):
        return F.relu(self.fc(feature)[0])


class PredictionNet(nn.Module):
    '''
        Prediction Rating according to the user and item feature
        : LFM
        : FM
        : MLP
    '''
    def __init__(self, opt):
        super(PredictionNet, self).__init__()
        self.output = opt.output
        if opt.output == "fm":
            self.model = FM_net(opt.feature_dim)
        elif opt.output == "lfm":
            self.model = LFM_net(opt.feature_dim, opt.user_num, opt.item_num)
        elif opt.output == 'mlp':
            self.model = MLP_net(opt.feature_dim)

    def forward(self, feature, uid, iid):
        if self.output == "lfm":
            return self.model(feature, uid, iid)
        else:
            return self.model(feature)
