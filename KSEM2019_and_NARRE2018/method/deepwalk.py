# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN


class NARRE(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt, uori='user'):
        super(NARRE, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)

        self.review_linear = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.attention_size, bias=False)
        self.attention_linear = nn.Linear(self.opt.attention_size, 1)
        self.fc_layer = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)

        self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb()
        self.init_model_weight()

    def forward(self, review, summary, index, u_i_id, max_num):
        # --------------- word embedding ----------------------------------
        review = self.word_embs(review)  # size * 300
        if self.opt.use_word_drop:
            review = self.dropout(review)

        id_emb = self.id_embedding(index)
        u_i_id_emb = self.u_i_id_embedding(u_i_id)

        # --------cnn for review and summary--------------------
        r_fea = self.r_encoder(review, max_num, self.opt.r_max_len)  # 32 * 5 * 100

        # ------------------linear attention-------------------------------
        rs_mix = F.relu(self.review_linear(r_fea) + self.id_linear(F.relu(u_i_id_emb)))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = r_fea * att_weight
        r_fea = r_fea.sum(1)
        r_fea = self.dropout(r_fea)
        all_feature = self.fc_layer(r_fea) + id_emb

        return all_feature

    def init_model_weight(self):
        nn.init.xavier_normal_(self.r_encoder.cnn.weight)
        nn.init.constant_(self.r_encoder.cnn.bias, 0.1)

        nn.init.uniform_(self.id_linear.weight, -0.1, 0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-1., b=1.)
        nn.init.uniform_(self.u_i_id_embedding.weight, a=-1., b=1.)
