# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN


class AMVRS(nn.Module):
    '''
    Attentive Multi View RS
    '''
    def __init__(self, opt, uori='user'):
        super(AMVRS, self).__init__()
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
        self.summary_linear = nn.Linear(self.opt.s_filters_num, self.opt.attention_size, bias=False)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.attention_size)
        self.attention_linear = nn.Linear(self.opt.attention_size, 1)

        self.r_fc_linear = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)
        self.s_fc_linear = nn.Linear(self.opt.s_filters_num, self.opt.attention_size)

        self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim)
        self.s_encoder = CNN(self.opt.s_filters_num, self.opt.kernel_size, self.opt.word_dim)

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb()
        self.init_model_weight()

    def attention_layer(self, k, v, fc1, fc2, fc3):

        score = fc3(F.relu(fc1(v) + fc2(k)))
        att_weight = F.softmax(score, 1)
        feature = v * att_weight
        feature = feature.sum(1)
        return feature

    def forward(self, review, summary, index, u_i_id, max_num):

        # --------------- word embedding ----------------------------------
        review = self.word_embs(review)  # size * 300
        summary = self.word_embs(summary)
        if self.opt.use_word_drop:
            review = self.dropout(review)
            summary = self.dropout(summary)

        id_emb = self.id_embedding(index)
        u_i_id_emb = self.u_i_id_embedding(u_i_id)

        # -------- attention cnn for review and summary--------------------
        r_fea = self.r_encoder(review, max_num, self.opt.r_max_len)  # 32 * 11 * 100
        s_fea = self.s_encoder(summary, max_num, self.opt.s_max_len)  # 32 * 11 * 100

        # ------------------linear attention-------------------------------
        r_fea = self.attention_layer(F.relu(u_i_id_emb), r_fea, self.review_linear, self.id_linear, self.attention_linear)  # k v fc1 fc2 fc3
        s_fea = self.attention_layer(F.relu(u_i_id_emb), s_fea, self.summary_linear, self.id_linear, self.attention_linear)

        r_fea = self.dropout(r_fea)
        s_fea = self.dropout(s_fea)

        r_fea = (self.r_fc_linear(r_fea)).unsqueeze(1)
        s_fea = (self.s_fc_linear(s_fea)).unsqueeze(1)

        if self.opt.use_view_att:
            rs_mix = torch.cat([r_fea, s_fea], 1)  # 32 * 2 * 32
            score = rs_mix.bmm(id_emb.unsqueeze(2))
            view_weight = F.softmax(score, 1)
            rs_mix = rs_mix * view_weight
            rs_mix = rs_mix.sum(1)
        else:
            # rs_mix = torch.cat([r_fea, s_fea], 1)
            rs_mix = r_fea + s_fea

        all_feature = rs_mix + id_emb
        return all_feature

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.r_encoder.cnn.weight)
        nn.init.uniform_(self.r_encoder.cnn.bias, a=-0.1, b=0.1)

        nn.init.xavier_uniform_(self.s_encoder.cnn.weight)
        nn.init.uniform_(self.s_encoder.cnn.bias, a=-0.1, b=0.1)

        nn.init.xavier_uniform_(self.review_linear.weight)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.xavier_uniform_(self.summary_linear.weight)
        # nn.init.constant_(self.summary_linear.bias, 0.1)

        linear_layers = [self.r_fc_linear, self.s_fc_linear,
                         self.attention_linear, self.id_linear]
        for linear in linear_layers:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0.1)

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.xavier_normal_(self.id_embedding.weight)
        nn.init.xavier_normal_(self.u_i_id_embedding.weight)
