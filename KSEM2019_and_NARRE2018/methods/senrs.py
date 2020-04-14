# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN
from .encoder import CharCNN, CharRNN


class SENRS(nn.Module):
    '''
    Sentiment Enhanced RS
    '''
    def __init__(self, opt, uori='user'):
        super(SENRS, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)  # user/item num * 32
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.char_embs = nn.Embedding(self.opt.char_size, self.opt.char_emb_size)  # * 50
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)

        self.review_linear = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)
        self.summary_linear = nn.Linear(self.opt.s_filters_num, self.opt.attention_size, bias=False)
        self.id_linear = nn.Linear(self.opt.id_emb_size, self.opt.attention_size, bias=False)
        self.attention_linear = nn.Linear(self.opt.attention_size, 1)

        self.fc_layer = nn.Linear(self.opt.r_filters_num, self.opt.attention_size)

        # self.char_cnn = CharCNN(self.opt.char_cnn_filters, self.opt.kernel_size, self.opt.char_emb_size)
        self.char_rnn = CharRNN(self.opt.char_emb_size, self.opt.char_emb_size)
        self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim+2*self.opt.char_emb_size)
        self.s_encoder = CNN(self.opt.s_filters_num, self.opt.kernel_size, self.opt.word_dim+2*self.opt.char_emb_size)

        self.word_s_linear = self.mlp_layer()
        self.review_s_linear = self.mlp_layer()

        self.word_global_qv = nn.Parameter(torch.randn(self.opt.s_filters_num))
        self.review_global_qv = nn.Parameter(torch.randn(self.opt.s_filters_num))

        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb()
        self.init_model_weight()

    def mlp_layer(self):
        return nn.Sequential(
                    nn.Linear(self.opt.s_filters_num, self.opt.query_mlp_size),
                    nn.LayerNorm(self.opt.query_mlp_size),
                    nn.ReLU(),
                    nn.Linear(self.opt.query_mlp_size, self.opt.s_filters_num),
            )

    def forward(self, review, summary, review_char, summary_char, index, u_i_id, max_num):

        # --------------- char embedding ----------------------------------
        review_char = self.char_embs(review_char)
        summary_char = self.char_embs(summary_char)
        review_char_embs = self.char_rnn(review_char, max_num, self.opt.r_max_len, self.opt.word_max_len)
        summary_char_embs = self.char_rnn(summary_char, max_num, self.opt.s_max_len, self.opt.word_max_len)

        # --------------- word embedding ----------------------------------
        review = self.word_embs(review)  # 32 * 11 * 224 * 300
        summary = self.word_embs(summary)  # 32 * 23 * 40 * 300

        review = torch.cat([review, review_char_embs], -1)
        summary = torch.cat([summary, summary_char_embs], -1)
        if self.opt.use_word_drop:
            review = self.dropout(review)
            summary = self.dropout(summary)

        id_emb = self.id_embedding(index)  # 32 * 1 * 32
        u_i_id_emb = self.u_i_id_embedding(u_i_id)  # 32 * 11 * 32
        # -------- cnn for review and summary--------------------
        s_fea = self.s_encoder(summary, max_num, self.opt.s_max_len, pooling="MAX")  # 32 * 23 * 100

        word_qv = self.word_s_linear(s_fea + self.word_global_qv)
        review_qv = self.review_s_linear(s_fea + self.review_global_qv)

        r_fea = self.r_encoder(review, max_num, self.opt.r_max_len, pooling="att", qv=word_qv)  # 32 * 11 * 100

        # ------------------linear attention-------------------------------
        if self.opt.use_id_att:
            rs_mix = F.relu(self.review_linear(r_fea) + self.summary_linear(review_qv) + self.id_linear(F.relu(u_i_id_emb)))
        else:
            rs_mix = F.relu(self.review_linear(r_fea) + self.summary_linear(review_qv))
        att_score = self.attention_linear(rs_mix)
        att_weight = F.softmax(att_score, 1)
        r_fea = r_fea * att_weight

        feature = r_fea.sum(1)

        if self.opt.r_id_method == 'cat':
            feature = torch.cat([self.fc_layer(feature), id_emb], -1)
        elif self.opt.r_id_method == 'add':
            feature = self.fc_layer(feature) + id_emb

        return feature

    def init_model_weight(self):
        nn.init.xavier_normal_(self.r_encoder.cnn.weight)
        nn.init.uniform_(self.r_encoder.cnn.bias, a=-0.1, b=0.1)
        nn.init.xavier_normal_(self.s_encoder.cnn.weight)
        nn.init.uniform_(self.s_encoder.cnn.bias, a=-0.1, b=0.1)

        # nn.init.xavier_normal_(self.char_cnn.cnn.weight)
        # nn.init.uniform_(self.char_cnn.cnn.bias, a=-0.1, b=0.1)

        nn.init.uniform_(self.review_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.review_linear.bias, 0.1)

        nn.init.uniform_(self.summary_linear.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.id_linear.weight)

        nn.init.uniform_(self.attention_linear.weight, -0.1, 0.1)
        nn.init.constant_(self.attention_linear.bias, 0.1)

        nn.init.uniform_(self.word_s_linear[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.review_s_linear[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.word_s_linear[-1].weight, -0.1, 0.1)
        nn.init.uniform_(self.review_s_linear[-1].weight, -0.1, 0.1)

        nn.init.uniform_(self.word_global_qv, -0.1, 0.1)
        nn.init.uniform_(self.review_global_qv, -0.1, 0.1)

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
        nn.init.uniform_(self.id_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.u_i_id_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.char_embs.weight, -1.0, 1.0)
