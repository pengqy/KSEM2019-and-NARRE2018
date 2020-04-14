import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .encoder import CNN, CharRNN, CharCNN


class Multi_Head(nn.Module):
    '''
    Multi_Head Attention Enhanced RS
    '''
    def __init__(self, opt, uori='user'):
        super(Multi_Head, self).__init__()
        self.opt = opt

        if uori == 'user':
            id_num = self.opt.user_num
            ui_id_num = self.opt.item_num
        else:
            id_num = self.opt.item_num
            ui_id_num = self.opt.user_num

        self.id_embedding = nn.Embedding(id_num, self.opt.id_emb_size)      # user/item num * 32
        self.u_i_id_embedding = nn.Embedding(ui_id_num, self.opt.id_emb_size)
        self.char_embs = nn.Embedding(self.opt.char_size, self.opt.char_emb_size)  # * 50

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300
        self.multi_head_embedding = nn.Parameter(torch.randn(self.opt.multi_size, 2 * self.opt.attention_size))
        self.multi_word_embedding = nn.Parameter(torch.randn(self.opt.multi_size, self.opt.r_filters_num))

        self.attention_linear = nn.Linear(self.opt.attention_size, 1)
        if self.opt.char_encoder == 'CNN':
            self.char_encoder = CharCNN(self.opt.char_cnn_filters, self.opt.kernel_size, self.opt.char_emb_size)
            self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim+self.opt.char_emb_size)
        else:
            self.char_encoder = CharRNN(self.opt.char_emb_size, self.opt.char_emb_size)
            self.r_encoder = CNN(self.opt.r_filters_num, self.opt.kernel_size, self.opt.word_dim+2*self.opt.char_emb_size)

        # self.word_fc = nn.Linear(self.opt.multi_size * self.opt.r_filters_num, self.opt.attention_size)
        # self.review_fc = nn.Linear(self.multi_head_embedding.size(1) * self.opt.multi_size, self.opt.attention_size)
        self.word_fc = self.mlp_layer(self.opt.multi_size * self.opt.r_filters_num, self.opt.attention_size)
        self.review_fc = self.mlp_layer(2*self.opt.attention_size * self.opt.multi_size, self.opt.attention_size)


        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb()
        self.init_model_weight()

    def mlp_layer(self, k1, k2):
        return nn.Sequential(
                    nn.Linear(k1, self.opt.query_mlp_size),
                    nn.LayerNorm(self.opt.query_mlp_size),
                    nn.ReLU(),
                    nn.Linear(self.opt.query_mlp_size, k2),
            )

    def forward(self, review, summary, review_char, summary_char, index, u_i_id, max_num):
        # character embedding
        review_char = self.char_embs(review_char)
        summary_char = self.char_embs(summary_char)
        review_char_embs = self.char_encoder(review_char, max_num, self.opt.r_max_len, self.opt.word_max_len)

        # --------------- word embedding ----------------------------------
        review = self.word_embs(review)  # size * 300
        review = torch.cat([review, review_char_embs], -1)
        if self.opt.use_word_drop:
            review = self.dropout(review)

        id_emb = self.id_embedding(index)
        u_i_id_emb = self.u_i_id_embedding(u_i_id)
        # -------- cnn for review--------------------
        r_fea = self.r_encoder(review, max_num, self.opt.r_max_len, "multi_att", self.multi_word_embedding)
        r_fea = self.word_fc(r_fea)
        r_fea = torch.cat([r_fea, F.relu(u_i_id_emb)], -1)

        # ------------------Multi_Head attention-----------------
        __import__('ipdb').set_trace()
        att_weight = torch.matmul(r_fea, self.multi_head_embedding.t())
        att_score = F.softmax(att_weight, dim=1)
        r_fea = torch.bmm(r_fea.permute(0, 2, 1), att_score)
        feature = r_fea.view(-1, self.opt.multi_size * 2 * self.opt.attention_size)
        feature = self.review_fc(feature) + id_emb
        return feature

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.r_encoder.cnn.weight)
        nn.init.uniform_(self.r_encoder.cnn.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.word_fc[0].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.word_fc[-1].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.review_fc[0].weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.review_fc[-1].weight, a=-0.1, b=0.1)
        # nn.init.constant_(self.word_fc.bias, 0.1)
        # nn.init.constant_(self.review_fc.bias, 0.1)

    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)
        nn.init.uniform_(self.id_embedding.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.multi_head_embedding, a=-0.5, b=0.5)
        nn.init.uniform_(self.char_embs.weight, -1.0, 1.0)
        c2v = torch.from_numpy(np.load(self.opt.c2v_path)).cuda()
        self.char_embs.weight.data.copy_(c2v)
