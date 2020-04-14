# -*- coding: utf-8 -*-
import numpy as np


class DefaultConfig:

    model = 'SENRS'  # prior gru double attention network
    dataset = 'Toys_and_Games_data'

    norm_emb = False  # whether norm word embedding or not
    drop_out = 0.8

    # --------------optimizer---------------------#
    optimizer = 'Adam'
    weight_decay = 1e-4  # optimizer rameteri
    lr = 1e-3
    eps = 1e-8

    # -------------main.py-----------------------#
    seed = 2019
    gpu_id = 1
    multi_gpu = False
    gpu_ids = []
    use_gpu = True   # user GPU or not
    num_epochs = 20  # the number of epochs for training
    num_workers = 20  # how many workers for loading data

    load_ckp = False
    ckp_path = ""
    fine_tune = True
    # ----------for confirmint the data -------------#
    use_word_embedding = True

    #  ----------id_embedding------------------------#
    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 100

    # ----------CNN---------------------------------#
    char_cnn_filters = 50
    char_encoder = 'CNN'
    r_filters_num = 100
    s_filters_num = 100
    kernel_size = 3
    attention_size = 32
    # -----------------gru/cnn----------------------#

    multi_size = 8

    use_id_att = True
    use_view_att = True
    r_id_method = 'add'
    ui_merge = 'dot'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    use_mask = False
    print_opt = 'def'
    prefer_user2v_path = './dataset/train/npy/'
    prefer_item2v_path = './dataset/train/npy/'

    use_word_drop = False

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''

        print("load npy from dist...")
        self.user_list = np.load(self.user_list_path, encoding='bytes')
        self.item_list = np.load(self.item_list_path, encoding='bytes')
        self.item_summary_char = np.load(self.item_summary_char_path, encoding='bytes')
        self.user2itemid_dict = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_dict = np.load(self.item2userid_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class Office_Products_data_Config(DefaultConfig):
    data_root = './dataset/Office_Products_data/'
    w2v_path = './dataset/Office_Products_data/train/npy/w2v.npy'
    id2v_path = './dataset/Office_Products_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Office_Products_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Office_Products_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Office_Products_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Office_Products_data/train/npy/itemSummary2Index.npy'
    vocab_size = 42888
    word_dim = 300
    r_max_len = 248  # review max length
    s_max_len = 32  # summary max length

    train_data_size = 42611
    test_data_size = 10647
    user_num = 4905
    item_num = 2420
    user_mlp = [500, 80]
    item_mlp = [500, 80]
    batch_size = 100
    print_step = 200


class Gourmet_Food_data_Config(DefaultConfig):
    data_root = './dataset/Gourmet_Food_data/'
    w2v_path = './dataset/Gourmet_Food_data/train/npy/w2v.npy'
    id2v_path = './dataset/Gourmet_Food_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Gourmet_Food_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Gourmet_Food_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Gourmet_Food_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Gourmet_Food_data/train/npy/itemSummary2Index.npy'

    user2itemid_path = './dataset/Gourmet_Food_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Gourmet_Food_data/train/npy/item_user2id.npy'
    vocab_size = 66418
    word_dim = 300
    r_max_len = 169  # review max length
    s_max_len = 32  # summary max length

    u_max_r = 12
    i_max_r = 17

    train_data_size = 121003
    test_data_size = 30251
    user_num = 14681
    item_num = 8713
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 64
    print_step = 1000


class Video_Games_data_Config(DefaultConfig):
    data_root = './dataset/Video_Games_data/'
    w2v_path = './dataset/Video_Games_data/train/npy/w2v.npy'
    id2v_path = './dataset/Video_Games_data/matrix/npy/rawMatrix.npy'

    user_list_path = './dataset/Video_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Video_Games_data/train/npy/itemReview2Index.npy'

    user_summary_path = './dataset/Video_Games_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Video_Games_data/train/npy/itemSummary2Index.npy'

    user2itemid_path = './dataset/Video_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Video_Games_data/train/npy/item_user2id.npy'

    vocab_size = 194583
    word_dim = 300
    r_max_len = 517  # review max length
    s_max_len = 29   # summary max length

    train_data_size = 185439
    test_data_size = 23170
    user_num = 24303 + 2
    item_num = 10672 + 2
    u_max_r = 16
    i_max_r = 46
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000

class Toys_and_Games_data_Config(DefaultConfig):
    data_root = './dataset/Toys_and_Games_data/'
    w2v_path = './dataset/Toys_and_Games_data/train/npy/w2v.npy'
    c2v_path = './dataset/Toys_and_Games_data/train/npy/c2v.npy'
    id2v_path = './dataset/Toys_and_Games_data/matrix/npy/rawMatrix.npy'

    user_list_path = './dataset/Toys_and_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Toys_and_Games_data/train/npy/itemReview2Index.npy'

    user_summary_path = './dataset/Toys_and_Games_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Toys_and_Games_data/train/npy/itemSummary2Index.npy'

    user_review_char_path = './dataset/Toys_and_Games_data/train/npy/userReviewChar2Index.npy'
    item_review_char_path = './dataset/Toys_and_Games_data/train/npy/itemReviewChar2Index.npy'

    user_summary_char_path = './dataset/Toys_and_Games_data/train/npy/userSummaryChar2Index.npy'
    item_summary_char_path = './dataset/Toys_and_Games_data/train/npy/itemSummaryChar2Index.npy'

    user2itemid_path = './dataset/Toys_and_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Toys_and_Games_data/train/npy/item_user2id.npy'

    vocab_size = 71039
    char_size = 39
    word_dim = 300
    char_emb_size = 50
    word_max_len = 6  # number of characters in word
    r_max_len = 180  # review max length
    s_max_len = 40  # summary max length

    train_data_size = 134104
    test_data_size = 33493
    user_num = 19412 + 2
    item_num = 11924 + 2
    u_max_r = 9
    i_max_r = 18
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 30
    print_step = 1000

class Kindle_Store_data_Config(DefaultConfig):
    data_root = './dataset/Kindle_Store_data/'
    w2v_path = './dataset/Kindle_Store_data/train/npy/w2v.npy'
    id2v_path = './dataset/Kindle_Store_data/matrix/npy/rawMatrix.npy'

    user_list_path = './dataset/Kindle_Store_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Kindle_Store_data/train/npy/itemReview2Index.npy'

    user_summary_path = './dataset/Kindle_Store_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Kindle_Store_data/train/npy/itemSummary2Index.npy'

    user2itemid_path = './dataset/Kindle_Store_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Kindle_Store_data/train/npy/item_user2id.npy'

    vocab_size = 278914
    word_dim = 300
    r_max_len = 270  # review max length
    s_max_len = 43  # summary max length

    train_data_size = 786159
    test_data_size = 98230
    user_num = 68223 + 2
    item_num = 61934 + 2
    u_max_r = 11
    i_max_r = 23
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 500

class Movies_and_TV_data_Config(DefaultConfig):
    data_root = './dataset/Movies_and_TV_data/'
    w2v_path = './dataset/Movies_and_TV_data/train/npy/w2v.npy'
    id2v_path = './dataset/Movies_and_TV_data/matrix/npy/rawMatrix.npy'

    user_list_path = './dataset/Movies_and_TV_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Movies_and_TV_data/train/npy/itemReview2Index.npy'

    user_summary_path = './dataset/Movies_and_TV_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Movies_and_TV_data/train/npy/itemSummary2Index.npy'

    user2itemid_path = './dataset/Movies_and_TV_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Movies_and_TV_data/train/npy/item_user2id.npy'

    vocab_size = 764339
    word_dim = 300
    r_max_len = 409  # review max length
    s_max_len = 46  # summary max length

    train_data_size = 1358101
    test_data_size = 169716
    user_num = 123960 + 2
    item_num = 50052 + 2
    u_max_r = 21
    i_max_r = 73
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 16
    print_step = 5000

class Clothing_Shoes_and_Jewelry_data_Config(DefaultConfig):
    data_root = './dataset/Clothing_Shoes_and_Jewelry_data/'
    w2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/w2v.npy'
    id2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemSummary2Index.npy'
    vocab_size = 67812
    word_dim = 300
    r_max_len = 97  # review max length
    s_max_len = 31  # summary max length

    train_data_size = 222984
    test_data_size = 55693
    user_num = 39387
    item_num = 23033
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000

class Sports_and_Outdoors_data_Config(DefaultConfig):
    data_root = './dataset/Sports_and_Outdoors_data/'
    w2v_path = './dataset/Sports_and_Outdoors_data/train/npy/w2v.npy'
    id2v_path = './dataset/Sports_and_Outdoors_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Sports_and_Outdoors_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Sports_and_Outdoors_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/itemSummary2Index.npy'
    vocab_size = 100129
    word_dim = 300
    r_max_len = 146  # review max length
    s_max_len = 29  # summary max length

    train_data_size = 237095
    test_data_size = 59242
    user_num = 35598
    item_num = 18357
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000
