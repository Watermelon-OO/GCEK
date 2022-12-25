# coding: utf-8
import os

import mxnet as mx

codes_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(codes_path)


class Config(object):

    def __init__(self):
        super(Config, self).__init__()
        # input data path
        self.rel2id_path = os.path.join(codes_path, 'data/relation2id_NYT.txt')
        self.kgrel2id_path = os.path.join(codes_path, 'data/relation2id_KB.txt')
        self.ent2id_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/entity2id_520K.txt')
        self.ent2id_path_570K = os.path.join(codes_path, 'data/NYT_data/NYT_570088/ent2id_570K.txt')
        ###############################################
        # after transform, based on preprocessed data #
        ###############################################
        self.train_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/train_520K.npy')
        self.testAll_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/test_520K_All.npy')

        self.train_path_570K = os.path.join(codes_path, 'data/NYT_data/NYT_570088/train_570K.npy')
        self.testAll_path_570K = os.path.join(codes_path, 'data/NYT_data/NYT_570088/test_570K_All.npy')
        ##########################################
        # after preprocess, organized in the bag #
        ##########################################
        # original data
        self.bag_train_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/bags_train.txt')
        self.bag_test_path = os.path.join(root_path, 'data/NYT_data/NYT_522611/bags_test.txt')
        self.bag_train_path_570K = os.path.join(root_path, 'data/NYT_data/NYT_570088/bags_train.txt')
        # pre-train vector path
        self.wv_path = os.path.join(codes_path, 'data/word_vector.npy')
        self.lwpe_path = os.path.join(codes_path, 'data/lwpe_vector.npy')
        self.rwpe_path = os.path.join(codes_path, 'data/rwpe_vector.npy')
        self.entEmbed_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/FB60K_entEmbeds_woNA_d100.npy')
        self.relEmbed_path_520K = os.path.join(codes_path, 'data/NYT_data/NYT_522611/FB60K_relEmbeds_woNA_d100.npy')
        self.entEmbed_path_570K = os.path.join(codes_path,
                                               'data/NYT_data/NYT_570088/FB60K_entEmbeds_woNA_d100_570K.npy')
        self.relEmbed_path_570K = os.path.join(codes_path,
                                               'data/NYT_data/NYT_570088/FB60K_relEmbeds_woNA_d100_570K.npy')
        self.entEmbed_path = self.entEmbed_path_570K
        self.relEmbed_path = self.relEmbed_path_570K
        # sentence max length
        self.fixLen_sen = 80
        # look-up table size
        self.vocab_size = 114043
        self.vocab_cls_size = 114044
        self.pos_size = 102
        self.rel_num = 53
        self.ent_num_520K = 69506
        self.ent_num_570K = 69513
        self.ent_num = self.ent_num_520K
        self.kgrel_num = 1322
        # vector dimensions
        self.word_dim = 50
        self.wpe_dim = 5
        self.embed_dim = self.word_dim + self.wpe_dim * 2
        self.ent_dim = 100
        self.rel_dim = 100
        # model hyper-parameter
        self.num_epochs = 10
        self.batch_size = 160
        self.optimizer = 'momentum'
        self.HyperParams = 0
        # GPU
        self.ctx = [mx.cpu()]
        # model
        self.lib = 'baseline'
        self.model = 'BaseAtt'
        self.encoder = 'pcnn'
        self.att = 'baseAtt'
        # normalize embeddings
        self.is_norm_emb = False
        # other setting
        self.data_version = 520
        self.ld = 0
        self.record = 0
        self.nni = 0
