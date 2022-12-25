# coding: utf-8
import os
import sys

from mxnet import nd
from mxnet.gluon import nn

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_directory)
sys.path.append(root_path)
from modules.encoders import *
from modules.attention import *


class GCEK(nn.Block):

    def __init__(self, config, **kwargs):
        super(GCEK, self).__init__(**kwargs)
        # configurations
        self.config = config
        self.model_name = 'GCEK'
        with self.name_scope():
            # encoder
            self.encoder = GlobalContextEnhancedPCNN(config)
            out_dim_sen = self.encoder.output_dim
            # Attention module
            self.BagAtt = StructuredSelfAttention(hidden_size=self.config.HyperParams['att_hidden_b'])
            # Dropout
            self.Dropout = nn.Dropout(self.config.HyperParams["output_dropout"])
            # relation embedding
            self.rel_weight = self.params.get('weight_sen', shape=(self.config.rel_num, out_dim_sen))
            self.rel_bias = self.params.get('bias_sen', shape=(self.config.rel_num, 1))

    def _predict(self, LocalContext: nd.NDArray, GlobalContext: nd.NDArray,
                 sen_num: nd.NDArray, rel_weight, rel_bias) -> (nd.NDArray, list):
        """inference process

        Args:
            LocalContext: nd.NDArray
                shape is (sentences_num, rel_dim)
            GlobalContext: nd.NDArray
                shape is (sentences_num, word_dim)
            sen_num: nd.NDArray
                shape is (batch_size, )
            rel_weight
            rel_bias
        Returns:
            out: nd.NDArray
               shape is (batch_size, rel_num)
            alpha: list
        """

        bags_feature = []
        att_Weight = []
        start = 0
        ctx = LocalContext.context
        # get current context
        # 抛弃原方法，采用遍历bag的方式，加速推理过程
        query = nd.concat(LocalContext, GlobalContext, dim=1)
        # shape is (1, sen_num, feature_dim)
        query = nd.expand_dims(query, axis=0)
        value = nd.expand_dims(LocalContext, axis=0)
        for current_num in sen_num:
            num = current_num.asscalar()
            # shape is (1, rel_dim)
            bag_feature, alpha = self.BagAtt(query[0:1, start:start + num], value[0:1, start:start + num])
            # 这里没有添加Dropout, gluon在训练时进行了放缩，所以不需要人为进行放缩
            bags_feature.append(bag_feature)
            att_Weight.append(alpha.asnumpy())
            start += num
        # (batch_size, 1, rel_dim)
        bags_feature = nd.concat(*bags_feature, dim=0)
        bags_feature = nd.squeeze(bags_feature, axis=1)
        bags_feature = self.Dropout(bags_feature)
        # (batch_size, rel_dim) * (rel_dim, rel_num) -> (batch_size, rel_num)
        out = nd.dot(bags_feature, rel_weight.data(ctx).T) + rel_bias.data(ctx).T
        return out, att_Weight

    def forward(self, sen_num, ent_ids, entWord_ids, ent_pos, words_ids,
                lpos_ids, rpos_ids, masks, valid_length, labels=None):
        """
        Args:
            sen_num: nd.NDArray
                shape is (batch_size, )
            ent_ids:  nd.NDArray
                shape is (batch_size, 2)
            entWord_ids:  nd.NDArray
                shape is (sentences_num, 2)
            ent_pos:  nd.NDArray
                shape is (sentences_num, 2)
            words_ids: nd.NDArray
                shape is (sentences_num, sen_len)
            lpos_ids: nd.NDArray
                shape is (sentences_num, sen_len)
            rpos_ids: nd.NDArray
                shape is (sentences_num, sen_len)
            masks: nd.NDArray
                shape is (sentences_num, sen_len)
            valid_length: nd.NDArray
                shape is (sentences_num, )
            labels: nd.NDArray
                shape is (batch_size, )

        Returns:
            out: nd.NDArray
                 shape is (batch_size, rel_num)
        """

        # encoding all sentences' features in all bags at a once
        # 将sen_num转换为整型，因为切片操作不支持float
        bag_sen_num = sen_num.astype('int32')
        senEnt_ids = []
        start = 0
        for ids, current_num in enumerate(bag_sen_num):
            num = current_num.asscalar()
            senEnt_ids.extend([ent_ids[ids:ids + 1]] * num)
            start += num
        # shape is (sentences_num, 2)
        senEnt_ids = nd.concat(*senEnt_ids, dim=0)
        # shape is (sentences_num, rel_dim)
        GlobalContext, LocalContext = self.encoder(senEnt_ids, entWord_ids, words_ids, lpos_ids, rpos_ids, masks)
        # shape is (batch_size, rel_num)
        pred_out, _ = self._predict(LocalContext, GlobalContext, bag_sen_num, self.rel_weight, self.rel_bias)
        return pred_out
