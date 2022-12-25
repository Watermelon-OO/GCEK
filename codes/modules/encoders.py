# coding: utf-8
import numpy as np
from mxnet import nd
from mxnet.gluon import nn

from .attention import *

__all__ = ['Custom_Softmax', 'Custom_Softmax_Hybrid', 'embedding_layer', 'conv_block', 'PCNN',
           'GlobalContextEnhancedPCNN']


def Custom_Softmax(data, axis):
    value = data - nd.max(data, axis=axis, keepdims=True)
    exp = nd.exp(value)
    data_softmax = exp / nd.sum(exp, axis=axis, keepdims=True)
    return data_softmax


def Custom_Softmax_Hybrid(F, data, axis):
    value = F.broadcast_minus(data, F.max(data, axis=axis, keepdims=True))
    exp = F.exp(value)
    data_softmax = F.broadcast_div(exp, F.sum(exp, axis=axis, keepdims=True))
    return data_softmax


class embedding_layer(nn.HybridBlock):

    def __init__(self, vector_path, embed_dim, vocab_size, ctx, is_norm_emb=False, grad_req=True, **kwargs):
        super(embedding_layer, self).__init__(**kwargs)
        self.vector_path = vector_path
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.grad_req = grad_req
        self.norm_emb = is_norm_emb
        self.ctx = ctx
        with self.name_scope():
            # embedding layer
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self._init_embed()

    def _init_embed(self):
        # initialize embedding layer
        self.embedding.initialize(ctx=self.ctx)
        # load pre train vector
        embedding_vec = np.load(self.vector_path)
        # normalize embeddings
        if self.norm_emb:
            embedding_vec = np.divide(embedding_vec, np.linalg.norm(embedding_vec, 2, 1, keepdims=True))
            # process value: nan
            embedding_vec[np.isnan(embedding_vec)] = 0
        # transform to nd.NDArray
        embedding_vec = nd.array(embedding_vec)
        self.embedding.weight.set_data(embedding_vec)
        if not self.grad_req:
            self.embedding.collect_params().setattr('grad_req', 'null')

    def hybrid_forward(self, F, vec_ids, **kwargs):
        """
        Args:
            F:
            vec_ids: NDArray or symbol
                shape is (sentences_num, sen_len)
        Returns:
            embedding_vec: NDArray or symbol
                shape is (sentences_num, sen_len, embed_dim)
        """

        # get pre-trained embedding features
        embedding_vec = self.embedding(vec_ids)

        return embedding_vec


class conv_block(nn.HybridBlock):

    def __init__(self, config, kernel_size, **kwargs):
        super(conv_block, self).__init__(**kwargs)
        # configurations
        self.config = config
        self.kernel_size = kernel_size
        with self.name_scope():
            # convolution layer
            self.ConvLayer = nn.Conv2D(self.config.HyperParams['channels'],
                                       in_channels=1,
                                       kernel_size=self.kernel_size,
                                       padding=[1, 0],
                                       strides=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        Args:
            F:
            x: nd.NDArray or symbol
                shape is (sentences_num, sen_len, embed_dim)

        Returns:
            conv_vec: nd.NDArray or symbol
                shape is (sentences_num, channels, sen_len, 1)
        """

        # expand channel. shape is (sentences_num, 1, sen_len, (word_dim + 2wpe_dim))
        bag_embeds = x.expand_dims(axis=1)
        # get all sentences feature
        # convolution result. shape is  (sentences_num, channels, sen_len, 1)
        conv_vec = self.ConvLayer(bag_embeds)
        return conv_vec


class PCNN(nn.HybridBlock):
    """
    PCNN model with mask piece wise max pooling
    """

    def __init__(self, config, kernel_size, **kwargs):
        super(PCNN, self).__init__(**kwargs)
        # configurations
        self.config = config
        self.kernel_size = kernel_size
        self.output_dim = self.config.HyperParams['channels'] * 3
        self.piece_masks = nd.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.name_scope():
            # mask embedding
            self.mask_embed = nn.Embedding(4, 3)
            self.mask_embed.initialize(ctx=self.config.ctx)
            self.mask_embed.weight.set_data(self.piece_masks)
            self.mask_embed.collect_params().setattr('grad_req', 'null')
            # embedding + convolution layer
            self.ConvLayer = conv_block(self.config, kernel_size=self.kernel_size)
            self.PieceWiseMaxPool = nn.MaxPool2D(pool_size=[self.config.fixLen_sen, 1])

    def hybrid_forward(self, F, x, masks, **kwargs):
        """
        Args:
            F:
            x: nd.NDArray or symbol
                shape is (sentences_num, sen_len, embed_dim)
            masks: nd.NDArray or symbol
                shape is (sentences_num, sen_len)

        Returns:
            bag_embeds: nd.NDArray or symbol
                shape is (sentences_num, channels * 3)
        """
        # (sentences_num, channels, sen_len, 1)
        conv_vec = self.ConvLayer(x)
        # mask piece pooling
        # shape is (sentences_num, sen_len, 3)
        mask_vec = self.mask_embed(masks)
        # shape is (sentences_num, 1, sen_len, 3)
        mask_vec = mask_vec.expand_dims(axis=1) * 100
        # shape is (sentences_num, channels, sen_len, 3)
        conv_vec = F.broadcast_add(conv_vec, mask_vec)
        # shape is (sentences_num, channels, 1, 3)
        bag_embeds = self.PieceWiseMaxPool(conv_vec) - 100
        # shape is (sentences_num, channels * 3)
        bag_embeds = bag_embeds.reshape(shape=(-1, self.config.HyperParams['channels'] * 3)).tanh()
        return bag_embeds


class KnowledgeAwareWordEmbeds(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(KnowledgeAwareWordEmbeds, self).__init__(**kwargs)
        self.config = config
        self.d_h = self.config.word_dim + self.config.ent_dim
        self.embed_dim = self.d_h
        # hyper params
        self.Lambda = self.config.HyperParams['lambda']
        with self.name_scope():
            # embedding layer
            self.word_embedding_layer = embedding_layer(self.config.wv_path, self.config.word_dim,
                                                        self.config.vocab_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=True)
            self.lwpe_embedding_layer = embedding_layer(self.config.lwpe_path, self.config.wpe_dim,
                                                        self.config.pos_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=True)
            self.rwpe_embedding_layer = embedding_layer(self.config.rwpe_path, self.config.wpe_dim,
                                                        self.config.pos_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=True)
            self.ent_embedding_layer = embedding_layer(self.config.entEmbed_path, self.config.ent_dim,
                                                       self.config.ent_num, self.config.ctx,
                                                       grad_req=self.config.HyperParams['grad_req'])
            # custom weight params
            self.weight_g1 = self.params.get('weight_g1', shape=(self.d_h, self.d_h))
            self.bias_g1 = self.params.get('bias_g1', shape=(self.d_h, 1))
            self.weight_g2 = self.params.get('weight_g2', shape=(self.d_h, self.config.embed_dim))
            self.bias_g2 = self.params.get('bias_g2', shape=(self.d_h, 1))

    def _get_KnowledgeAwareEmbedding(self, F, words_embeds: nd.NDArray, lpos_embeds: nd.NDArray,
                                     rpos_embeds: nd.NDArray, ent_embeds: nd.NDArray, **kwargs) -> nd.NDArray:
        """extract entities embedding from pre-train Knowledge Graph embeddings,
            and concat it with original word embeddings which combine wpe.
        Args:
            words_embeds: nd.NDArray or symbol
                input word embeddings
                shape is (sentences_num, sen_len, word_dim)
            lpos_embeds: nd.NDArray or symbol
                input word embeddings
                shape is (sentences_num, sen_len, wpe_dim)
            rpos_embeds: nd.NDArray or symbol
                input word embeddings
                shape is (sentences_num, sen_len, wpe_dim)
            ent_embeds: nd.NDArray or symbol
                head entity and tail entity's index in the sentences
                shape is (sentences_num, 2, ent_dim)
            kwargs: nd.NDArray or symbol
                custom params
        Returns:
            knowledgeAware_embeds: nd.NDArray or symbol
                Knowledge-Aware embedding, shape is (sentences_num, sen_len, word_dim+ent_dim)
        """

        # get word+wpe embedding
        # shape is (sentences_num, sen_len, embed_dim)
        wordsWithWPE_embeds = F.concat(words_embeds, lpos_embeds, rpos_embeds, dim=2)

        # get word+knowledge embedding
        # shape is (sentences_num, 1, ent_dim)
        KG_embeds = ent_embeds.slice(begin=(0, 1, 0), end=(None, 2, None)) - \
                    ent_embeds.slice(begin=(0, 0, 0), end=(None, 1, None))
        # shape is (sentences_num, sen_len, ent_dim)
        KG_embeds = F.broadcast_axis(KG_embeds, axis=1, size=self.config.fixLen_sen)
        # shape is (sentences_num, sen_len, word_dim+ent_dim)
        wordsWithKG_embeds = F.concat(words_embeds, KG_embeds, dim=2)
        # fusion two kinds of embeddings, get final Knowledge-Aware embeddings
        temp_value = F.dot(wordsWithKG_embeds, kwargs.get('weight_g1'), transpose_b=True)
        temp_value = F.broadcast_add(temp_value, F.transpose(kwargs.get('bias_g1')))
        # shape is (sentences_num, sen_len, d_h)
        alpha = F.sigmoid(self.Lambda * temp_value)
        xp_hat = F.dot(wordsWithWPE_embeds, kwargs.get('weight_g2'), transpose_b=True)
        xp_hat = F.tanh(F.broadcast_add(xp_hat, F.transpose(kwargs.get('bias_g2'))))
        # shape is (sentences_num, sen_len, d_h)
        knowledgeAware_embeds = alpha * wordsWithKG_embeds + (1 - alpha) * xp_hat
        return knowledgeAware_embeds

    def hybrid_forward(self, F, ent_ids, words_ids, lpos_ids, rpos_ids, *args, **kwargs):
        """

        Args:
            F:
            ent_ids:  nd.NDArray or symbol
                shape is (sentences_num, 2)
            words_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            lpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            rpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            *args:
            **kwargs:

        Returns:

        """
        words_embeds = self.word_embedding_layer(words_ids)
        lpos_embeds = self.lwpe_embedding_layer(lpos_ids)
        rpos_embeds = self.rwpe_embedding_layer(rpos_ids)
        ent_embeds = self.ent_embedding_layer(ent_ids)
        # shape is (sentences_num, sen_len, d_h)
        knowledgeAware_embeds = self._get_KnowledgeAwareEmbedding(F, words_embeds,
                                                                  lpos_embeds, rpos_embeds,
                                                                  ent_embeds, **kwargs)
        return knowledgeAware_embeds


class KnowledgeAwareWordEmbeds_SplitPos(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(KnowledgeAwareWordEmbeds_SplitPos, self).__init__(**kwargs)
        self.config = config
        self.d_h = self.config.word_dim + self.config.ent_dim
        self.embed_dim = self.config.word_dim * 2 + self.config.ent_dim * 2 + self.config.wpe_dim * 2
        # hyper params
        self.Lambda = self.config.HyperParams['lambda']
        with self.name_scope():
            # embedding layer
            self.word_embedding_layer = embedding_layer(self.config.wv_path, self.config.word_dim,
                                                        self.config.vocab_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=self.config.HyperParams['grad_reqW'])
            self.lwpe_embedding_layer = embedding_layer(self.config.lwpe_path, self.config.wpe_dim,
                                                        self.config.pos_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=True)
            self.rwpe_embedding_layer = embedding_layer(self.config.rwpe_path, self.config.wpe_dim,
                                                        self.config.pos_size, self.config.ctx,
                                                        is_norm_emb=self.config.HyperParams['is_norm_emb'],
                                                        grad_req=True)
            self.ent_embedding_layer = embedding_layer(self.config.entEmbed_path, self.config.ent_dim,
                                                       self.config.ent_num, self.config.ctx,
                                                       grad_req=self.config.HyperParams['grad_reqK'])
            # custom weight params
            self.proj_k1 = nn.Dense(in_units=self.d_h, units=self.d_h, use_bias=True, flatten=False)
            self.proj_k2 = nn.Dense(in_units=self.config.word_dim, units=self.d_h,
                                    use_bias=True, activation='tanh', flatten=False)

            self.proj_e1 = nn.Dense(in_units=self.d_h, units=self.d_h, use_bias=True, flatten=False)
            self.proj_e2 = nn.Dense(in_units=self.config.word_dim, units=self.d_h,
                                    use_bias=True, activation='tanh', flatten=False)

    def _get_KnowledgeAwareEmbedding(self, F, words_embeds: nd.NDArray, lpos_embeds: nd.NDArray,
                                     rpos_embeds: nd.NDArray, entWords_embeds: nd.NDArray,
                                     ent_embeds: nd.NDArray) -> nd.NDArray:
        """generate knowledge-aware word embeddings using structured information from KG and
            semantic information from word embeddings respectively.Then concat them with wpe.
        Args:
            F:
            words_embeds: nd.NDArray or symbol
                shape is (sentences_num, sen_len, word_dim)
            lpos_embeds: nd.NDArray or symbol
                shape is (sentences_num, sen_len, wpe_dim)
            rpos_embeds: nd.NDArray or symbol
                shape is (sentences_num, sen_len, wpe_dim)
            entWords_embeds:  nd.NDArray or symbol
                shape is (sentences_num, 2, word_dim)
            ent_embeds:  nd.NDArray or symbol
                shape is (sentences_num, 2, ent_dim)
        Returns:

        """
        # get word+entity embedding
        # shape is (sentences_num, 1, word_dim*2)
        entPair_embeds = F.reshape(entWords_embeds, (-1, 1, self.config.word_dim * 2))
        # shape is (sentences_num, sen_len, word_dim*2)
        entPair_embeds = F.broadcast_axis(entPair_embeds, axis=1, size=self.config.fixLen_sen)
        # shape is (sentences_num, sen_len, word_dim*3)
        wordsWithEP_embeds = F.concat(words_embeds, entPair_embeds, dim=2)
        # get word+knowledge embedding
        # shape is (sentences_num, 1, ent_dim)
        KG_embeds = ent_embeds.slice(begin=(0, 1, 0), end=(None, 2, None)) - \
                    ent_embeds.slice(begin=(0, 0, 0), end=(None, 1, None))
        # shape is (sentences_num, sen_len, ent_dim)
        KG_embeds = F.broadcast_axis(KG_embeds, axis=1, size=self.config.fixLen_sen)
        # shape is (sentences_num, sen_len, word_dim+ent_dim)b
        wordsWithKG_embeds = F.concat(words_embeds, KG_embeds, dim=2)
        # structured information
        # fusion word embeddings with structured information, get structure-Aware embeddings
        # shape is (sentences_num, sen_len, d_h)
        alpha_k = F.sigmoid(self.Lambda * self.proj_k1(wordsWithKG_embeds))
        # shape is (sentences_num, sen_len, d_h)
        xk_hat = self.proj_k2(words_embeds)
        # shape is (sentences_num, sen_len, d_h)
        structuredAware_embeds = alpha_k * wordsWithKG_embeds + (1 - alpha_k) * xk_hat
        # semantic information
        #  fusion word embeddings with semantic information, get semantic-Aware embeddings
        alpha_e = F.sigmoid(self.Lambda * self.proj_e1(wordsWithEP_embeds))
        xe_hat = self.proj_e2(words_embeds)
        semanticAware_embeds = alpha_e * wordsWithEP_embeds + (1 - alpha_e) * xe_hat
        # fusion two kinds of aware embeddings
        knowledgeAware_embeds = F.concat(structuredAware_embeds, semanticAware_embeds, lpos_embeds, rpos_embeds, dim=2)
        return knowledgeAware_embeds

    def hybrid_forward(self, F, ent_ids, entWord_ids, words_ids, lpos_ids, rpos_ids, *args, **kwargs):
        """

        Args:
            F:
            ent_ids:  nd.NDArray or symbol
                shape is (sentences_num, 2)
            entWord_ids:  nd.NDArray
                shape is (sentences_num, 2)
            words_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            lpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            rpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            *args:
            **kwargs:

        Returns:

        """
        words_embeds = self.word_embedding_layer(words_ids)
        lpos_embeds = self.lwpe_embedding_layer(lpos_ids)
        rpos_embeds = self.rwpe_embedding_layer(rpos_ids)
        entWords_embeds = self.word_embedding_layer(entWord_ids)
        ent_embeds = self.ent_embedding_layer(ent_ids)
        # shape is (sentences_num, sen_len, d_h)
        knowledgeAware_embeds = self._get_KnowledgeAwareEmbedding(F, words_embeds, lpos_embeds, rpos_embeds,
                                                                  entWords_embeds, ent_embeds)
        return knowledgeAware_embeds


class GlobalContextEnhancedPCNN(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super(GlobalContextEnhancedPCNN, self).__init__(**kwargs)
        self.config = config
        with self.name_scope():
            # embedding layer
            self.KnowledgeAwareWordEmbeds = KnowledgeAwareWordEmbeds_SplitPos(config)
            self.d_h = self.KnowledgeAwareWordEmbeds.embed_dim
            # encoder layer
            self.sen_encoder = PCNN(self.config, kernel_size=[3, self.d_h])
            # Structured attention
            self.StructuredAttention = StructuredSelfAttention(hidden_size=self.config.HyperParams['att_hidden_s'],
                                                               mask_req=True, ctxs=self.config.ctx)
        self.output_dim = self.sen_encoder.output_dim

    def hybrid_forward(self, F, ent_ids, entWord_ids, words_ids, lpos_ids, rpos_ids, masks, *args, **kwargs):
        """

        Args:
            F:
            ent_ids:  nd.NDArray or symbol
                shape is (sentences_num, 2)
            entWord_ids: nd.NDArray or symbol
                shape is (sentences_num, 2)
            words_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            lpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            rpos_ids: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            masks: nd.NDArray or symbol
                shape is (sentences_num, sen_len)
            *args:
            **kwargs:

        Returns:

        """
        # shape is (sentences_num, sen_len, d_h)
        knowledgeAware_embeds = self.KnowledgeAwareWordEmbeds(ent_ids, entWord_ids, words_ids, lpos_ids, rpos_ids)
        # shape is (sentences_num, 1, d_h)
        GlobalContext, _ = self.StructuredAttention(knowledgeAware_embeds, knowledgeAware_embeds, masks)
        GlobalContext = F.squeeze(GlobalContext, axis=1)
        # sentence local feature, shape is (sentences_num, channels*3)
        LocalContext = self.sen_encoder(knowledgeAware_embeds, masks)
        return GlobalContext, LocalContext
