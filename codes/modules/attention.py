from mxnet import ndarray as nd
from mxnet.gluon import nn

__all__ = ['StructuredSelfAttention']

class StructuredSelfAttention(nn.HybridBlock):
    """
        [ICLR 2017] A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
    """

    def __init__(self, hidden_size, mask_req=False, ctxs=None, **kwargs):
        super(StructuredSelfAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.proj_w = nn.Dense(units=hidden_size, use_bias=False, flatten=False)
            self.proj_v = nn.Dense(units=1, use_bias=False, flatten=False)
            # mask embedding
            if mask_req:
                self.piece_masks = nd.array([[-10000], [0], [0], [0]])
                self.mask_embed = nn.Embedding(4, 1)
                self.mask_embed.initialize(ctx=ctxs)
                self.mask_embed.weight.set_data(self.piece_masks)
                self.mask_embed.collect_params().setattr('grad_req', 'null')

    def hybrid_forward(self, F, query, value=None, masks=None, *args, **kwargs):
        """

        Args:
            F:
            query: nd.NDArray or symbol
                shape is (batch_size, query_num, query_dim)
            value: nd.NDArray or symbol
                shape is (batch_size, value_num, value_dim)
            masks: nd.NDArray or symbol
                shape is (batch_size, query_num)
            *args:
            **kwargs:

        Returns:
            out: nd.NDArray or symbol
                shape is (batch_size, 1, value_dim)

        """
        if value is None:
            value = query
        # shape is (batch_size, query_num, 1)
        alpha = self.proj_v(F.tanh(self.proj_w(query)))
        if masks is not None:
            # shape is (batch_size, query_num, 1)
            mask_vec = self.mask_embed(masks)
            alpha = alpha + mask_vec
        # alpha = encoders.Custom_Softmax_Hybrid(F, alpha, axis=-2)
        alpha = F.softmax(alpha, axis=-2)
        # shape is (batch_size, 1, sen_dim)
        out = F.batch_dot(F.transpose(alpha, axes=(0, 2, 1)), value)
        return out, alpha

