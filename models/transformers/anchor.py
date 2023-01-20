import copy
from typing import Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import inverse_sigmoid

import warnings
import torch
from torch.nn.functional import  linear,softmax,dropout,pad
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F


import torch

from torch.nn import grad  # noqa: F401

from torch._jit_internal import Optional

def multi_head_rcda_forward(query_row,  # type: Tensor
                            query_col,  # type: Tensor
                            key_row,  # type: Tensor
                            key_col,  # type: Tensor
                            value,  # type: Tensor
                            embed_dim_to_check,  # type: int
                            num_heads,  # type: int
                            in_proj_weight,  # type: Tensor
                            in_proj_bias,  # type: Tensor
                            bias_k_row,  # type: Optional[Tensor]
                            bias_k_col,  # type: Optional[Tensor]
                            bias_v,  # type: Optional[Tensor]
                            add_zero_attn,  # type: bool
                            dropout_p,  # type: float
                            out_proj_weight,  # type: Tensor
                            out_proj_bias,  # type: Tensor
                            training=True,  # type: bool
                            key_padding_mask=None,  # type: Optional[Tensor]
                            need_weights=True,  # type: bool
                            attn_mask=None,  # type: Optional[Tensor]
                            use_separate_proj_weight=False,  # type: bool
                            q_row_proj_weight=None,  # type: Optional[Tensor]
                            q_col_proj_weight=None,  # type: Optional[Tensor]
                            k_row_proj_weight=None,  # type: Optional[Tensor]
                            k_col_proj_weight=None,  # type: Optional[Tensor]
                            v_proj_weight=None,  # type: Optional[Tensor]
                            static_k=None,  # type: Optional[Tensor]
                            static_v=None  # type: Optional[Tensor]
                            ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight.
        q_row_proj_weight, q_col_proj_weight, k_row_proj_weight, k_col_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
    """

    bsz, tgt_len, embed_dim = query_row.size()
    src_len_row = key_row.size()[2]
    src_len_col = key_col.size()[1]


    assert embed_dim == embed_dim_to_check
    # assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5


    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_row = linear(query_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 1
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q_col = linear(query_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = embed_dim * 3
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_row = linear(key_row, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 3
    _end = embed_dim * 4
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k_col = linear(key_col, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 4
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = linear(value, _w, _b)

    q_row = q_row.transpose(0, 1)
    q_col = q_col.transpose(0, 1)
    k_row = k_row.mean(1).transpose(0, 1)
    k_col = k_col.mean(2).transpose(0, 1)

    q_row = q_row * scaling
    q_col = q_col * scaling


    q_row = q_row.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q_col = q_col.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    if k_row is not None:
        k_row = k_row.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if k_col is not None:
        k_col = k_col.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().permute(1,2,0,3).reshape(src_len_col,src_len_row, bsz*num_heads, head_dim).permute(2,0,1,3)


    attn_output_weights_row = torch.bmm(q_row, k_row.transpose(1, 2))
    attn_output_weights_col = torch.bmm(q_col, k_col.transpose(1, 2))
    assert list(attn_output_weights_row.size()) == [bsz * num_heads, tgt_len, src_len_row]
    assert list(attn_output_weights_col.size()) == [bsz * num_heads, tgt_len, src_len_col]


    if key_padding_mask is not None:
        mask_row=key_padding_mask[:,0,:].unsqueeze(1).unsqueeze(2)
        mask_col=key_padding_mask[:,:,0].unsqueeze(1).unsqueeze(2)

        attn_output_weights_row = attn_output_weights_row.view(bsz, num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz, num_heads, tgt_len, src_len_col)

        attn_output_weights_row = attn_output_weights_row.masked_fill(mask_row,float('-inf'))
        attn_output_weights_col = attn_output_weights_col.masked_fill(mask_col, float('-inf'))

        attn_output_weights_row = attn_output_weights_row.view(bsz * num_heads, tgt_len, src_len_row)
        attn_output_weights_col = attn_output_weights_col.view(bsz * num_heads, tgt_len, src_len_col)

    attn_output_weights_col = softmax(attn_output_weights_col, dim=-1)
    attn_output_weights_row = softmax(attn_output_weights_row, dim=-1)

    attn_output_weights_col = dropout(attn_output_weights_col, p=dropout_p, training=training)
    attn_output_weights_row = dropout(attn_output_weights_row, p=dropout_p, training=training)

    efficient_compute=True
    # This config will not affect the performance.
    # It will compute the short edge first which can save the memory and run slightly faster but both of them should get the same results.
    # You can also set it "False" if your graph needs to be always the same.
    if efficient_compute:
        if src_len_col<src_len_row:
            b_ein,q_ein,w_ein = attn_output_weights_row.shape
            b_ein,h_ein,w_ein,c_ein = v.shape
            attn_output_row = torch.matmul(attn_output_weights_row,v.permute(0,2,1,3).reshape(b_ein,w_ein,h_ein*c_ein)).reshape(b_ein,q_ein,h_ein,c_ein).permute(0,2,1,3)
            attn_output = torch.matmul(attn_output_weights_col.permute(1,0,2)[:,:,None,:],attn_output_row.permute(2,0,1,3)).squeeze(-2).reshape(tgt_len,bsz,embed_dim)
            ### the following code base on einsum get the same results
            # attn_output_row = torch.einsum("bqw,bhwc->bhqc",attn_output_weights_row,v)
            # attn_output = torch.einsum("bqh,bhqc->qbc",attn_output_weights_col,attn_output_row).reshape(tgt_len,bsz,embed_dim)
        else:
            b_ein,q_ein,h_ein=attn_output_weights_col.shape
            b_ein,h_ein,w_ein,c_ein = v.shape
            attn_output_col = torch.matmul(attn_output_weights_col,v.reshape(b_ein,h_ein,w_ein*c_ein)).reshape(b_ein,q_ein,w_ein,c_ein)
            attn_output = torch.matmul(attn_output_weights_row[:,:,None,:],attn_output_col).squeeze(-2).permute(1,0,2).reshape(tgt_len, bsz, embed_dim)
            ### the following code base on einsum get the same results
            # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
            # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)
    else:
        b_ein, q_ein, h_ein = attn_output_weights_col.shape
        b_ein, h_ein, w_ein, c_ein = v.shape
        attn_output_col = torch.matmul(attn_output_weights_col, v.reshape(b_ein, h_ein, w_ein * c_ein)).reshape(b_ein, q_ein, w_ein, c_ein)
        attn_output = torch.matmul(attn_output_weights_row[:, :, None, :], attn_output_col).squeeze(-2).permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
        ### the following code base on einsum get the same results
        # attn_output_col = torch.einsum("bqh,bhwc->bqwc", attn_output_weights_col, v)
        # attn_output = torch.einsum("bqw,bqwc->qbc", attn_output_weights_row, attn_output_col).reshape(tgt_len, bsz,embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        return attn_output,torch.einsum("bqw,bqh->qbhw",attn_output_weights_row,attn_output_weights_col).reshape(tgt_len,bsz,num_heads,src_len_col,src_len_row).mean(2)
    else:
        return attn_output, None



class MultiheadRCDA(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference:
        Anchor DETR: Query Design for Transformer-Based Detector
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = MultiheadRCDA(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query_row, query_col, key_row, key_col, value)
    """
    __annotations__ = {
        'bias_k_row': torch._jit_internal.Optional[torch.Tensor],
        'bias_k_col': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_row_proj_weight', 'q_col_proj_weight', 'k_row_proj_weight', 'k_col_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadRCDA, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_row_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.q_col_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_row_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.k_col_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(5 * embed_dim, embed_dim))
            self.register_parameter('q_row_proj_weight', None)
            self.register_parameter('q_col_proj_weight', None)
            self.register_parameter('k_row_proj_weight', None)
            self.register_parameter('k_col_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(5 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k_row = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_k_col = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k_row = self.bias_k_col = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_row_proj_weight)
            xavier_uniform_(self.q_col_proj_weight)
            xavier_uniform_(self.k_row_proj_weight)
            xavier_uniform_(self.k_col_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k_row is not None:
            xavier_normal_(self.bias_k_row)
        if self.bias_k_col is not None:
            xavier_normal_(self.bias_k_col)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadRCDA, self).__setstate__(state)

    def forward(self, query_row, query_col, key_row, key_col, value,
                key_padding_mask=None, need_weights=False, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query_row, query_col, key_row, key_col, value: map a query and a set of key-value pairs to an output.
            See "Anchor DETR: Query Design for Transformer-Based Detector" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query_row: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - query_col: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key_row: :math:`(N, H, W, E)`, where W is the source sequence row length, N is the batch size, E is
          the embedding dimension.
        - key_col: :math:`(N, H, W, E)`, where H is the source sequence column length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(N, H, W, E)` where HW is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, H, W)`, ByteTensor, where N is the batch size, HW is the source sequence length.
        - attn_mask: Not Implemented
        - static_k: Not Implemented
        - static_v: Not Implemented
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, HW)` where N is the batch size,
          L is the target sequence length, HW is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_rcda_forward(
                query_row,query_col, key_row, key_col, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k_row,self.bias_k_col, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_row_proj_weight=self.q_row_proj_weight, q_col_proj_weight=self.q_col_proj_weight,
                k_row_proj_weight=self.k_row_proj_weight, k_col_proj_weight=self.k_col_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_rcda_forward(
                query_row,query_col, key_row,key_col, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k_row,self.bias_k_col, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)



class IDAUpV3_bis(nn.Module):
    # bilinear upsampling version of IDA
    def __init__(self, o, channels):
        super(IDAUpV3_bis, self).__init__()
        for i in range(0, len(channels)):
            c = channels[i]
            if i == 0:
                node = nn.Conv2d(c, o, 3, 1, 1)
            else:
                node = nn.Conv2d(c, c, 3, 1, 1)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        
        for i in range(0, -endp, -1):
            i= (i+startp) % endp
            # print(f"layers[{i}] before:", layers[i].shape)
            tmp = nn.functional.adaptive_avg_pool2d(layers[i], layers[i - 1].shape[-2:])  # ch 256-> 256
            node = getattr(self, 'node_' + str(i))
            layers[i-1] = node(tmp + layers[i - 1])
        # layers[startp] = self.up(layers[startp])  # 256=>256
        node = getattr(self, 'node_' + str(startp))
        layers[startp] = node(layers[startp])
        return layers # [layers[startp]]  #keeps multiscsales

class Obj():pass
class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.,
                 activation="relu", num_feature_levels=3,attention_type="RCDA"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.idaup = IDAUpV3_bis(d_model, [d_model for _ in range(num_feature_levels)])
        num_feature_levels = 1

        self.attention_type = attention_type
        encoder_layer = TransformerEncoderLayerSpatial(d_model, dim_feedforward,
                                                          dropout, activation, nhead , attention_type)
        encoder_layer_level = TransformerEncoderLayerLevel(d_model, dim_feedforward,
                                                          dropout, activation, nhead)

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, nhead,
                                                          num_feature_levels, attention_type)

        if num_feature_levels == 1:
            self.num_encoder_layers_level = 0
        else:
            self.num_encoder_layers_level = num_encoder_layers // 2
        self.num_encoder_layers_spatial = num_encoder_layers - self.num_encoder_layers_level

        self.encoder_layers = _get_clones(encoder_layer, self.num_encoder_layers_spatial)
        self.encoder_layers_level = _get_clones(encoder_layer_level, self.num_encoder_layers_level)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.num_layers = num_decoder_layers
        self.decoder = Obj()
        self.decoder.num_layers = num_decoder_layers
        num_classes = 1

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):

        num_pred = self.num_layers
        num_classes = 1
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1][0].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1][0].bias.data, 0)

        nn.init.constant_(self.bbox_embed.layers[-1][0].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])


    def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None):
        out_l = 1
        add_keys = srcs[-1]
        srcs = self.idaup(srcs[:-1], out_l, len(srcs[:-1]))[out_l][:,None]


        # prepare input for decoder
        bs, l, c, h, w = srcs.shape


        reference_points = ref_pts[None,:,:2]
        tgt = query_embed[None]

        mask = masks[out_l].repeat(1,l,1,1).reshape(bs*l,h,w)
        pos_col, pos_row = mask2pos(mask)
        if self.attention_type=="RCDA":
            posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
            posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
            posemb_2d = None
        else:
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1), pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)],dim=-1)
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d))
            posemb_row = posemb_col = None

        outputs = srcs.reshape(bs * l, c, h, w)

        for idx in range(len(self.encoder_layers)):
            outputs = self.encoder_layers[idx](outputs, mask, posemb_row, posemb_col,posemb_2d)
            if idx < self.num_encoder_layers_level:
                outputs = self.encoder_layers_level[idx](outputs, level_emb=self.level_embed.weight.unsqueeze(1).unsqueeze(0).repeat(bs,1,1,1).reshape(bs*l,1,c))

        srcs = outputs.reshape(bs, l, c, h, w)

        output = tgt

        outputs_classes = []
        outputs_coords = []
        hs = []
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, reference_points, srcs, mask, adapt_pos2d=self.adapt_pos2d,
                           adapt_pos1d=self.adapt_pos1d, posemb_row=posemb_row, posemb_col=posemb_col,posemb_2d=posemb_2d)
            reference = inverse_sigmoid(reference_points)
            hs.append(output)
            outputs_class = self.class_embed[lid](output)
            tmp = self.bbox_embed[lid](output)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class[None,])
            outputs_coords.append(outputs_coord[None,])

        output = torch.stack(hs), torch.cat(outputs_classes, dim=0), torch.cat(outputs_coords, dim=0), 3, None
        
        return output


class TransformerEncoderLayerSpatial(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # self attention
        self.self_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None,posemb_2d=None):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        if self.attention_type=="RCDA":
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src2 = self.self_attn((src + posemb_row).reshape(bz, h * w, c), (src + posemb_col).reshape(bz, h * w, c),
                                  src + posemb_row, src + posemb_col,
                                  src, key_padding_mask=padding_mask)[0].transpose(0, 1).reshape(bz, h, w, c)
        else:
            src2 = self.self_attn((src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  (src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  src.reshape(bz, h * w, c).transpose(0, 1), key_padding_mask=padding_mask.reshape(bz, h*w))[0].transpose(0, 1).reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


class TransformerEncoderLayerLevel(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8):
        super().__init__()

        # self attention
        self.self_attn_level = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, level_emb=0):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        src2 = self.self_attn_level(src.reshape(bz, h * w, c) + level_emb, src.reshape(bz, h * w, c) + level_emb,
                                    src.reshape(bz, h * w, c))[0].reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0., activation="relu", n_heads=8,
                 n_levels=3, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # cross attention
        self.cross_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)


        # level combination
        if n_levels>1:
            self.level_fc = nn.Linear(d_model * n_levels, d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, srcs, src_padding_masks=None, adapt_pos2d=None,
                adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None):
        tgt_len = tgt.shape[1]

        query_pos = adapt_pos2d(pos2posemb2d(reference_points))
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        bz, l, c, h, w = srcs.shape
        srcs = srcs.reshape(bz * l, c, h, w).permute(0, 2, 3, 1)

        if self.attention_type == "RCDA":
            query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
            query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src_row = src_col = srcs
            k_row = src_row + posemb_row
            k_col = src_col + posemb_col
            tgt2 = self.cross_attn((tgt + query_pos_x).repeat(l, 1, 1), (tgt + query_pos_y).repeat(l, 1, 1), k_row, k_col,
                                   srcs, key_padding_mask=src_padding_masks)[0].transpose(0, 1)
        else:
            tgt2 = self.cross_attn((tgt + query_pos).repeat(l, 1, 1).transpose(0, 1),
                                   (srcs + posemb_2d).reshape(bz * l, h * w, c).transpose(0,1),
                                   srcs.reshape(bz * l, h * w, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bz*l, h*w))[0].transpose(0,1)

        if l > 1:
            tgt2 = self.level_fc(tgt2.reshape(bz, l, tgt_len, c).permute(0, 2, 3, 1).reshape(bz, tgt_len, c * l))

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt


class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k),nn.LeakyReLU()) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_feature_levels=args.num_feature_levels,
        num_query_position=args.num_query_position,
        num_query_pattern=args.num_query_pattern,
        spatial_prior=args.spatial_prior,
        attention_type=args.attention_type,
)





def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t //2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t //2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


def build_deforamble_transformer(args):

        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=not args.decoder_cross_self,
        sigmoid_attn=args.sigmoid_attn,
        extra_track_attn=args.extra_track_attn,
        memory_bank=args.memory_bank_type == 'MemoryBankFeat'




def build_deforamble_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,

        activation="relu",
        num_feature_levels=args.num_feature_levels,
        attention_type="RCDA")