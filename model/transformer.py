import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model.embedding_functions import input_embedding, positional_encoding


class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, linear_voca):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.linear_voca = linear_voca


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, target, context_vec, attn_mask, src_mask):
        return self.decoder(self.tgt_embed(target), context_vec, attn_mask, src_mask)

    def forward(self, src, tgt):
        src_mask = self.source_mask(src)
        masked_attn_mask = self.attention_mask(tgt)
        cross_attn_mask = self.cross_attention_mask(src, tgt)
        context_vec = self.encode(src, src_mask)
        prediction = self.decode(tgt, context_vec, masked_attn_mask, cross_attn_mask)
        out = self.linear_voca(prediction)
        out = F.log_softmax(out, dim=-1)

        return out, prediction


    # training input; e.g. english
    def source_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    # target input; e.g. french
    def masked_attention_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        return pad_mask & seq_mask

    def make_subsequent_mask(self, query, key): # 어텐션 마스크 -> # 1 0 0
        query_seq_len, key_seq_len = query.size(1), key.size(1)  # 1 1 0
                                                                 # 1 1 1
        low_triangle = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('unit8')
        mask = torch.tensor(low_triangle, dtype=torch.bool, requires_grad=False, device=query.device)

        return mask

    # context vector input -> self-attention input as key, value in decoder
    def cross_attention_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_idx=1):
        # query: (n_batch, query_seq_len) <- tgt
        # key: (n_batch, key_seq_len)     <- context_vec
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        # .ne -> not equal 연산을 통해 마스크 생성, 마스크는 0(False)
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False

        return mask

