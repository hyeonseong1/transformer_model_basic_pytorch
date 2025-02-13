import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.module import ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(self, attention, ffn, norm, dr_rate):
        super(DecoderBlock, self).__init__()
        self.masked_attention = attention
        self.residual_1 = ResidualConnection(copy.deepcopy(norm), dr_rate)
        self.cross_attention = attention
        self.residual_2 = ResidualConnection(copy.deepcopy(norm), dr_rate)
        self.ffn = ffn
        self.residual_3 = ResidualConnection(copy.deepcopy(norm), dr_rate)

    def forward(self, target, context_vec, attn_mask, src_mask): # encoder src
        src_target = target
        out = self.masked_attention(src_target, src_target, src_target, attn_mask)
        out = self.residual_1(src_target, out)
        src_target = out
        out = self.cross_attention(context_vec, context_vec, out, src_mask)
        out = self.residual_2(src_target, out)
        src_target = out
        out = self.ffn(out)
        out = self.residual_3(src_target, out)

        return out

class Decoder(nn.Module):
    def __init__(self, decoder_layer, n_block, norm):
        super(Decoder, self).__init__()
        self.n_block = n_block
        self.layers = []
        for _ in range(self.n_block):
            self.layers.append(copy.deepcopy(decoder_layer))
        self.norm = norm

    def forward(self, target, context_vec, attn_mask, src_mask):
        out = target
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)

        return out