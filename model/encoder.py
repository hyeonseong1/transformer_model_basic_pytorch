import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.module import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, attention, ffn, norm, dr_rate):
        super(EncoderBlock).__init__()
        self.attention = attention
        self.residual_1 = ResidualConnection(copy.deepcopy(norm), dr_rate)
        self.ffn = ffn
        self.residual_2 = ResidualConnection(copy.deepcopy(norm), dr_rate)

    def forward(self, embed_matrix, mask):
        src = embed_matrix
        out = self.attention(src, src, src, mask)
        out = self.residual_1(src, out) # embed_matrix -> (embed_matrix + attention)= r1
        src = out
        out = self.ffn(out)
        out = self.residual_2(src, out) # r1 -> r1 + ffn(r1)

        return out


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_block, norm):
        super(Encoder, self).__init__()
        self.n_block = n_block
        self.layers = []
        for _ in range(self.n_block):
            self.layers.append(copy.deepcopy(encoder_layer))
        self.norm = norm

    def forward(self, src, src_mask): # 여러 개의 인코더 블록 실행
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)

        return out
