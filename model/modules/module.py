import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_head, w_qkv, w_out): # w_qkv는 fc layer로 취급
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.w_query = copy.deepcopy(w_qkv) # embed_dim x model_dim
        self.w_key = copy.deepcopy(w_qkv)   # embed_dim x model_dim
        self.w_value = copy.deepcopy(w_qkv) # embed_dim x model_dim
        self.w_out = w_out

    def attention(self, query, key, value, mask):
        key_dim = key.shape[-1] # 마지막 차원
        score = query @ key.T
        scaled_score = score / key_dim
        if mask is not None:
            scaled_score = scaled_score.masked_fill(mask==0, -math.inf)
        out = F.softmax(scaled_score, dim=-1) @ value

        return out

    # fc는 torch.nn의 Linear transform
    def head_split(self, embed_matrix, fc):         # n x seq_len x embed_dim
        out = fc(embed_matrix)                      # n x seq_len x model_dim
        # 가중치 행렬을 헤드 개수만큼 쪼갬 (-1은 자동);   # n x seq_len x num_head x key_dim
        out = out.view(self.n_batch, -1, self.num_head, self.model_dim//self.num_head)

        return out.transpose(1, 2)                  # n x num_head x seq_len  x key_dim

    def forward(self, query, key, value, mask=None):
        self.n_batch = key.size(0)

        query = self.head_split(query, self.w_query)
        key = self.head_split(key, self.w_key)
        value = self.head_split(value, self.w_key)

        out = self.attention(query, key, value, mask)
        out = out.T.contiguous().view(self.n_batch, -1, self.model_dim) # concat
        out = self.w_out(out)

        return out


class FFN(nn.Module):
    def __init__(self, fc1, fc2):
        super(FFN, self).__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.activation = nn.ReLU()

    def forward(self, attention_matrix):
        out = self.fc1(attention_matrix)
        out = self.activation(out)
        out = self.fc2(out)

        return out


class ResidualConnection(nn.Module):
    def __init__(self, norm, dr_rate=0):
        super(ResidualConnection, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, attention_matrix, sub_layer):
        out = self.norm(attention_matrix)
        out = sub_layer(out)
        out = self.dropout(out)

        return out + attention_matrix