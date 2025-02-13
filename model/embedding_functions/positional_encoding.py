import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1024):
        super(PositionalEncoding, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pe = torch.zeros(max_len, embed_dim) # ex 1024x512
        pe.requires_grad = False
        # pos: 토큰의 위치
        position = torch.arange(0, max_len).float().unsqueeze(1) # 포지션 뒤에 차원 추가 1024x1
        # i: 임베딩 차원의 인덱스, 계산 용이를 위해 로그로 변환 <- 계산해보면 똑같음
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))
        # PE(pos, 2i), PE(pos, 2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(device) # 배치 차원으로 확장

    def forward(self, x):
        _, seq_len, _ = x.size()  # 입력의 시퀀스 길이
        pos_embed = self.pe[:, :seq_len, :]  # 입력 길이에 맞는 포지셔널 인코딩만 사용
        out = x + pos_embed
        return out