import math
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(InputEmbedding, self).__init__() # input 임베딩의 부모 클래스인 nn.Module 호출
        self.embedding = nn.Embedding(vocab_size, input_dim) # 단어 개수, 토큰 개수; (input_dim x vocab_size)

    def forward(self, input_matrix):
        out = self.embedding(input_matrix) * math.sqrt(self.input_dim) # vocab_size 범위 내의 정수 인덱스로 변환
        return out # 임베딩 가중치 행렬