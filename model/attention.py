import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k, q, v = self.key_gen(embedded), self.query_gen(embedded), self.value_gen(embedded)
        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores /= math.sqrt(attention_dim)

        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        scores = scores.masked_fill(lower_triangular == 0, float("-inf"))
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ v, decimals=4)