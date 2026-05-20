import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.genkey = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.genquery = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.genval = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.genkey(embedded)
        q = self.genquery(embedded)
        v = self.genval(embedded)

        scores = q @ torch.transpose(k, 1, 2)
        context_len, dk = k.shape[1], k.shape[2]
        scores /= dk**0.5

        lower_tri = torch.tril(torch.ones(context_len, context_len))
        mask = lower_tri == 0
        scores = scores.masked_fill(mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim=2)
        out = scores @ v

        return torch.round(out, decimals=4)
