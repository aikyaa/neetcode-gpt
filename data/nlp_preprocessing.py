import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_s = positive + negative

        vocab = sorted({word for s in all_s for word in s.split()})
        word_id = {word: idx + 1 for idx, word in enumerate(vocab)}

        encoded = [torch.tensor([word_id[word] for word in s.split()]) for s in all_s]
        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True)

        return padded
