from typing import List, Dict


class Solution:
    def tokenize(self, text: str, vocab: Dict[str, int]) -> List[int]:
        tokens = []
        i = 0
        while i < len(text):
            best = None
            for d in range(len(text) - i, 0, -1):
                if text[i : i + d] in vocab:
                    best = text[i : i + d]
                    tokens.append(best)
                    i += d
                    break
            if best is None:
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        token_lists = []
        for number in numbers:
            text = str(number)
            tokens = self.tokenize(text, vocab)
            token_lists.append(tokens)
        return token_lists

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self.tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)
