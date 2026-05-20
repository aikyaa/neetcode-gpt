from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break
            pairs = {}
            for i in range(len(tokens) - 1):
                key = (tokens[i], tokens[i + 1])
                pairs[key] = pairs.get(key, 0) + 1
            if not pairs:
                break
            most_freq = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == most_freq)
            best = candidates[0]
            merges.append([candidates[0][0], candidates[0][1]])

            i = 0
            updated = []

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == candidates[0][0]
                    and tokens[i + 1] == candidates[0][1]
                ):
                    updated.append(candidates[0][0] + candidates[0][1])
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            tokens = updated
        return merges
