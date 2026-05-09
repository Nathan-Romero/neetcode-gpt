from typing import List, Dict


class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._greedy_tokenize(str(n), vocab) for n in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(len(self._greedy_tokenize(text, vocab)) / len(text.split()), 4)

    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0

        while i < len(text):
            if best := next(
                (
                    substr
                    for k in range(len(text) - i, 0, -1)
                    if (substr := text[i : i + k]) in vocab
                ),
                None,
            ):
                tokens.append(best)
                i += len(best)
            else:
                tokens.append(text[i])
                i += 1

        return tokens