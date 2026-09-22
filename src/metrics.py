"""Simple, dependency-free metrics: Levenshtein-based WER and CTC greedy decode."""

from __future__ import annotations

import torch


def levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Standard word-level edit distance (insert/delete/substitute = cost 1)."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def word_error_rate(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    """Corpus-level WER = total edit distance / total reference words."""
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        total_errors += levenshtein(ref, hyp)
        total_words += max(len(ref), 1)
    return total_errors / max(total_words, 1)


def ctc_greedy_decode_ids(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> list[list[int]]:
    """Greedy CTC decode: argmax per frame, collapse repeats, drop blanks.

    log_probs: (B, T, V); lengths: (B,) true frame counts (before padding).
    """
    pred_ids = log_probs.argmax(dim=-1)  # (B, T)
    results: list[list[int]] = []
    for b in range(pred_ids.size(0)):
        seq = pred_ids[b, : int(lengths[b].item())].tolist()
        collapsed: list[int] = []
        prev = None
        for token in seq:
            if token != prev and token != blank_id:
                collapsed.append(token)
            prev = token
        results.append(collapsed)
    return results
