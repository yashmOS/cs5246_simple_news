from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .preprocess import extract_dates, extract_entities, extract_numbers, tokenize


def ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))


def rouge_n(pred: str, ref: str, n: int = 1) -> float:
    pred_t = tokenize(pred)
    ref_t = tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0
    pred_ngrams = ngrams(pred_t, n)
    ref_ngrams = ngrams(ref_t, n)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    recall = overlap / max(1, sum(ref_ngrams.values()))
    precision = overlap / max(1, sum(pred_ngrams.values()))
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_t = tokenize(pred)
    ref_t = tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0
    lcs = lcs_length(pred_t, ref_t)
    recall = lcs / len(ref_t)
    precision = lcs / len(pred_t)
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def count_sentences(text: str) -> int:
    return len(re.findall(r'[.!?]+', text)) or 1


def count_syllables_in_word(word: str) -> int:
    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    prev = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev:
            count += 1
        prev = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def readability(text: str) -> Dict[str, float]:
    words = [w for w in re.findall(r'[A-Za-z]+', text)]
    n_words = max(1, len(words))
    n_sent = max(1, count_sentences(text))
    syllables = sum(count_syllables_in_word(w) for w in words)
    fre = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    fkgl = 0.39 * (n_words / n_sent) + 11.8 * (syllables / n_words) - 15.59
    return {
        'word_count': n_words,
        'sentence_count': n_sent,
        'avg_sentence_len': n_words / n_sent,
        'flesch_reading_ease': fre,
        'fkgl': fkgl,
    }


def set_coverage(source_items: Iterable[str], pred_items: Iterable[str]) -> float:
    source_set = {x.lower() for x in source_items if str(x).strip()}
    pred_set = {x.lower() for x in pred_items if str(x).strip()}
    if not source_set:
        return 1.0
    return len(source_set & pred_set) / len(source_set)


def fact_coverage(source: str, pred: str) -> Dict[str, float]:
    return {
        'entity_coverage': set_coverage(extract_entities(source), extract_entities(pred)),
        'number_coverage': set_coverage(extract_numbers(source), extract_numbers(pred)),
        'date_coverage': set_coverage(extract_dates(source), extract_dates(pred)),
    }


def novelty_metrics(source: str, pred: str) -> Dict[str, float]:
    source_tokens = set(tokenize(source))
    pred_tokens = tokenize(pred)
    if not pred_tokens:
        return {'novel_token_ratio': 0.0}
    novel = [t for t in pred_tokens if t not in source_tokens]
    return {'novel_token_ratio': len(novel) / len(pred_tokens)}


def evaluate_output(source: str, reference: str, pred: str) -> Dict[str, float]:
    metrics = {
        'rouge1': rouge_n(pred, reference, 1) if reference else 0.0,
        'rouge2': rouge_n(pred, reference, 2) if reference else 0.0,
        'rougel': rouge_l(pred, reference) if reference else 0.0,
    }
    metrics.update(readability(pred))
    metrics.update(fact_coverage(source, pred))
    metrics.update(novelty_metrics(source, pred))
    return metrics
