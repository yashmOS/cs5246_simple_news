from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

from .preprocess import STOPWORDS, extract_dates, extract_entities, extract_numbers, split_sentences, tokenize


def ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


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
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
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
    return len(split_sentences(text))


def count_syllables_in_word(word: str) -> int:
    cleaned = re.sub(r'[^a-z]', '', word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    groups = re.findall(r'[aeiouy]+', cleaned)
    count = len(groups)
    if cleaned.endswith('e'):
        count -= 1
    if cleaned.endswith('le') and len(cleaned) > 2 and cleaned[-3] not in 'aeiouy':
        count += 1
    return max(1, count)


def readability(text: str) -> Dict[str, float]:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or '')
    if not words:
        return {
            'word_count': 0.0,
            'sentence_count': 0.0,
            'avg_sentence_len': 0.0,
            'flesch_reading_ease': 0.0,
            'fkgl': 0.0,
        }

    n_words = len(words)
    n_sent = max(1, count_sentences(text))
    syllables = sum(count_syllables_in_word(word) for word in words)

    fre = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (syllables / n_words)
    fkgl = 0.39 * (n_words / n_sent) + 11.8 * (syllables / n_words) - 15.59

    return {
        'word_count': float(n_words),
        'sentence_count': float(n_sent),
        'avg_sentence_len': n_words / n_sent,
        'flesch_reading_ease': fre,
        'fkgl': fkgl,
    }


def _normalize_items(items: Iterable[str]) -> set[str]:
    normalized = set()
    for item in items:
        cleaned = re.sub(r'\s+', ' ', str(item).strip()).lower()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def set_coverage(source_items: Iterable[str], pred_items: Iterable[str]) -> float:
    source_set = _normalize_items(source_items)
    pred_set = _normalize_items(pred_items)
    if not source_set:
        return 1.0
    return len(source_set & pred_set) / len(source_set)


def set_overlap_metrics(source_items: Iterable[str], pred_items: Iterable[str]) -> Dict[str, float]:
    source_set = _normalize_items(source_items)
    pred_set = _normalize_items(pred_items)

    if not source_set and not pred_set:
        return {'recall': 1.0, 'precision': 1.0, 'f1': 1.0, 'unsupported_ratio': 0.0}
    if not source_set:
        return {
            'recall': 1.0,
            'precision': 0.0 if pred_set else 1.0,
            'f1': 0.0 if pred_set else 1.0,
            'unsupported_ratio': 1.0 if pred_set else 0.0,
        }

    overlap = len(source_set & pred_set)
    recall = overlap / len(source_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    f1 = 0.0 if recall + precision == 0 else (2 * recall * precision / (recall + precision))
    unsupported_ratio = len(pred_set - source_set) / len(pred_set) if pred_set else 0.0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'unsupported_ratio': unsupported_ratio,
    }


def fact_coverage(source: str, pred: str) -> Dict[str, float]:
    entity_stats = set_overlap_metrics(extract_entities(source), extract_entities(pred))
    number_stats = set_overlap_metrics(extract_numbers(source), extract_numbers(pred))
    date_stats = set_overlap_metrics(extract_dates(source), extract_dates(pred))

    return {
        'entity_coverage': entity_stats['recall'],
        'number_coverage': number_stats['recall'],
        'date_coverage': date_stats['recall'],
        'entity_precision': entity_stats['precision'],
        'entity_f1': entity_stats['f1'],
        'entity_unsupported_ratio': entity_stats['unsupported_ratio'],
        'number_precision': number_stats['precision'],
        'number_f1': number_stats['f1'],
        'number_unsupported_ratio': number_stats['unsupported_ratio'],
        'date_precision': date_stats['precision'],
        'date_f1': date_stats['f1'],
        'date_unsupported_ratio': date_stats['unsupported_ratio'],
    }


def novelty_metrics(source: str, pred: str) -> Dict[str, float]:
    source_tokens = set(tokenize(source))
    pred_tokens = tokenize(pred)
    if not pred_tokens:
        return {
            'novel_token_ratio': 0.0,
            'copy_token_ratio': 0.0,
            'novel_content_token_ratio': 0.0,
        }

    novel_tokens = [tok for tok in pred_tokens if tok not in source_tokens]
    pred_content_tokens = [tok for tok in pred_tokens if tok not in STOPWORDS and not tok.isdigit()]
    novel_content_tokens = [tok for tok in pred_content_tokens if tok not in source_tokens]

    return {
        'novel_token_ratio': len(novel_tokens) / len(pred_tokens),
        'copy_token_ratio': 1.0 - (len(novel_tokens) / len(pred_tokens)),
        'novel_content_token_ratio': (
            len(novel_content_tokens) / len(pred_content_tokens) if pred_content_tokens else 0.0
        ),
    }


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
