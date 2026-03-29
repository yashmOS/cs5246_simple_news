from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .preprocess import SIMPLE_WORDS, extract_dates, extract_entities, extract_numbers, tokenize


def protected_spans(text: str) -> set[str]:
    protected = set()
    for item in extract_entities(text) + extract_numbers(text) + extract_dates(text):
        for token in tokenize(item):
            protected.add(token.lower())
    return protected


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _is_hard_word(
    word: str,
    frequency_rank: Dict[str, int],
    max_word_length: int,
    min_word_frequency_rank: int,
) -> bool:
    rank = frequency_rank.get(word, 999999)
    return len(word) >= max_word_length or rank >= min_word_frequency_rank


def conservative_lexical_simplify(
    text: str,
    frequency_rank: Dict[str, int],
    max_word_length: int = 9,
    min_word_frequency_rank: int = 3000,
    max_replacements_per_doc: int = 8,
    protected_words: List[str] | None = None,
    use_glossary_fallback: bool = True,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    protected_words_set = {word.lower() for word in (protected_words or [])}
    protected_words_set.update(protected_spans(text))

    replacements: List[Dict[str, str]] = []
    glossary: List[str] = []
    replaced_words: set[str] = set()
    used = 0

    def add_glossary_if_helpful(lower: str) -> None:
        if not use_glossary_fallback or lower in replaced_words:
            return
        entry = SIMPLE_WORDS.get(lower)
        if entry is None:
            return
        if not _is_hard_word(lower, frequency_rank, max_word_length, min_word_frequency_rank):
            return
        glossary.append(entry[1])

    def repl(match: re.Match[str]) -> str:
        nonlocal used

        word = match.group(0)
        lower = word.lower()

        if lower in protected_words_set or word.isupper():
            add_glossary_if_helpful(lower)
            return word

        entry = SIMPLE_WORDS.get(lower)
        if entry is None:
            return word

        if not _is_hard_word(lower, frequency_rank, max_word_length, min_word_frequency_rank):
            return word

        if used >= max_replacements_per_doc:
            add_glossary_if_helpful(lower)
            return word

        simple, gloss = entry
        if simple.lower() == lower:
            return word

        used += 1
        replaced_words.add(lower)
        replacements.append({'original': word, 'replacement': simple})
        glossary.append(gloss)
        return _preserve_case(word, simple)

    simplified = re.sub(r'\b[A-Za-z][A-Za-z\-]+\b', repl, text)

    if use_glossary_fallback:
        for token in tokenize(text):
            lower = token.lower()
            if lower in replaced_words or lower in protected_words_set:
                continue
            add_glossary_if_helpful(lower)
            if len(dict.fromkeys(glossary)) >= 6:
                break

    glossary = list(dict.fromkeys(glossary)) if use_glossary_fallback else []
    return simplified, replacements, glossary[:6]
