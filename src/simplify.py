from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .preprocess import SIMPLE_WORDS, extract_dates, extract_entities, extract_numbers, tokenize


def protected_spans(text: str) -> set:
    protected = set()
    for item in extract_entities(text) + extract_numbers(text) + extract_dates(text):
        for token in tokenize(item):
            protected.add(token.lower())
    return protected


def conservative_lexical_simplify(
    text: str,
    frequency_rank: Dict[str, int],
    max_word_length: int = 9,
    min_word_frequency_rank: int = 3000,
    max_replacements_per_doc: int = 8,
    protected_words: List[str] | None = None,
    use_glossary_fallback: bool = True,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    protected_words = set(w.lower() for w in (protected_words or []))
    protected_words.update(protected_spans(text))
    replacements = []
    glossary = []
    used = 0

    def repl(match: re.Match) -> str:
        nonlocal used
        word = match.group(0)
        lower = word.lower()
        if lower in protected_words:
            return word
        if used >= max_replacements_per_doc:
            return word
        entry = SIMPLE_WORDS.get(lower)
        if entry is None:
            return word
        rank = frequency_rank.get(lower, 999999)
        if len(lower) < max_word_length and rank < min_word_frequency_rank:
            return word
        simple, gloss = entry
        used += 1
        replacements.append({'original': word, 'replacement': simple})
        glossary.append(gloss)
        return simple.capitalize() if word[:1].isupper() else simple

    simplified = re.sub(r'\b[A-Za-z][A-Za-z\-]+\b', repl, text)
    glossary = list(dict.fromkeys(glossary)) if use_glossary_fallback else []
    return simplified, replacements, glossary
