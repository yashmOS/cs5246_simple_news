from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .lexicon import LEXICON, PHRASE_REPLACEMENTS
from .preprocess import SIMPLE_WORDS, extract_dates, extract_entities, extract_numbers, split_sentences, tokenize, normalize_whitespace

# Stronger but still conservative simplification:
# 1) optional sentence-level cleanup / splitting
# 2) contextual lexical simplification with POS/context/semantic gating
# 3) glossary fallback when replacement confidence is low

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]+\b")

DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its",
    "our", "their", "each", "every", "some", "any", "many", "few", "several", "another",
}
PREPOSITIONS = {
    "in", "on", "at", "by", "from", "to", "for", "with", "about", "after", "before",
    "during", "under", "over", "into", "onto", "across", "through", "between", "among", "of",
}
COPULARS = {"is", "are", "was", "were", "be", "been", "being", "seem", "seems", "became", "become"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they", "someone", "something", "people", "officials",
    "residents", "children", "government", "minister", "president", "police", "court", "judge",
    "company", "authorities",
}
NEGATORS = {"not", "never", "no"}
CONTEXT_STOP = {"and", "or", "but", "if", "than", "because", "while"}

NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ship", "ance", "ence", "ism", "ist", "er", "or", "ure", "al"
)
VERB_SUFFIXES = ("ate", "ify", "ize", "ise", "ing", "ed")
ADJ_SUFFIXES = ("ous", "ful", "ive", "less", "able", "ible", "ical", "ic", "ary", "ory", "ent", "ant")
ADV_SUFFIXES = ("ly",)

LEXICON_META: Dict[str, Dict[str, object]] = {
    "approximately": {"pos": "ADV", "confidence": 0.98},
    "purchase": {"pos": "VERB", "confidence": 0.95},
    "purchased": {"pos": "VERB", "confidence": 0.95},
    "residents": {"pos": "NOUN", "confidence": 0.97},
    "assistance": {"pos": "NOUN", "confidence": 0.96},
    "demonstrate": {"pos": "VERB", "confidence": 0.93},
    "demonstrated": {"pos": "VERB", "confidence": 0.93},
    "additional": {"pos": "ADJ", "confidence": 0.95},
    "numerous": {"pos": "ADJ", "confidence": 0.96},
    "commence": {"pos": "VERB", "confidence": 0.96},
    "terminate": {"pos": "VERB", "confidence": 0.95},
    "individuals": {"pos": "NOUN", "confidence": 0.98},
    "requirement": {"pos": "NOUN", "confidence": 0.94},
    "requirements": {"pos": "NOUN", "confidence": 0.94},
    "investigation": {"pos": "NOUN", "confidence": 0.90},
    "investigations": {"pos": "NOUN", "confidence": 0.90},
    "announced": {"pos": "VERB", "confidence": 0.88},
    "attempted": {"pos": "VERB", "confidence": 0.92},
    "significant": {"pos": "ADJ", "confidence": 0.90},
    "significantly": {"pos": "ADV", "confidence": 0.88},
    "sufficient": {"pos": "ADJ", "confidence": 0.93},
    "children": {"pos": "NOUN", "confidence": 0.98},
    "located": {"pos": "VERB", "confidence": 0.82},
    "requested": {"pos": "VERB", "confidence": 0.90},
    "authorities": {"pos": "NOUN", "confidence": 0.93},
    "department": {"pos": "NOUN", "confidence": 0.90},
    "officials": {"pos": "NOUN", "confidence": 0.92},
    "military": {"pos": "NOUN", "confidence": 0.86},
    "minister": {"pos": "NOUN", "confidence": 0.83},
    "congress": {"pos": "NOUN", "confidence": 0.88},
    "legislation": {"pos": "NOUN", "confidence": 0.96},
    "economic": {"pos": "ADJ", "confidence": 0.82},
    "economy": {"pos": "NOUN", "confidence": 0.78},
    "recession": {"pos": "NOUN", "confidence": 0.84},
    "inflation": {"pos": "NOUN", "confidence": 0.84},
    "statement": {"pos": "NOUN", "confidence": 0.85},
    "facility": {"pos": "NOUN", "confidence": 0.86},
    "currently": {"pos": "ADV", "confidence": 0.97},
    "previously": {"pos": "ADV", "confidence": 0.97},
    "subsequently": {"pos": "ADV", "confidence": 0.94},
    "therefore": {"pos": "ADV", "confidence": 0.95},
    "however": {"pos": "ADV", "confidence": 0.96},
    "nevertheless": {"pos": "ADV", "confidence": 0.91},
    "furthermore": {"pos": "ADV", "confidence": 0.91},
    "resigned": {"pos": "VERB", "confidence": 0.93},
    "acquired": {"pos": "VERB", "confidence": 0.82},
    "committed": {"pos": "VERB", "confidence": 0.78},
    "suspect": {"pos": "NOUN", "confidence": 0.76},
    "suspects": {"pos": "NOUN", "confidence": 0.76},
    "alleged": {"pos": "ADJ", "confidence": 0.72},
    "allegedly": {"pos": "ADV", "confidence": 0.72},
    "injured": {"pos": "ADJ", "confidence": 0.90},
    "fatal": {"pos": "ADJ", "confidence": 0.90},
    "deceased": {"pos": "ADJ", "confidence": 0.88},
    "vehicle": {"pos": "NOUN", "confidence": 0.88},
    "vehicles": {"pos": "NOUN", "confidence": 0.88},
    "residence": {"pos": "NOUN", "confidence": 0.90},
    "residential": {"pos": "ADJ", "confidence": 0.82},
    "occurred": {"pos": "VERB", "confidence": 0.95},
    "substantial": {"pos": "ADJ", "confidence": 0.82},
    "substantially": {"pos": "ADV", "confidence": 0.82},
    "maintain": {"pos": "VERB", "confidence": 0.84},
    "maintained": {"pos": "VERB", "confidence": 0.84},
    "obtain": {"pos": "VERB", "confidence": 0.90},
    "obtained": {"pos": "VERB", "confidence": 0.90},
    "utilize": {"pos": "VERB", "confidence": 0.97},
    "utilized": {"pos": "VERB", "confidence": 0.97},
    "commencement": {"pos": "NOUN", "confidence": 0.92},
    "termination": {"pos": "NOUN", "confidence": 0.92},
    "consequently": {"pos": "ADV", "confidence": 0.90},
    "endeavor": {"pos": "VERB", "confidence": 0.86},
    "endeavored": {"pos": "VERB", "confidence": 0.86},
    "sustain": {"pos": "VERB", "confidence": 0.78},
    "sustained": {"pos": "VERB", "confidence": 0.78},
    "encountered": {"pos": "VERB", "confidence": 0.80},
    "commuters": {"pos": "NOUN", "confidence": 0.76},
    "evacuated": {"pos": "VERB", "confidence": 0.88},
    "evacuation": {"pos": "NOUN", "confidence": 0.88},
    "detained": {"pos": "VERB", "confidence": 0.86},
    "detention": {"pos": "NOUN", "confidence": 0.84},
}

PAREN_RE = re.compile(r"\([^)]{1,80}\)")
APPOSITIVE_RE = re.compile(r",\s*(?:a|an|the)\s+[^,]{1,45},")
RELCLAUSE_RE = re.compile(r",\s*(?:who|which)\s+[^,]{1,50},")
REPORTING_TAIL_RE = re.compile(r",\s*(?:officials|police|the company|the ministry|the statement)\s+(?:said|says|added)\.?$", flags=re.I)


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


def _is_hard_word(word: str, frequency_rank: Dict[str, int], max_word_length: int, min_word_frequency_rank: int) -> bool:
    rank = frequency_rank.get(word, 999999)
    return len(word) >= max_word_length or rank >= min_word_frequency_rank


def _suffix_pos(word: str) -> Optional[str]:
    lower = word.lower()
    if lower.endswith(ADV_SUFFIXES):
        return "ADV"
    if lower.endswith(VERB_SUFFIXES):
        return "VERB"
    if lower.endswith(ADJ_SUFFIXES):
        return "ADJ"
    if lower.endswith(NOUN_SUFFIXES):
        return "NOUN"
    return None


def _looks_like_plural_noun(word: str) -> bool:
    lower = word.lower()
    return len(lower) > 3 and lower.endswith("s") and not lower.endswith(("ous", "ss"))


def _guess_pos(tokens: List[str], idx: int) -> str:
    word = tokens[idx]
    lower = word.lower()
    prev1 = tokens[idx - 1].lower() if idx - 1 >= 0 else ""
    prev2 = tokens[idx - 2].lower() if idx - 2 >= 0 else ""
    next1 = tokens[idx + 1].lower() if idx + 1 < len(tokens) else ""

    meta_pos = str(LEXICON_META.get(lower, {}).get("pos", "")) or None
    suffix_pos = _suffix_pos(lower)

    if lower in NEGATORS:
        return "ADV"
    if prev1 in MODALS or prev1 == "to":
        return "VERB"
    if prev2 in MODALS and prev1 == "not":
        return "VERB"
    if prev1 in DETERMINERS:
        if suffix_pos == "ADJ":
            return "ADJ"
        if next1 and next1 not in CONTEXT_STOP and next1 not in PREPOSITIONS:
            return "NOUN" if suffix_pos != "ADJ" else "ADJ"
    if prev1 in COPULARS:
        if suffix_pos in {"ADJ", "ADV"}:
            return suffix_pos
        return meta_pos or "ADJ"
    if next1 in {"that", "which", "who"}:
        return "NOUN"
    if prev1 in PREPOSITIONS:
        if suffix_pos == "VERB" and next1 not in {"to", ""}:
            return "VERB"
        return suffix_pos or meta_pos or "NOUN"
    if prev1 in PRONOUNS:
        if suffix_pos == "VERB":
            return "VERB"
        if suffix_pos == "ADV":
            return "ADV"
    if suffix_pos:
        return suffix_pos
    if meta_pos:
        return meta_pos
    if _looks_like_plural_noun(lower):
        return "NOUN"
    return "UNK"


def _candidate_pos(word: str, candidate: str) -> str:
    meta_pos = str(LEXICON_META.get(word.lower(), {}).get("pos", "")) or None
    if meta_pos:
        return meta_pos
    return _suffix_pos(candidate.lower()) or _suffix_pos(word.lower()) or "UNK"


def _slot_accepts_pos(pos: str, tokens: List[str], idx: int) -> bool:
    prev1 = tokens[idx - 1].lower() if idx - 1 >= 0 else ""
    prev2 = tokens[idx - 2].lower() if idx - 2 >= 0 else ""
    next1 = tokens[idx + 1].lower() if idx + 1 < len(tokens) else ""

    if pos == "VERB":
        return prev1 in MODALS or prev1 == "to" or prev1 in PRONOUNS or prev2 in MODALS or prev1 in NEGATORS or next1 in {"to", "that", "up", "down", ""}
    if pos == "NOUN":
        return prev1 in DETERMINERS or prev1 in PREPOSITIONS or next1 in {"is","are","was","were","of","and","or","","who","that","which"} or next1.endswith("s")
    if pos == "ADJ":
        return prev1 in COPULARS or prev1 in {"very","more","most","too"} or (next1 not in PREPOSITIONS and next1 not in CONTEXT_STOP and next1 != "")
    if pos == "ADV":
        return next1 not in DETERMINERS or prev1 in COPULARS or prev1 in MODALS or prev1 == "to"
    return True


@lru_cache(maxsize=4096)
def _wordnet_similarity(word: str, candidate: str, pos: str) -> Optional[float]:
    pos_map = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
    wn_pos = pos_map.get(pos)
    if wn_pos is None:
        return None
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return None
    try:
        word_synsets = wn.synsets(word, pos=wn_pos)
        cand_synsets = wn.synsets(candidate, pos=wn_pos)
    except LookupError:
        return None
    if not word_synsets or not cand_synsets:
        return None
    best = 0.0
    for s1 in word_synsets[:3]:
        for s2 in cand_synsets[:3]:
            sim = s1.wup_similarity(s2)
            if sim is None:
                sim = s1.path_similarity(s2)
            if sim is not None and sim > best:
                best = float(sim)
    return best if best > 0 else None


def _manual_semantic_confidence(word: str, candidate: str) -> float:
    meta_conf = float(LEXICON_META.get(word.lower(), {}).get("confidence", 0.82))
    if word.lower() == candidate.lower():
        return 0.0
    len_ratio = len(candidate) / max(1, len(word))
    length_penalty = 1.0 - min(0.15, abs(1.0 - len_ratio) * 0.2)
    return max(0.0, min(1.0, meta_conf * length_penalty))


def _semantic_similarity_confidence(word: str, candidate: str, pos: str) -> float:
    wn_sim = _wordnet_similarity(word.lower(), candidate.lower(), pos)
    manual = _manual_semantic_confidence(word, candidate)
    if wn_sim is None:
        return manual
    return 0.6 * manual + 0.4 * float(wn_sim)


def _context_confidence(tokens: List[str], idx: int, pos: str) -> float:
    return 1.0 if _slot_accepts_pos(pos, tokens, idx) else 0.0


def _difficulty_score(word: str, frequency_rank: Dict[str, int], max_word_length: int, min_word_frequency_rank: int) -> float:
    rank = frequency_rank.get(word.lower(), 999999)
    rarity = min(1.0, math.log10(rank + 10) / 5.0)
    length_score = min(1.0, max(0, len(word) - max_word_length + 1) / 6.0)
    threshold_bonus = 0.2 if rank >= min_word_frequency_rank else 0.0
    return min(1.0, 0.45 * rarity + 0.35 * length_score + threshold_bonus)


def _pluralize_simple(word: str) -> str:
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _past_simple(word: str) -> str:
    if word.endswith("e"):
        return word + "d"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ied"
    return word + "ed"


def _ing_simple(word: str) -> str:
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    return word + "ing"



def _lexicon_entry(word: str) -> Optional[Dict[str, str]]:
    return LEXICON.get(word.lower())


def _candidate_forms_from_lemma(lower: str) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []

    def _append_if_replace(form: str, replacement_form: Optional[str] = None):
        entry = _lexicon_entry(form)
        if entry is None or entry.get("mode") != "replace":
            return
        simple = replacement_form if replacement_form is not None else entry["replacement"]
        variants.append((simple, entry["gloss"]))

    _append_if_replace(lower)
    if lower.endswith("s") and len(lower) > 3:
        singular = lower[:-1]
        entry = _lexicon_entry(singular)
        if entry is not None and entry.get("mode") == "replace":
            variants.append((_pluralize_simple(entry["replacement"]), entry["gloss"]))
    if lower.endswith("ed") and len(lower) > 4:
        for base in {lower[:-2], lower[:-1]}:
            entry = _lexicon_entry(base)
            if entry is not None and entry.get("mode") == "replace":
                variants.append((_past_simple(entry["replacement"]), entry["gloss"]))
    if lower.endswith("ing") and len(lower) > 5:
        for base in {lower[:-3], lower[:-3] + "e"}:
            entry = _lexicon_entry(base)
            if entry is not None and entry.get("mode") == "replace":
                variants.append((_ing_simple(entry["replacement"]), entry["gloss"]))
    seen = set()
    out: List[Tuple[str, str]] = []
    for simple, gloss in variants:
        key = (simple.lower(), gloss)
        if key in seen:
            continue
        seen.add(key)
        out.append((simple, gloss))
    return out


def _candidate_entries_for(word: str) -> List[Tuple[str, str]]:
    return _candidate_forms_from_lemma(word.lower())


def _decide_replacement(
    word: str,
    tokens: List[str],
    idx: int,
    frequency_rank: Dict[str, int],
    max_word_length: int,
    min_word_frequency_rank: int,
    protected_words_set: set[str],
) -> Optional[Dict[str, object]]:
    lower = word.lower()
    if lower in protected_words_set or word.isupper():
        return None
    if not _is_hard_word(lower, frequency_rank, max_word_length, min_word_frequency_rank):
        return None
    candidates = _candidate_entries_for(lower)
    if not candidates:
        return None

    orig_pos = _guess_pos(tokens, idx)
    best: Optional[Dict[str, object]] = None

    for simple, gloss in candidates:
        cand_pos = _candidate_pos(lower, simple)
        pos_ok = orig_pos == "UNK" or cand_pos == "UNK" or cand_pos == orig_pos
        if not pos_ok:
            continue
        if not _slot_accepts_pos(cand_pos, tokens, idx):
            continue

        pos_for_sem = cand_pos if cand_pos != "UNK" else (orig_pos if orig_pos != "UNK" else "NOUN")
        semantic_conf = _semantic_similarity_confidence(lower, simple, pos_for_sem)
        difficulty = _difficulty_score(lower, frequency_rank, max_word_length, min_word_frequency_rank)
        context_conf = _context_confidence(tokens, idx, cand_pos)

        confidence = 0.50 * semantic_conf + 0.25 * context_conf + 0.25 * difficulty
        cand = {
            "original": word,
            "replacement": simple,
            "gloss": gloss,
            "original_pos": orig_pos,
            "candidate_pos": cand_pos,
            "semantic_confidence": round(float(semantic_conf), 4),
            "context_confidence": round(float(context_conf), 4),
            "difficulty_score": round(float(difficulty), 4),
            "confidence": round(float(confidence), 4),
        }
        if best is None or cand["confidence"] > best["confidence"]:
            best = cand
    return best


def _apply_token_replacements(sentence: str, replacement_map: Dict[int, str]) -> str:
    pieces: List[str] = []
    last = 0
    for idx, match in enumerate(WORD_RE.finditer(sentence)):
        start, end = match.span()
        pieces.append(sentence[last:start])
        pieces.append(replacement_map.get(idx, match.group(0)))
        last = end
    pieces.append(sentence[last:])
    return ''.join(pieces)


def _split_long_coordination(sentence: str) -> str:
    if len(tokenize(sentence)) < 22:
        return sentence
    return re.sub(
        r",\s+(and|but|so)\s+(?=(?:[A-Z][a-z]+|he|she|they|it|we|officials|police|the|a|an)\b)",
        r". \1 ",
        sentence,
        flags=re.I,
    )



def apply_phrase_simplifications(text: str) -> str:
    out = text
    for phrase, replacement in PHRASE_REPLACEMENTS.items():
        out = re.sub(rf"\b{re.escape(phrase)}\b", replacement, out, flags=re.I)
    return normalize_whitespace(out)


def structural_sentence_simplify(text: str) -> str:
    simplified_sents: List[str] = []
    for sent in split_sentences(text):
        s = normalize_whitespace(sent)
        s = PAREN_RE.sub("", s)
        s = APPOSITIVE_RE.sub(", ", s)
        s = RELCLAUSE_RE.sub(", ", s)
        s = REPORTING_TAIL_RE.sub("", s)
        s = re.sub(r"\s*;\s*", ". ", s)
        s = _split_long_coordination(s)
        s = re.sub(r"\s+", " ", s).strip(" ,")
        if not s:
            continue
        simplified_sents.extend(split_sentences(s))
    return " ".join(simplified_sents).strip()



def conservative_lexical_simplify(
    text: str,
    frequency_rank: Dict[str, int],
    max_word_length: int = 8,
    min_word_frequency_rank: int = 2000,
    max_replacements_per_doc: int = 10,
    protected_words: List[str] | None = None,
    use_glossary_fallback: bool = True,
    min_replacement_confidence: float = 0.70,
    max_glossary_items: int = 8,
    use_structural_simplification: bool = False,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    working_text = apply_phrase_simplifications(text)
    if use_structural_simplification:
        working_text = structural_sentence_simplify(working_text)

    protected_words_set = {word.lower() for word in (protected_words or [])}
    protected_words_set.update(protected_spans(working_text))

    replacements: List[Dict[str, str]] = []
    glossary_scores: Dict[str, float] = {}
    replaced_words: set[str] = set()
    remaining_budget = max_replacements_per_doc
    new_sentences: List[str] = []

    for sentence in split_sentences(working_text):
        token_matches = list(WORD_RE.finditer(sentence))
        tokens = [m.group(0) for m in token_matches]
        replacement_map: Dict[int, str] = {}

        for idx, token in enumerate(tokens):
            lower = token.lower()
            entry = _lexicon_entry(lower)
            decision = _decide_replacement(
                token,
                tokens,
                idx,
                frequency_rank=frequency_rank,
                max_word_length=max_word_length,
                min_word_frequency_rank=min_word_frequency_rank,
                protected_words_set=protected_words_set,
            )

            if decision is not None and remaining_budget > 0 and float(decision["confidence"]) >= min_replacement_confidence:
                replacement_map[idx] = _preserve_case(token, str(decision["replacement"]))
                replaced_words.add(lower)
                remaining_budget -= 1
                replacements.append({
                    "original": token,
                    "replacement": str(decision["replacement"]),
                    "pos": str(decision["candidate_pos"]),
                    "confidence": f'{float(decision["confidence"]):.3f}',
                    "semantic_confidence": f'{float(decision["semantic_confidence"]):.3f}',
                    "context_confidence": f'{float(decision["context_confidence"]):.3f}',
                })
                glossary_scores[str(decision["gloss"])] = max(
                    glossary_scores.get(str(decision["gloss"]), 0.0),
                    0.85 + float(decision["confidence"]) * 0.10,
                )
                continue

            if not use_glossary_fallback or lower in replaced_words or lower in protected_words_set:
                continue
            if entry is None or not _is_hard_word(lower, frequency_rank, max_word_length, min_word_frequency_rank):
                continue

            difficulty = _difficulty_score(lower, frequency_rank, max_word_length, min_word_frequency_rank)
            mode_bonus = 0.15 if entry.get("mode") == "gloss" else 0.05
            confidence_penalty = 0.0
            if decision is not None:
                confidence_penalty = max(0.0, min_replacement_confidence - float(decision["confidence"]))
            glossary_scores[entry["gloss"]] = max(
                glossary_scores.get(entry["gloss"], 0.0),
                difficulty + mode_bonus - confidence_penalty,
            )

        new_sentences.append(_apply_token_replacements(sentence, replacement_map))

    simplified = normalize_whitespace(" ".join(new_sentences))

    glossary: List[str] = []
    if use_glossary_fallback:
        ranked = sorted(glossary_scores.items(), key=lambda kv: (-kv[1], kv[0].lower()))
        glossary = [item for item, _ in ranked[:max_glossary_items]]

    return simplified, replacements, glossary
