from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

# Lightweight preprocessing utilities.
# These stay dependency-free so the notebook can run without spaCy or NLTK.

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of',
    'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'about', 'after', 'against', 'all', 'also', 'among',
    'any', 'because', 'been', 'before', 'being', 'but', 'can', 'could', 'did', 'do', 'does', 'had', 'have', 'her',
    'his', 'if', 'into', 'more', 'most', 'not', 'or', 'other', 'our', 'out', 'over', 'she', 'than', 'their', 'them',
    'they', 'this', 'those', 'under', 'up', 'we', 'you'
}

MONTHS = {
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
    'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
}
MONTH_PATTERN = r'(?:' + '|'.join(sorted(MONTHS, key=len, reverse=True)) + r')'

DAYS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
TEMPORAL_WORDS = MONTHS | DAYS | {'today', 'yesterday', 'tomorrow', 'tonight'}

SIMPLE_WORDS = {
    'approximately': ('about', 'approximately = about'),
    'purchase': ('buy', 'purchase = buy'),
    'residents': ('people', 'residents = people who live there'),
    'assistance': ('help', 'assistance = help'),
    'demonstrate': ('show', 'demonstrate = show clearly'),
    'additional': ('more', 'additional = more'),
    'numerous': ('many', 'numerous = many'),
    'commence': ('start', 'commence = start'),
    'terminate': ('end', 'terminate = end'),
    'individuals': ('people', 'individuals = people'),
    'requirement': ('need', 'requirement = something needed'),
    'investigation': ('inquiry', 'investigation = careful official check'),
    'announced': ('said', 'announced = said publicly'),
    'attempted': ('tried', 'attempted = tried'),
    'significant': ('big', 'significant = important or large'),
    'sufficient': ('enough', 'sufficient = enough'),
    'children': ('kids', 'children = kids'),
    'located': ('found', 'located = found in a place'),
    'requested': ('asked', 'requested = asked for'),
}

TOKEN_PATTERN = re.compile(r"(?:[A-Za-z]\.){2,}|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:,\d{3})*(?:\.\d+)?%?")
NUMBER_PATTERN = re.compile(r'(?<!\w)(?:[$€£]\s*)?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)')
CASE_TOKEN_PATTERN = re.compile(r"(?:[A-Z]\.){2,}|[A-Z]{2,}|[A-Z][A-Za-z'’\-]*|[a-z]+(?:[-'][a-z]+)*|\d+|&")

ABBREVIATIONS = {
    'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'St.', 'vs.', 'etc.', 'e.g.', 'i.e.',
    'U.S.', 'U.K.', 'No.', 'Inc.', 'Ltd.', 'Co.', 'Corp.', 'Jan.', 'Feb.', 'Mar.', 'Apr.',
    'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'
}

LEADING_ENTITY_NOISE = {'On', 'In', 'At', 'By', 'From', 'After', 'Before', 'During', 'As', 'When'}
ROLE_TOKENS = {
    'president', 'prime', 'minister', 'ceo', 'chief', 'secretary', 'judge', 'justice', 'senator',
    'governor', 'gov', 'director', 'commissioner', 'spokesperson', 'chair', 'dr', 'mr', 'mrs', 'ms',
    'prof', 'professor'
}
ENTITY_CONNECTORS = {'of', 'the', 'and', 'for', 'to', '&', 'de', 'la', 'da'}
SINGLETON_ENTITY_BLOCKLIST = {
    'The', 'A', 'An', 'He', 'She', 'They', 'It', 'This', 'That', 'These', 'Those', 'Today', 'Yesterday',
    'Tomorrow'
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        key = re.sub(r'\s+', ' ', item.strip()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output


def _protect_sentence_breakers(text: str) -> str:
    """Protect common period uses so sentence splitting is less brittle."""
    safe = text
    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        safe = safe.replace(abbr, abbr.replace('.', '<prd>'))
    safe = re.sub(r'\b(?:[A-Z]\.){2,}', lambda m: m.group(0).replace('.', '<prd>'), safe)
    safe = re.sub(r'(?<=\d)\.(?=\d)', '<dec>', safe)
    return safe


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    safe = _protect_sentence_breakers(text)
    safe = re.sub(
        r'([.!?]["\')\]]*)\s+(?=(?:["\'(\[]?[A-Z0-9]))',
        r'\1<split>',
        safe,
    )

    sentences = []
    for part in safe.split('<split>'):
        sent = part.replace('<prd>', '.').replace('<dec>', '.').strip()
        if sent:
            sentences.append(sent)
    return sentences


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(text or '')
    return [tok.lower().replace('.', '') for tok in tokens]


def token_frequency_rank(texts: List[str]) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    ordered = [w for w, _ in counter.most_common()]
    return {w: i + 1 for i, w in enumerate(ordered)}


def extract_numbers(text: str) -> List[str]:
    return _dedupe_preserve_order(m.group(0) for m in NUMBER_PATTERN.finditer(text or ''))


def _collect_non_overlapping_matches(text: str, patterns: List[re.Pattern[str]]) -> List[str]:
    matches: List[tuple[int, int, str]] = []
    occupied: List[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        start, end = span
        return any(not (end <= prev_start or start >= prev_end) for prev_start, prev_end in occupied)

    for pattern in patterns:
        for match in pattern.finditer(text):
            span = match.span()
            if overlaps(span):
                continue
            occupied.append(span)
            matches.append((span[0], span[1], match.group(0).strip()))

    matches.sort(key=lambda item: (item[0], item[1]))
    return _dedupe_preserve_order(item[2] for item in matches)


def extract_dates(text: str) -> List[str]:
    text = text or ''
    patterns = [
        re.compile(rf'\b{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}\b', flags=re.I),
        re.compile(rf'\b{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?\b', flags=re.I),
        re.compile(rf'\b\d{{1,2}}\s+{MONTH_PATTERN}(?:\s+\d{{4}})?\b', flags=re.I),
        re.compile(rf'\b{MONTH_PATTERN}\s+\d{{4}}\b', flags=re.I),
        re.compile(r'\b\d{4}-\d{1,2}-\d{1,2}\b'),
        re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    ]
    hits = _collect_non_overlapping_matches(text, patterns)

    year_pattern = re.compile(r'\b(?:19|20)\d{2}\b')
    existing_spans = []
    for date in hits:
        for match in re.finditer(re.escape(date), text):
            existing_spans.append(match.span())

    for match in year_pattern.finditer(text):
        span = match.span()
        if any(not (span[1] <= prev_start or span[0] >= prev_end) for prev_start, prev_end in existing_spans):
            continue
        hits.append(match.group(0))

    return _dedupe_preserve_order(hits)


def _looks_like_proper_token(token: str) -> bool:
    return bool(re.fullmatch(r"(?:[A-Z]\.){2,}|[A-Z]{2,}|[A-Z][A-Za-z'’\-]*", token))


def _trim_entity_tokens(tokens: List[str]) -> List[str]:
    trimmed = list(tokens)

    while len(trimmed) > 1 and (trimmed[0] in LEADING_ENTITY_NOISE or trimmed[0].lower() in TEMPORAL_WORDS):
        trimmed = trimmed[1:]

    while trimmed and trimmed[-1].lower() in ENTITY_CONNECTORS:
        trimmed = trimmed[:-1]

    return trimmed


def _valid_entity_tokens(tokens: List[str]) -> bool:
    if not tokens:
        return False

    if all(tok.lower().rstrip('.') in ROLE_TOKENS for tok in tokens):
        return False

    if len(tokens) == 1:
        token = tokens[0].rstrip('.')
        if token in SINGLETON_ENTITY_BLOCKLIST:
            return False
        if token.lower() in TEMPORAL_WORDS:
            return False
        if token.lower() in ROLE_TOKENS:
            return False
        if len(token) <= 1:
            return False

    return True


def _split_role_mediated_span(tokens: List[str]) -> List[List[str]]:
    # Example: "Apple CEO Tim Cook" -> ["Apple"], ["Tim", "Cook"]
    role_idx = None
    for idx in range(1, len(tokens) - 1):
        if tokens[idx].lower().rstrip('.') in ROLE_TOKENS:
            role_idx = idx
            break

    if role_idx is None:
        return [tokens]

    left = tokens[:role_idx]
    right_start = role_idx
    while right_start < len(tokens) and tokens[right_start].lower().rstrip('.') in ROLE_TOKENS:
        right_start += 1
    right = tokens[right_start:]

    pieces: List[List[str]] = []
    if left:
        pieces.append(left)
    if right:
        pieces.append(right)
    return pieces or [tokens]


def extract_entities(text: str) -> List[str]:
    entities: List[str] = []

    for sentence in split_sentences(text):
        tokens = CASE_TOKEN_PATTERN.findall(sentence)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if not _looks_like_proper_token(token):
                i += 1
                continue

            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if _looks_like_proper_token(nxt):
                    j += 1
                    continue
                if nxt.lower() in ENTITY_CONNECTORS and j + 1 < len(tokens) and _looks_like_proper_token(tokens[j + 1]):
                    j += 2
                    continue
                break

            candidate = _trim_entity_tokens(tokens[i:j])
            for piece in _split_role_mediated_span(candidate):
                piece = _trim_entity_tokens(piece)
                if _valid_entity_tokens(piece):
                    entities.append(' '.join(piece))
            i = max(j, i + 1)

    return _dedupe_preserve_order(entities)


def sentence_features(sentence: str) -> Dict[str, object]:
    tokens = tokenize(sentence)
    content = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    return {
        'sentence': sentence,
        'tokens': tokens,
        'content_tokens': content,
        'entities': extract_entities(sentence),
        'numbers': extract_numbers(sentence),
        'dates': extract_dates(sentence),
        'length_tokens': len(tokens),
    }


def article_features(article: str, title: str = '') -> Dict[str, object]:
    sentences = split_sentences(article)
    sent_feats = [sentence_features(s) for s in sentences]
    entity_counter = Counter(entity for sf in sent_feats for entity in sf['entities'])
    title_entities = extract_entities(title)
    return {
        'sentences': sentences,
        'sentence_features': sent_feats,
        'entities': entity_counter,
        'numbers': extract_numbers(article),
        'dates': extract_dates(article),
        'title_entities': title_entities,
    }
