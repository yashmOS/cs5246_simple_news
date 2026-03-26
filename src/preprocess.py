from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# TODO: using simple hardcoding for stopwords, months, and simple_words for now. Need to refactor this.

STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he','in','is','it','its','of','on','that','the','to','was','were','will','with',
    'about','after','against','all','also','among','any','because','been','before','being','but','can','could','did','do','does','had','have','her',
    'his','if','into','more','most','not','or','other','our','out','over','she','than','their','them','they','this','those','under','up','we','you'
}

MONTHS = {'january','february','march','april','may','june','july','august','september','october','november','december',
          'jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec'}

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
    'assistance': ('help', 'assistance = help'),
    'children': ('kids', 'children = kids'),
    'located': ('found', 'located = found in a place'),
    'requested': ('asked', 'requested = asked for'),
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:,\d+)*(?:\.\d+)?", text.lower())


def token_frequency_rank(texts: List[str]) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    ordered = [w for w, _ in counter.most_common()]
    return {w: i + 1 for i, w in enumerate(ordered)}


def extract_numbers(text: str) -> List[str]:
    return re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)


def extract_dates(text: str) -> List[str]:
    hits = []
    for m in re.finditer(r'\b(?:' + '|'.join(MONTHS) + r')\b(?:\s+\d{1,2})?(?:,?\s+\d{4})?', text, flags=re.I):
        hits.append(m.group(0))
    hits.extend(re.findall(r'\b\d{4}\b', text))
    return list(dict.fromkeys(hits))


def extract_entities(text: str) -> List[str]:
    entities = []
    pattern = re.compile(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b')
    for match in pattern.finditer(text):
        phrase = match.group(0).strip()
        if phrase.lower() not in MONTHS and len(phrase) > 1:
            entities.append(phrase)
    return entities


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
    entity_counter = Counter([e for sf in sent_feats for e in sf['entities']])
    title_entities = extract_entities(title)
    return {
        'sentences': sentences,
        'sentence_features': sent_feats,
        'entities': entity_counter,
        'numbers': extract_numbers(article),
        'dates': extract_dates(article),
        'title_entities': title_entities,
    }
