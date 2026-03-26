from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import article_features


def lead3(article: str, max_sentences: int = 3) -> str:
    feats = article_features(article)
    return ' '.join(feats['sentences'][:max_sentences]).strip()


def textrank_scores(sentences: List[str]) -> List[float]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(sentences)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, weight='weight')
    return [scores[i] for i in range(len(sentences))]


def textrank_select(article: str, max_sentences: int = 3) -> str:
    feats = article_features(article)
    sentences = feats['sentences']
    scores = textrank_scores(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    ranked = sorted(ranked)
    return ' '.join(sentences[i] for i in ranked).strip()


def coverage_aware_select(article: str, title: str = '', max_sentences: int = 3, top_entity_k: int = 6) -> str:
    feats = article_features(article, title=title)
    sentences = feats['sentences']
    sent_feats = feats['sentence_features']
    if not sentences:
        return ''
    scores = textrank_scores(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    target_entities = [e for e, _ in feats['entities'].most_common(top_entity_k)]
    target_entities.extend([e for e in feats['title_entities'] if e not in target_entities])
    target_numbers = list(dict.fromkeys(feats['numbers']))
    target_dates = list(dict.fromkeys(feats['dates']))

    chosen = []
    covered_entities, covered_numbers, covered_dates = set(), set(), set()

    for idx in ranked:
        if len(chosen) >= max_sentences:
            break
        chosen.append(idx)
        covered_entities.update(sent_feats[idx]['entities'])
        covered_numbers.update(sent_feats[idx]['numbers'])
        covered_dates.update(sent_feats[idx]['dates'])

    unmet_entities = [e for e in target_entities if e not in covered_entities]
    unmet_numbers = [n for n in target_numbers if n not in covered_numbers]
    unmet_dates = [d for d in target_dates if d not in covered_dates]

    for idx in ranked:
        if idx in chosen:
            continue
        gain = 0
        sf = sent_feats[idx]
        gain += sum(1 for e in sf['entities'] if e in unmet_entities)
        gain += sum(1 for n in sf['numbers'] if n in unmet_numbers)
        gain += sum(1 for d in sf['dates'] if d in unmet_dates)
        if gain <= 0:
            continue
        if len(chosen) < max_sentences + 1:
            chosen.append(idx)
            covered_entities.update(sf['entities'])
            covered_numbers.update(sf['numbers'])
            covered_dates.update(sf['dates'])
            unmet_entities = [e for e in target_entities if e not in covered_entities]
            unmet_numbers = [n for n in target_numbers if n not in covered_numbers]
            unmet_dates = [d for d in target_dates if d not in covered_dates]
        if not unmet_entities and not unmet_numbers and not unmet_dates:
            break

    chosen = sorted(set(chosen))
    return ' '.join(sentences[i] for i in chosen).strip()
