from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import article_features


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if abs(high - low) < 1e-12:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _fallback_scores(sentences: List[str]) -> List[float]:
    # Light lead bias is a better fallback than crashing or returning zeros.
    raw = [1.0 / (idx + 1) for idx in range(len(sentences))]
    total = sum(raw) or 1.0
    return [value / total for value in raw]


def _similarity_matrix(sentences: List[str]) -> np.ndarray:
    if len(sentences) <= 1:
        return np.zeros((len(sentences), len(sentences)))

    try:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        matrix = vect.fit_transform(sentences)
        if matrix.shape[1] == 0:
            return np.zeros((len(sentences), len(sentences)))
        sim = cosine_similarity(matrix)
        np.fill_diagonal(sim, 0.0)
        return sim
    except ValueError:
        return np.zeros((len(sentences), len(sentences)))


def lead3(article: str, max_sentences: int = 3) -> str:
    feats = article_features(article)
    return ' '.join(feats['sentences'][:max(0, max_sentences)]).strip()


def textrank_scores(sentences: List[str], similarity_threshold: float = 0.05) -> List[float]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]

    sim = _similarity_matrix(sentences)
    if sim.size == 0:
        return []

    sim = sim.copy()
    sim[sim < max(0.0, similarity_threshold)] = 0.0
    if not np.any(sim):
        return _fallback_scores(sentences)

    graph = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank(graph, weight='weight')
    except Exception:
        return _fallback_scores(sentences)

    raw = [float(scores.get(i, 0.0)) for i in range(len(sentences))]
    total = sum(raw)
    if total <= 0:
        return _fallback_scores(sentences)
    return [value / total for value in raw]


def textrank_select(article: str, max_sentences: int = 3, similarity_threshold: float = 0.05) -> str:
    feats = article_features(article)
    sentences = feats['sentences']
    scores = textrank_scores(sentences, similarity_threshold=similarity_threshold)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max(0, max_sentences)]
    ranked = sorted(ranked)
    return ' '.join(sentences[i] for i in ranked).strip()


def coverage_aware_select(
    article: str,
    title: str = '',
    max_sentences: int = 3,
    top_entity_k: int = 6,
    similarity_threshold: float = 0.05,
    require_number_coverage: bool = True,
    require_date_coverage: bool = True,
) -> str:
    feats = article_features(article, title=title)
    sentences = feats['sentences']
    sent_feats = feats['sentence_features']
    if not sentences or max_sentences <= 0:
        return ''

    base_scores = _normalize_scores(textrank_scores(sentences, similarity_threshold=similarity_threshold))
    sim_matrix = _similarity_matrix(sentences)

    target_entities = [entity for entity, _ in feats['entities'].most_common(max(0, top_entity_k))]
    for entity in feats['title_entities']:
        if entity not in target_entities:
            target_entities.append(entity)

    target_entity_set = set(target_entities)
    title_entity_set = set(feats['title_entities'])
    target_number_set = set(dict.fromkeys(feats['numbers'])) if require_number_coverage else set()
    target_date_set = set(dict.fromkeys(feats['dates'])) if require_date_coverage else set()

    chosen: List[int] = []
    covered_entities, covered_numbers, covered_dates = set(), set(), set()

    limit = min(max_sentences, len(sentences))
    while len(chosen) < limit:
        best_idx = None
        best_score = float('-inf')

        for idx in range(len(sentences)):
            if idx in chosen:
                continue

            sf = sent_feats[idx]
            sentence_entities = set(sf['entities'])
            sentence_numbers = set(sf['numbers'])
            sentence_dates = set(sf['dates'])

            new_entities = (sentence_entities & target_entity_set) - covered_entities
            new_numbers = (sentence_numbers & target_number_set) - covered_numbers
            new_dates = (sentence_dates & target_date_set) - covered_dates
            title_hits = sentence_entities & title_entity_set

            entity_gain = len(new_entities) / max(1, len(target_entity_set)) if target_entity_set else 0.0
            number_gain = len(new_numbers) / max(1, len(target_number_set)) if target_number_set else 0.0
            date_gain = len(new_dates) / max(1, len(target_date_set)) if target_date_set else 0.0
            title_gain = len(title_hits) / max(1, len(title_entity_set)) if title_entity_set else 0.0

            coverage_gain = entity_gain + 0.75 * number_gain + 0.75 * date_gain + 0.50 * title_gain
            redundancy_penalty = max((sim_matrix[idx, chosen_idx] for chosen_idx in chosen), default=0.0)
            lead_bonus = 1.0 / (idx + 1)

            candidate_score = (
                base_scores[idx]
                + coverage_gain
                + 0.05 * lead_bonus
                - 0.35 * redundancy_penalty
            )

            if candidate_score > best_score or (
                abs(candidate_score - best_score) < 1e-12 and (best_idx is None or idx < best_idx)
            ):
                best_score = candidate_score
                best_idx = idx

        if best_idx is None:
            break

        chosen.append(best_idx)
        covered_entities.update(set(sent_feats[best_idx]['entities']) & target_entity_set)
        covered_numbers.update(set(sent_feats[best_idx]['numbers']) & target_number_set)
        covered_dates.update(set(sent_feats[best_idx]['dates']) & target_date_set)

    chosen = sorted(chosen)
    return ' '.join(sentences[idx] for idx in chosen).strip()
