from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .config import ExperimentConfig
from .evaluation import evaluate_output
from .preprocess import token_frequency_rank
from .selectors import coverage_aware_select, lead3, textrank_select
from .simplify import conservative_lexical_simplify


SYSTEM_LABELS = {
    's0_lead3': 'S0 Lead-3',
    's1_textrank': 'S1 TextRank',
    's2_coverage': 'S2 Coverage-aware TextRank',
    's3_simplified': 'S3 Coverage-aware + simplification',
}

SYSTEM_TO_SHORT = {
    's0_lead3': 'S0',
    's1_textrank': 'S1',
    's2_coverage': 'S2',
    's3_simplified': 'S3',
}


def _resolve_systems(config: ExperimentConfig, systems: Optional[Iterable[str]] = None) -> List[str]:
    if systems is None:
        return list(config.systems)
    out = []
    for sys_name in systems:
        if sys_name not in SYSTEM_LABELS:
            raise ValueError(f'Unknown system: {sys_name}')
        out.append(sys_name)
    return out


def _compute_base_outputs(source: str, title: str, config: ExperimentConfig, freq_rank: dict[str, int]):
    s0 = lead3(source, max_sentences=config.selector.max_sentences)
    s1 = textrank_select(
        source,
        max_sentences=config.selector.max_sentences,
        similarity_threshold=config.selector.similarity_threshold,
    )
    s2 = coverage_aware_select(
        source,
        title=title,
        max_sentences=config.selector.max_sentences,
        top_entity_k=config.selector.top_entity_k,
        similarity_threshold=config.selector.similarity_threshold,
        require_number_coverage=config.selector.require_number_coverage,
        require_date_coverage=config.selector.require_date_coverage,
    )

    if config.simplifier.enabled:
        s3_text, replacements, glossary = conservative_lexical_simplify(
            s2,
            frequency_rank=freq_rank,
            max_word_length=config.simplifier.max_word_length,
            min_word_frequency_rank=config.simplifier.min_word_frequency_rank,
            max_replacements_per_doc=config.simplifier.max_replacements_per_doc,
            protected_words=config.simplifier.protected_words,
            use_glossary_fallback=config.simplifier.use_glossary_fallback,
            min_replacement_confidence=config.simplifier.min_replacement_confidence,
            max_glossary_items=config.simplifier.max_glossary_items,
            use_structural_simplification=config.simplifier.use_structural_simplification,
        )
    else:
        s3_text, replacements, glossary = s2, [], []

    return {
        's0_lead3': (s0, [], []),
        's1_textrank': (s1, [], []),
        's2_coverage': (s2, [], []),
        's3_simplified': (s3_text, replacements, glossary),
    }


def generate_system_outputs(
    df: pd.DataFrame,
    config: ExperimentConfig,
    systems: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    systems = _resolve_systems(config, systems)
    articles = df['article'].fillna('').astype(str).tolist()
    freq_rank = token_frequency_rank(articles)

    rows = []
    for row in df.itertuples(index=False):
        source = str(getattr(row, 'article', '') or '')
        reference = str(getattr(row, 'reference', '') or '')
        title = str(getattr(row, 'title', '') or '')
        all_outputs = _compute_base_outputs(source=source, title=title, config=config, freq_rank=freq_rank)

        for sys_name in systems:
            output, reps, gloss = all_outputs[sys_name]
            metrics = evaluate_output(source=source, reference=reference, pred=output)
            rows.append({
                'doc_id': str(getattr(row, 'doc_id', '')),
                'system': sys_name,
                'system_short': SYSTEM_TO_SHORT[sys_name],
                'system_label': SYSTEM_LABELS[sys_name],
                'title': title,
                'article': source,
                'reference': reference,
                'output': output,
                'replacements': reps,
                'glossary': gloss,
                **metrics,
            })

    return pd.DataFrame(rows)


def generate_single_system_output(df: pd.DataFrame, config: ExperimentConfig, system: str) -> pd.DataFrame:
    return generate_system_outputs(df=df, config=config, systems=[system])