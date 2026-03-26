from __future__ import annotations

from typing import Dict, List

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


def generate_system_outputs(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    freq_rank = token_frequency_rank(df['article'].tolist())
    rows = []
    for row in df.itertuples(index=False):
        source = row.article
        reference = row.reference
        title = row.title

        s0 = lead3(source, max_sentences=config.selector.max_sentences)
        s1 = textrank_select(source, max_sentences=config.selector.max_sentences)
        s2 = coverage_aware_select(
            source,
            title=title,
            max_sentences=config.selector.max_sentences,
            top_entity_k=config.selector.top_entity_k,
        )
        s3_text, replacements, glossary = conservative_lexical_simplify(
            s2,
            frequency_rank=freq_rank,
            max_word_length=config.simplifier.max_word_length,
            min_word_frequency_rank=config.simplifier.min_word_frequency_rank,
            max_replacements_per_doc=config.simplifier.max_replacements_per_doc,
            protected_words=config.simplifier.protected_words,
            use_glossary_fallback=config.simplifier.use_glossary_fallback,
        )

        system_outputs = {
            's0_lead3': (s0, [], []),
            's1_textrank': (s1, [], []),
            's2_coverage': (s2, [], []),
            's3_simplified': (s3_text, replacements, glossary),
        }

        for sys_name in config.systems:
            output, reps, gloss = system_outputs[sys_name]
            metrics = evaluate_output(source=source, reference=reference, pred=output)
            rows.append({
                'doc_id': row.doc_id,
                'system': sys_name,
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
