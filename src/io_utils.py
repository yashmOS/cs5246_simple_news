from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


ARTICLE_CANDIDATES = ['article', 'text', 'document', 'story', 'source', 'content']
SUMMARY_CANDIDATES = ['highlights', 'summary', 'target', 'reference', 'abstract']
TITLE_CANDIDATES = ['title', 'headline']


class DataFormatError(ValueError):
    pass


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in columns:
        col_low = col.lower()
        if any(c in col_low for c in candidates):
            return col
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    article_col = _pick_column(df.columns, ARTICLE_CANDIDATES)
    summary_col = _pick_column(df.columns, SUMMARY_CANDIDATES)
    title_col = _pick_column(df.columns, TITLE_CANDIDATES)
    if article_col is None:
        raise DataFormatError(f'Could not detect article column from: {list(df.columns)}')
    return {'article': article_col, 'summary': summary_col, 'title': title_col}


def load_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = pd.read_csv(path)
    colmap = detect_columns(df)
    return df, colmap


def load_default_dataset(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Path]:
    preferred = ['train.csv', 'validation.csv', 'test.csv']
    for name in preferred:
        path = data_dir / name
        if path.exists():
            df, colmap = load_csv(path)
            return df, colmap, path
    csvs = sorted(data_dir.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No CSV files found in {data_dir.resolve()}')
    df, colmap = load_csv(csvs[0])
    return df, colmap, csvs[0]


def standardize_dataframe(df: pd.DataFrame, colmap: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out['article'] = df[colmap['article']].fillna('').astype(str)
    out['reference'] = df[colmap['summary']].fillna('').astype(str) if colmap['summary'] else ''
    out['title'] = df[colmap['title']].fillna('').astype(str) if colmap['title'] else ''
    out = out.reset_index(drop=True)
    out['doc_id'] = out.index
    return out[['doc_id', 'title', 'article', 'reference']]
