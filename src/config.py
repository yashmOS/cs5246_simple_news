from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Paths:
    root: Path = Path('.')
    data_dir: Path = Path('./data')
    output_dir: Path = Path('./outputs')
    figure_dir: Path = Path('./outputs/figures')
    table_dir: Path = Path('./outputs/tables')
    sample_dir: Path = Path('./outputs/samples')


@dataclass
class SelectorConfig:
    max_sentences: int = 3
    similarity_threshold: float = 0.05
    top_entity_k: int = 6
    require_number_coverage: bool = True
    require_date_coverage: bool = True


@dataclass
class SimplifyConfig:
    enabled: bool = True
    max_word_length: int = 9
    min_word_frequency_rank: int = 3000
    max_replacements_per_doc: int = 8
    use_glossary_fallback: bool = True
    protected_words: List[str] = field(default_factory=lambda: [
        'said', 'says', 'mr', 'mrs', 'ms', 'government', 'minister', 'president',
        'police', 'court', 'judge', 'company', 'officials'
    ])


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    sample_size: int = 200
    systems: List[str] = field(default_factory=lambda: ['s0_lead3', 's1_textrank', 's2_coverage', 's3_simplified'])
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    simplifier: SimplifyConfig = field(default_factory=SimplifyConfig)


DEFAULT_CONFIG = ExperimentConfig()
