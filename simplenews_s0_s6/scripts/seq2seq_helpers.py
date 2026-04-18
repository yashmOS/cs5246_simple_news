from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .common import clear_memory, detect_device, finalize_system_output
from src.evaluation import evaluate_output
from src.preprocess import extract_dates, extract_entities, extract_numbers, normalize_whitespace, tokenize
from src.selectors import coverage_aware_select


@dataclass
class S4Settings:
    max_sentences: int = 5
    top_entity_k: int = 8
    similarity_threshold: float = 0.05
    require_number_coverage: bool = True
    require_date_coverage: bool = True
    min_entity_f1: float = 0.58
    min_number_f1: float = 0.75
    min_date_f1: float = 0.75
    max_entity_unsupported: float = 0.25
    max_number_unsupported: float = 0.15
    max_date_unsupported: float = 0.15
    min_words: int = 35
    max_words: int = 110
    min_new_tokens: int = 52
    max_new_tokens: int = 160
    num_beams: int = 5
    no_repeat_ngram_size: int = 3
    length_penalty: float = 0.95
    repetition_penalty: float = 1.08
    max_input_tokens: int = 768
    batch_size: int = 2


@dataclass
class S5Settings:
    max_new_tokens: int = 144
    min_new_tokens: int = 64
    num_beams: int = 5
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    repetition_penalty: float = 1.06
    max_input_tokens: int = 1024
    batch_size: int = 4


@dataclass
class S6Settings:
    max_new_tokens: int = 144
    batch_size: int = 8
    max_input_tokens: int = 1024
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    repetition_penalty: float = 1.05


_model_cache: Dict[str, tuple] = {}


def _clean_generation_config(model, tokenizer):
    gen_cfg = copy.deepcopy(model.generation_config)
    if hasattr(gen_cfg, "max_length"):
        gen_cfg.max_length = None
    if hasattr(gen_cfg, "min_length"):
        gen_cfg.min_length = None
    if getattr(gen_cfg, "forced_bos_token_id", None) is None and getattr(tokenizer, "bos_token_id", None) is not None:
        gen_cfg.forced_bos_token_id = tokenizer.bos_token_id
    return gen_cfg


def load_model_bundle(model_name_or_path: str):
    if model_name_or_path in _model_cache:
        return _model_cache[model_name_or_path]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    device = detect_device()
    if device != "cpu":
        model = model.to(device)
    model.eval()
    model.generation_config = _clean_generation_config(model, tokenizer)
    _model_cache[model_name_or_path] = (tokenizer, model)
    return tokenizer, model


def generate_texts(
    model_name_or_path: str,
    prompts: List[str],
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.05,
    max_input_tokens: int = 1024,
    batch_size: int = 4,
) -> List[str]:
    import torch

    tokenizer, model = load_model_bundle(model_name_or_path)
    device = detect_device()
    outputs: List[str] = []

    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = prompts[start:start + max(1, batch_size)]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        if device != "cpu":
            enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_ids = model.generate(
                        **enc,
                        generation_config=model.generation_config,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                    )
            else:
                out_ids = model.generate(
                    **enc,
                    generation_config=model.generation_config,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                )
        outputs.extend([normalize_whitespace(t) for t in tokenizer.batch_decode(out_ids, skip_special_tokens=True)])
    return outputs


def build_s4_prompt(selected_text: str) -> str:
    return (
        "You are simplifying a factual news summary for adult ESL readers.\n"
        "Rewrite the summary in plain English while preserving every factual detail from the source summary.\n"
        "Rules:\n"
        "- Preserve all names, dates, numbers, money amounts, percentages, and places exactly.\n"
        "- Do not add any unsupported facts.\n"
        "- Prefer shorter sentences and common words when meaning stays the same.\n"
        "- Keep the chronology and the main actors the same.\n"
        "- Write 3 to 5 short coherent sentences.\n"
        "- Aim for about 45 to 85 words.\n\n"
        f"Source summary:\n{selected_text}\n\n"
        "Simplified summary:"
    )


def build_s5_prompt(article_text: str) -> str:
    return (
        "Summarize this news article in plain English for adult ESL readers.\n"
        "Rules:\n"
        "- Keep the most important names, dates, numbers, and places.\n"
        "- Do not add any new facts.\n"
        "- Write 3 to 5 short, coherent sentences.\n"
        "- Aim for about 60 to 90 words.\n\n"
        f"Article:\n{article_text}\n\n"
        "Summary:"
    )


def build_s5_input(article_text: str, family_model_name: Optional[str] = None) -> str:
    model_name = str(family_model_name or "").lower()
    if "bart" in model_name:
        return str(article_text)
    return build_s5_prompt(str(article_text))


def _word_count(text: str) -> int:
    return len(tokenize(text))


def s4_guard_details(source_text: str, candidate_text: str, settings: S4Settings) -> Dict[str, object]:
    metrics = evaluate_output(source=source_text, reference="", pred=candidate_text)
    src_entities = extract_entities(source_text)
    src_numbers = extract_numbers(source_text)
    src_dates = extract_dates(source_text)

    wc = _word_count(candidate_text)

    entity_f1 = float(metrics.get("entity_f1", 0.0))
    number_f1 = float(metrics.get("number_f1", 0.0))
    date_f1 = float(metrics.get("date_f1", 0.0))

    entity_unsup = float(metrics.get("entity_unsupported_ratio", 0.0))
    number_unsup = float(metrics.get("number_unsupported_ratio", 0.0))
    date_unsup = float(metrics.get("date_unsupported_ratio", 0.0))

    entity_ok = (not src_entities) or (entity_f1 >= settings.min_entity_f1 and entity_unsup <= settings.max_entity_unsupported)
    number_ok = (not src_numbers) or (number_f1 >= settings.min_number_f1 and number_unsup <= settings.max_number_unsupported)
    date_ok = (not src_dates) or (date_f1 >= settings.min_date_f1 and date_unsup <= settings.max_date_unsupported)
    length_ok = settings.min_words <= wc <= settings.max_words

    reject_reasons: List[str] = []
    if not entity_ok:
        reject_reasons.append("entity_guard")
    if not number_ok:
        reject_reasons.append("number_guard")
    if not date_ok:
        reject_reasons.append("date_guard")
    if not length_ok:
        reject_reasons.append("length_guard")

    return {
        "accepted": bool(entity_ok and number_ok and date_ok and length_ok),
        "reject_reasons": reject_reasons,
        "src_entities_count": len(src_entities),
        "src_numbers_count": len(src_numbers),
        "src_dates_count": len(src_dates),
        "raw_word_count": wc,
        "entity_f1": entity_f1,
        "number_f1": number_f1,
        "date_f1": date_f1,
        "entity_unsupported_ratio": entity_unsup,
        "number_unsupported_ratio": number_unsup,
        "date_unsupported_ratio": date_unsup,
    }


def run_s4_dataframe(df: pd.DataFrame, model_name_or_path: str, settings: Optional[S4Settings] = None) -> pd.DataFrame:
    settings = settings or S4Settings()
    rows = []
    prompts: List[str] = []
    scaffolds: List[str] = []

    for row in df.itertuples(index=False):
        title = str(getattr(row, "title", "") or "")
        scaffold = coverage_aware_select(
            str(getattr(row, "article", "") or ""),
            title=title,
            max_sentences=settings.max_sentences,
            top_entity_k=settings.top_entity_k,
            similarity_threshold=settings.similarity_threshold,
            require_number_coverage=settings.require_number_coverage,
            require_date_coverage=settings.require_date_coverage,
        )
        scaffolds.append(scaffold)
        prompts.append(build_s4_prompt(scaffold))

    raw_outputs = generate_texts(
        model_name_or_path,
        prompts,
        max_new_tokens=settings.max_new_tokens,
        min_new_tokens=settings.min_new_tokens,
        num_beams=settings.num_beams,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        length_penalty=settings.length_penalty,
        repetition_penalty=settings.repetition_penalty,
        max_input_tokens=settings.max_input_tokens,
        batch_size=settings.batch_size,
    )

    for row, scaffold, raw in zip(df.itertuples(index=False), scaffolds, raw_outputs):
        article = str(getattr(row, "article", "") or "")
        reference = str(getattr(row, "reference", "") or "")
        details = s4_guard_details(scaffold, raw, settings=settings)
        final = raw if details["accepted"] else scaffold
        metrics = evaluate_output(source=article, reference=reference, pred=final)
        rows.append({
            "doc_id": str(getattr(row, "doc_id", "")),
            "system": "S4",
            "title": str(getattr(row, "title", "") or ""),
            "article": article,
            "reference": reference,
            "output": final,
            "s4_scaffold": scaffold,
            "s4_raw": raw,
            "s4_fallback_used": not details["accepted"],
            "s4_accept": details["accepted"],
            "s4_reject_reasons": ";".join(details["reject_reasons"]),
            "s4_src_entities_count": details["src_entities_count"],
            "s4_src_numbers_count": details["src_numbers_count"],
            "s4_src_dates_count": details["src_dates_count"],
            "s4_raw_word_count": details["raw_word_count"],
            "s4_entity_f1": details["entity_f1"],
            "s4_number_f1": details["number_f1"],
            "s4_date_f1": details["date_f1"],
            "s4_entity_unsupported_ratio": details["entity_unsupported_ratio"],
            "s4_number_unsupported_ratio": details["number_unsupported_ratio"],
            "s4_date_unsupported_ratio": details["date_unsupported_ratio"],
            **metrics,
        })
    clear_memory()
    return finalize_system_output(pd.DataFrame(rows), system="S4")


def run_s5_dataframe(
    df: pd.DataFrame,
    model_name_or_path: str,
    *,
    family_model_name: Optional[str] = None,
    settings: Optional[S5Settings] = None,
) -> pd.DataFrame:
    settings = settings or S5Settings()
    prompts = [build_s5_input(str(a), family_model_name=family_model_name or model_name_or_path) for a in df["article"].fillna("").astype(str).tolist()]
    outputs = generate_texts(
        model_name_or_path,
        prompts,
        max_new_tokens=settings.max_new_tokens,
        min_new_tokens=settings.min_new_tokens,
        num_beams=settings.num_beams,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        length_penalty=settings.length_penalty,
        repetition_penalty=settings.repetition_penalty,
        max_input_tokens=settings.max_input_tokens,
        batch_size=settings.batch_size,
    )
    rows = []
    for row, output in zip(df.itertuples(index=False), outputs):
        article = str(getattr(row, "article", "") or "")
        reference = str(getattr(row, "reference", "") or "")
        metrics = evaluate_output(source=article, reference=reference, pred=output)
        rows.append({
            "doc_id": str(getattr(row, "doc_id", "")),
            "system": "S5",
            "title": str(getattr(row, "title", "") or ""),
            "article": article,
            "reference": reference,
            "output": output,
            "s5_mode": "single_pass",
            "s5_num_chunks": 1,
            **metrics,
        })
    clear_memory()
    return finalize_system_output(pd.DataFrame(rows), system="S5")


def run_s6_dataframe(
    df: pd.DataFrame,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    settings: Optional[S6Settings] = None,
) -> pd.DataFrame:
    settings = settings or S6Settings()

    from s6_dpo.model import get_model, generate_summary_batched
    from transformers import AutoTokenizer

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model_name = model_name
    device = detect_device()
    model = get_model(cfg, checkpoint_path=checkpoint_path)
    if device != "cpu":
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    articles = df["article"].fillna("").astype(str).tolist()
    outputs: List[str] = []
    batch_size = max(1, settings.batch_size)
    for start in range(0, len(articles), batch_size):
        batch_articles = articles[start:start + batch_size]
        outputs.extend(
            generate_summary_batched(model, tokenizer, batch_articles, max_new_tokens=settings.max_new_tokens)
        )

    rows = []
    for row, output in zip(df.itertuples(index=False), outputs):
        article = str(getattr(row, "article", "") or "")
        reference = str(getattr(row, "reference", "") or "")
        output = normalize_whitespace(str(output))
        metrics = evaluate_output(source=article, reference=reference, pred=output)
        rows.append({
            "doc_id": str(getattr(row, "doc_id", "")),
            "id": str(getattr(row, "id", getattr(row, "doc_id", ""))),
            "system": "S6",
            "system_label": "S6 DPO",
            "title": str(getattr(row, "title", "") or ""),
            "article": article,
            "reference": reference,
            "output": output,
            "replacements": "[]",
            "glossary": "[]",
            "s5_mode": "dpo",
            "s5_num_chunks": np.nan,
            **metrics,
        })
    clear_memory()
    return finalize_system_output(pd.DataFrame(rows), system="S6")
