# SimpleNews module improvements

I created a revised drop-in `src/` package so your notebook can import the stronger modules without you having to overwrite the original root files.

## What changed

### `src/preprocess.py`
- Replaced the very fragile sentence splitter with a safer rule-based splitter that protects common abbreviations, initials, and decimal numbers before splitting.
- Reworked date extraction so month names are matched longest-first and added support for month-year, month-day-year, day-month-year, and common numeric dates.
- Reworked entity extraction so it scans sentence tokens instead of relying on one broad regex. It now strips common leading noise like `On Tuesday ...`, handles connectors like `of/the/and`, and splits role-mediated spans such as `Apple CEO Tim Cook` into more useful entities.
- Kept everything dependency-light so you do not need spaCy/NLTK in the notebook.

### `src/selectors.py`
- Added a defensive TextRank fallback so short or stopword-only inputs do not crash TF-IDF.
- Made `similarity_threshold` actually do something by filtering weak sentence-similarity edges before PageRank.
- Rebuilt `coverage_aware_select()` as a bounded greedy selector that always respects `max_sentences`.
- Added a redundancy penalty so the selector is less likely to choose near-duplicate sentences.
- Wired number/date coverage requirements into the selector so those config flags now matter.

### `src/evaluation.py`
- Switched sentence counting in readability to the improved sentence splitter.
- Made readability safe on empty output instead of pretending empty text has one word and one sentence.
- Kept your existing `entity_coverage`, `number_coverage`, and `date_coverage` columns for backward compatibility.
- Added precision/F1/unsupported-fact ratios so the evaluation can now expose when a system adds unsupported entities, numbers, or dates.
- Added `copy_token_ratio` and `novel_content_token_ratio` so novelty is less one-dimensional.

### `src/simplify.py`
- Added safer case preservation for replacements.
- Strengthened the gating logic so replacements only happen for genuinely hard words.
- Made glossary fallback more useful by allowing hard unreplaced words to still contribute glossary hints.

### `src/pipeline.py`
- Passed `similarity_threshold`, `require_number_coverage`, and `require_date_coverage` from config into the selector.
- Made `simplifier.enabled` actually control whether S3 runs lexical simplification.
- Added a few safer string conversions so missing values are less likely to break execution.

## Why I made these changes

### Why `preprocess.py` changed
Your selector and your factual metrics both depend on preprocessing. In the original setup, one bad regex could affect sentence boundaries, entity coverage, date coverage, and the final sentence selection at the same time. The revised preprocessing is still lightweight, but it is less likely to misread `Dr. Smith`, `April 2025`, or `On Tuesday President Joe Biden ...`.

### Why `selectors.py` changed
The original coverage selector had two structural problems: it could exceed the sentence cap, and it only *appended* extra sentences after an initial top-k selection. That meant a weak early sentence could stay locked in. The revised version scores every candidate by salience, new fact coverage, redundancy, and a small lead bias each round, then picks the best next sentence. That is much closer to what a coverage-aware extractor should do.

### Why `evaluation.py` changed
Recall-only fact coverage can make a system look better than it is. A summary can cover many source facts and still hallucinate extra ones. By keeping your old recall columns and adding precision/F1/unsupported ratios, you can still compare with old notebook outputs while getting a more honest factual picture.

### Why `simplify.py` changed
Lexical simplification should be conservative in news. It is better to miss some simplifications than to change names, dates, or the meaning of a sentence. The updated module stays conservative, but it is a bit more informative when it chooses not to replace a hard word.

### Why `pipeline.py` changed
A config field that never affects runtime is misleading. The revised pipeline makes your existing config knobs real, so experiments in the notebook now line up better with what the config says should happen.

## Compatibility notes
- Your original root files are unchanged.
- The notebook can use the improved code by importing from the new `src/` package in `/mnt/data/src`.
- The notebook summary cell that uses `rouge1`, `rouge2`, `rougel`, `flesch_reading_ease`, `fkgl`, `entity_coverage`, `number_coverage`, `date_coverage`, `novel_token_ratio`, `word_count`, `sentence_count`, and `avg_sentence_len` will still work.
- New metrics were added, not substituted, so you can extend your summary tables later if you want.
