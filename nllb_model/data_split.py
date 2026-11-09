"""Deterministic, purged multilingual holdout; no ML dependencies.

Rows are (source_text, target_text, source_language, target_language, corpus, line).
Generated concatenations have no provenance IDs. Rather than infer giant linked
groups from random concatenations, keep a fixed holdout and quarantine training
rows that overlap its texts or sentence spans, in either language/direction.
"""
from collections import Counter, defaultdict
import hashlib
import os
import re
import unicodedata
from itertools import zip_longest
from pathlib import Path

VERSION = "purged-multilingual-v4"
MIN_OVERLAP_TOKENS = 8


def default_validation_path():
    """Use this repository's reviewed benchmark unless explicitly overridden."""
    if os.environ.get("NLLB_VALIDATION_PATH"):
        return os.environ["NLLB_VALIDATION_PATH"]
    root = Path(__file__).resolve().parents[1]
    return str(root / "validation data")


def prepare_split(path, val_size=1000, tatoeba_path=None, db_eng_path=None,
                  external_path=None, synthetic_path=None):
    """Read the same corpora for training and the standalone audit."""
    files = [("de", Path(path) / "german.txt", Path(path) / "eastfrisian.txt", "deu_Latn")]
    if tatoeba_path is not None:
        files.append(("tatoeba", Path(tatoeba_path) / "english_tatoeba.txt",
                      Path(tatoeba_path) / "eastfrisian_tatoeba.txt", "eng_Latn"))
    if db_eng_path is not None:
        files.append(("dictionary_en", Path(db_eng_path) / "english_db.txt",
                      Path(db_eng_path) / "eastfrisian_for_english_db.txt", "eng_Latn"))
    additional = Path(path) / "additional"
    if (additional / "english.txt").exists() or (additional / "eastfrisian_for_english.txt").exists():
        files.append(("additional_en", additional / "english.txt",
                      additional / "eastfrisian_for_english.txt", "eng_Latn"))
    hashes = {}

    def read(source, src, tgt, language):
        for file in (src, tgt):
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        result = []
        with src.open(encoding="utf-8") as a, tgt.open(encoding="utf-8") as b:
            for i, (s, t) in enumerate(zip_longest(a, b), 1):
                if s is None or t is None:
                    raise ValueError(f"Parallel length mismatch at line {i}: {src} / {tgt}")
                if bool(s.strip()) != bool(t.strip()):
                    raise ValueError(f"One-sided blank at line {i}: {src} / {tgt}")
                result.append((s.strip(), t.strip(), language, "frs_Latn", source, i))
        return result

    rows, quotas = [], {"de": val_size}
    for source, src, tgt, language in files:
        loaded = read(source, src, tgt, language)
        rows.extend(loaded)
        if source != "de":
            quotas[source] = min(200 if source == "tatoeba" else 500, len(loaded) // 10)
    # Read current human data directly: generated files may predate new additions.
    # These sources get no new holdout quota, preserving the existing benchmark.
    manual_sources = [("additional_de", additional)]
    if tatoeba_path is not None:
        manual_sources.append(("tatoeba_de", Path(tatoeba_path)))
    for source, directory in manual_sources:
        if any((directory / name).exists() for name in ("german.txt", "eastfrisian.txt")):
            rows.extend(read(source, directory / "german.txt", directory / "eastfrisian.txt", "deu_Latn"))
    if tatoeba_path is not None:
        reference = Path(tatoeba_path).parent / "reference.tsv"
        if reference.exists():
            hashes[str(reference)] = hashlib.sha256(reference.read_bytes()).hexdigest()
            english_by_german = {}
            for line in reference.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t", 2)
                if len(fields) == 3 and fields[0].strip():
                    english_by_german.setdefault(fields[1], fields[0])
            rows.extend((english_by_german[r[0]], r[1], "eng_Latn", "frs_Latn", "tatoeba_en_current", r[5])
                        for r in list(rows) if r[4] == "tatoeba_de" and r[0] in english_by_german)
    external = []
    if external_path is not None:
        directory = Path(external_path)
        external = read("external", directory / "german.txt", directory / "eastfrisian.txt", "deu_Latn")
    train, validation, manifest = split_records(rows, quotas, external)
    # Freeze real validation first. Synthetic pairs are a separate train-only source.
    if synthetic_path is not None:
        directory = Path(synthetic_path)
        if any((directory / name).exists() for name in ("german.txt", "eastfrisian.txt")):
            synthetic = read("synthetic", directory / "german.txt", directory / "eastfrisian.txt", "deu_Latn")
            hits = overlapping_rows(synthetic, external + validation)
            seen = {record_id(row) for row in train + validation}
            accepted, excluded = [], []
            for i, row in enumerate(synthetic):
                key = record_id(row)
                if not normalize(row[0]) or not normalize(row[1]):
                    reason = "synthetic_blank"
                elif i in hits:
                    reason = "synthetic_heldout_overlap"
                elif key in seen:
                    reason = "synthetic_duplicate"
                else:
                    accepted.append(row)
                    seen.add(key)
                    continue
                excluded.append((row, reason))
            train.extend(accepted)
            manifest["counts"]["input"] += len(synthetic)
            manifest["counts"]["train"] = len(train)
            for reason, count in Counter(reason for _, reason in excluded).items():
                manifest["counts"]["excluded"][reason] = count
            manifest["excluded"].extend([r[4], r[5], record_id(r), reason] for r, reason in excluded)
            manifest["synthetic"] = {"input": len(synthetic), "train": len(accepted),
                                     "validation": 0, "path": str(directory)}
    manifest["input_sha256"] = hashes
    manifest["overlap_policy"] = {
        "whole_text": "exact normalized match, any length, either language",
        "partial_text": "containment in either direction, by language",
        "minimum_partial_tokens": MIN_OVERLAP_TOKENS,
        "synthetic_training_overlap": "allowed; exact bilingual duplicates removed",
    }
    manifest["by_corpus"] = {
        "train": dict(Counter(r[4] for r in train)),
        "validation": dict(Counter(r[4] for r in validation)),
    }
    return train, validation, manifest


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def record_id(row):
    value = "\0".join((row[2], normalize(row[0]), row[3], normalize(row[1])))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spans(text):
    """Whole text plus all nonempty sentence fragments.

    The overlap checker applies its minimum length before partial matching.
    Token boundaries prevent 'an' matching inside 'another'.
    """
    result = {normalize(text)} - {""}
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        value = normalize(part)
        if value:
            result.add(value)
    return result


def text_index(rows):
    texts = defaultdict(dict)
    postings = defaultdict(lambda: defaultdict(set))
    for index, row in enumerate(rows):
        for text, language in ((row[0], row[2]), (row[1], row[3])):
            value = normalize(text)
            if not value:
                continue
            texts[language][index] = " " + value + " "
            for token in set(value.split()):
                postings[language][token].add(index)
    return texts, postings


def overlapping_rows(rows, held_out, min_overlap_tokens=MIN_OVERLAP_TOKENS):
    """Return row indices overlapping a held-out whole text or sentence span.

    Exact whole-text matches are protected at any length. Partial matches must
    contain at least min_overlap_tokens words: vocabulary and number overlap
    alone is not evidence of leakage. Match both containment directions and
    languages. This is not a paraphrase or document-origin detector.
    """
    texts, postings = text_index(rows)
    protected = defaultdict(set)
    exact = defaultdict(set)
    for row in held_out:
        for text, language in ((row[0], row[2]), (row[1], row[3])):
            value = normalize(text)
            if value:
                exact[language].add(" " + value + " ")
            protected[language].update(value for value in spans(text)
                                       if len(value.split()) >= min_overlap_tokens)
    result = {i for language, values in texts.items()
              for i, value in values.items() if value in exact[language]}
    for language, values in protected.items():
        for value in values:
            tokens = set(value.split())
            candidates = min((postings[language].get(t, set()) for t in tokens), key=len)
            needle = " " + value + " "
            result.update(i for i in candidates if needle in texts[language][i])
    # Reverse containment: candidate whole text contained in a held-out span.
    protected_texts = {}
    protected_postings = defaultdict(lambda: defaultdict(set))
    for language, values in protected.items():
        protected_texts[language] = [" " + v + " " for v in sorted(values)]
        for i, value in enumerate(protected_texts[language]):
            for token in set(value.split()):
                protected_postings[language][token].add(i)
    for language, values in texts.items():
        for i, value in values.items():
            if i in result:
                continue
            if len(value.split()) < min_overlap_tokens:
                continue
            candidates = min((protected_postings[language].get(t, set())
                              for t in set(value.split())), key=len)
            if any(value in protected_texts[language][j] for j in candidates):
                result.add(i)
    return result


def split_records(rows, quotas, external=(), seed=42):
    """Deduplicate, protect external evaluation, select holdout, then purge.

    Quotas count unique bilingual pairs, before reverse directions are added.
    An explicit quarantine preserves auditability without silently moving
    synthetic relatives into validation and changing its intended composition.
    """
    if any(n < 0 for n in quotas.values()):
        raise ValueError("Validation quotas must be nonnegative")
    seen, clean, removed, empty_pairs = set(), [], [], []
    for row in rows:
        # Intentional empty-input/empty-output examples belong in training.
        if not row[0].strip() and not row[1].strip():
            empty_pairs.append(row)
            continue
        if not normalize(row[0]) or not normalize(row[1]):
            removed.append((row, "blank"))
            continue
        key = record_id(row)
        if key in seen:
            removed.append((row, "duplicate"))
            continue
        seen.add(key)
        clean.append(row)
    external_hits = overlapping_rows(clean, external) if external else set()
    removed.extend((row, "external_overlap") for i, row in enumerate(clean) if i in external_hits)
    clean = [row for i, row in enumerate(clean) if i not in external_hits]
    validation_ids = set()
    for source, quota in quotas.items():
        candidates = [row for row in clean if row[4] == source]
        candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{record_id(row)}".encode()).hexdigest())
        if quota > len(candidates):
            raise ValueError(f"{source}: requested {quota} validation pairs, only {len(candidates)} available")
        validation_ids.update(record_id(row) for row in candidates[:quota])
    validation = [row for row in clean if record_id(row) in validation_ids]
    training = [row for row in clean if record_id(row) not in validation_ids]
    hits = overlapping_rows(training, validation)
    removed.extend((row, "internal_overlap") for i, row in enumerate(training) if i in hits)
    training = [row for i, row in enumerate(training) if i not in hits]
    training.extend(empty_pairs)
    if not training or not validation:
        raise ValueError("Split must retain nonempty training and validation sets")
    manifest = {
        "version": VERSION, "seed": seed, "quotas": quotas,
        "counts": {"input": len(rows), "train": len(training), "validation": len(validation),
                   "excluded": dict(Counter(reason for _, reason in removed))},
        "training_membership": "All input rows not listed in validation or excluded (corpus + physical line).",
        "validation": [[r[4], r[5], record_id(r)] for r in validation],
        "excluded": [[r[4], r[5], record_id(r), reason] for r, reason in removed],
    }
    return training, validation, manifest
