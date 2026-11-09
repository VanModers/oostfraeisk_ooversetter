"""Build the separate two-stage corpus from original aligned texts and the dictionary.

Run locally: python data/create_two_stage_dataset.py
The cluster needs only the generated data/two_stage files, not WFDOT.db.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nllb_model"))
from data_split import (MIN_OVERLAP_TOKENS, default_validation_path, normalize,
                        overlapping_rows, prepare_split, record_id)
from create_dataset import NOUN_LIMIT, SQL_SEED, cut_brackets, filter_all_abbreviations

SCHEMA = "original-aligned-two-stage-v1"
TEXT_CORPORA = {"tatoeba_de", "tatoeba_en", "additional_de", "additional_en", "synthetic"}
FIELDS = ("source", "target", "source_language", "target_language", "corpus", "source_id")


def digest(path, text=False):
    content = Path(path).read_bytes()
    if text:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_pairs(directory, corpus, language="deu_Latn", names=("german.txt", "eastfrisian.txt")):
    with ((directory / names[0]).open(encoding="utf-8") as source,
          (directory / names[1]).open(encoding="utf-8") as target):
        for line, (src, tgt) in enumerate(zip_longest(source, target), 1):
            if src is None or tgt is None or bool(src.strip()) != bool(tgt.strip()):
                raise ValueError(f"Unaligned pair: {directory}, line {line}")
            yield (src.strip(), tgt.strip(), language, "frs_Latn", corpus, line)


def clean_gloss(value):
    """Use the first semicolon-delimited gloss; retain its complete comma clauses."""
    return filter_all_abbreviations(cut_brackets((value or "").replace("\n", " ").split(";")[0])).strip()


def dictionary_rows(database):
    # A read-only connection also prevents silently creating an empty database.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        phrases = connection.execute(
            "SELECT ID, Ostfriesisch, Deutsch, Englisch FROM WB "
            "WHERE Deutsch != '-' AND Wortart = 'Phrase' ORDER BY ID")
        for identifier, frisian, german, english in phrases:
            frisian = frisian.replace("\n", " ").strip()
            for value, language, corpus in ((german, "deu_Latn", "dictionary_de"),
                                             (english, "eng_Latn", "dictionary_en")):
                gloss = clean_gloss(value)
                if normalize(gloss) and normalize(frisian):
                    yield (gloss, frisian, language, "frs_Latn", corpus, identifier)
        # Preserve the vocabulary used by the old noun templates, without producing
        # four artificial sentences per meaning. Comma-separated noun meanings
        # follow the existing dictionary's convention; phrase commas are never split.
        nouns = connection.execute(
            f"SELECT ID, Ostfriesisch, Deutsch FROM WB "
            f"WHERE Deutsch != '-' AND Wortart = 'Substantiv' "
            f"ORDER BY ((ID * {SQL_SEED}) % 100000), ID LIMIT {NOUN_LIMIT}")
        for identifier, frisian, german in nouns:
            frisian = frisian.replace("\n", " ").strip()
            for meaning in clean_gloss(german).split(", "):
                if normalize(meaning) and normalize(frisian):
                    yield (meaning.strip(), frisian, "deu_Latn", "frs_Latn", "dictionary_nouns", identifier)


def retain_training(rows, validation, external):
    unique, seen, exclusions = [], set(), Counter()
    for row in rows:
        if not normalize(row[0]) or not normalize(row[1]):
            exclusions["blank"] += 1
        elif record_id(row) in seen:
            exclusions["duplicate"] += 1
        else:
            unique.append(row)
            seen.add(record_id(row))
    hits = overlapping_rows(unique, validation + external)
    exclusions["heldout_overlap"] = len(hits)
    return [row for i, row in enumerate(unique) if i not in hits], dict(exclusions)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(zip(FIELDS, row)), ensure_ascii=False) + "\n")


def build(root, database, validation_path, output):
    # Retain exactly the current internal validation set for this experiment.
    # Generated training rows are used only to reconstruct that set, never as
    # training inputs in the rebuilt corpus.
    _, validation, previous = prepare_split(
        root, tatoeba_path=root / "tatoeba/training", db_eng_path=root,
        external_path=validation_path,
    )
    inputs = {}
    for path, value in previous["input_sha256"].items():
        path = Path(path).resolve()
        try:
            key = str(path.relative_to(root.resolve().parent)).replace("\\", "/")
        except ValueError:
            key = "benchmark/" + path.name
        inputs[key] = value
    inputs["dictionary/" + database.name] = digest(database)

    tatoeba = list(read_pairs(root / "tatoeba/training", "tatoeba_de"))
    rows = tatoeba + list(read_pairs(root / "additional", "additional_de"))
    # Derive English alignment directly from current German/FRS originals.
    reference = root / "tatoeba/reference.tsv"
    english_by_german = {}
    for line in reference.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0].strip():
            english_by_german.setdefault(fields[1], fields[0])
    rows.extend((english_by_german[r[0]], r[1], "eng_Latn", "frs_Latn", "tatoeba_en", r[5])
                for r in tatoeba if r[0] in english_by_german)
    rows.extend(read_pairs(root / "additional", "additional_en", "eng_Latn",
                           ("english.txt", "eastfrisian_for_english.txt")))
    rows.extend(read_pairs(root / "synthetic", "synthetic"))
    for name in ("german.txt", "eastfrisian.txt"):
        inputs["data/synthetic/" + name] = digest(root / "synthetic" / name)
    # Text sources take precedence over dictionary duplicates, preserving stage-2 membership.
    rows.extend(dictionary_rows(database))
    external = list(read_pairs(validation_path, "external"))
    train, exclusions = retain_training(rows, validation, external)
    text_count = sum(row[4] in TEXT_CORPORA for row in train)
    if not train or not validation or not text_count:
        raise ValueError("Both training stages and validation must be nonempty")
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train.jsonl", train)
    write_rows(output / "validation.jsonl", validation)
    manifest = {
        "schema": SCHEMA, "input_sha256": inputs,
        "files": {name: digest(output / name, text=True) for name in ("train.jsonl", "validation.jsonl")},
        "external_sha256": {name: digest(validation_path / name, text=True) for name in ("german.txt", "eastfrisian.txt")},
        "file_hash_policy": "Corpus and benchmark hashes normalize CRLF to LF for cross-platform checkouts.",
        "counts": {"input": len(rows), "stage1_pairs": len(train), "stage2_pairs": text_count,
                   "validation_pairs": len(validation), "excluded": exclusions},
        "by_corpus": dict(Counter(row[4] for row in train)),
        "stage2_corpora": sorted(TEXT_CORPORA), "bidirectional": True,
        "internal_validation": previous["validation"],
        "overlap_policy": {"whole_text": "exact normalized match, either language",
                           "minimum_partial_tokens": MIN_OVERLAP_TOKENS},
        "dictionary_policy": "First semicolon gloss with existing annotation cleanup; no length-based comma clipping. Nouns are lexical pairs.",
        "augmentation": "None: no noun templates or random concatenations.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "by_corpus": manifest["by_corpus"]}, indent=2))
    print(f"Wrote {output}; original training files are unchanged.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=REPO / "data")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--validation-path", type=Path, default=Path(default_validation_path()))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(args.data_path, args.database or args.data_path / "dictionary/WFDOT.db",
          args.validation_path, args.output_dir or args.data_path / "two_stage")


if __name__ == "__main__":
    main()
