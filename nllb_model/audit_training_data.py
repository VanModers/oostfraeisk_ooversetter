"""Read-only corpus audit; standard library only. Run from any directory.

Prints JSON to stdout. --baseline identifies the Git revision against which
manual additions are measured. No datasets are regenerated or modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
from data_split import prepare_split, overlapping_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tatoeba_de": ("data/tatoeba/training/german.txt", "data/tatoeba/training/eastfrisian.txt"),
    "additional": ("data/additional/german.txt", "data/additional/eastfrisian.txt"),
    "de": ("data/german.txt", "data/eastfrisian.txt"),
    "tatoeba": ("data/tatoeba/training/english_tatoeba.txt", "data/tatoeba/training/eastfrisian_tatoeba.txt"),
    "dictionary_en": ("data/english_db.txt", "data/eastfrisian_for_english_db.txt"),
    "external_validation": ("validation data/german.txt", "validation data/eastfrisian.txt"),
}


def read_source(paths):
    lines = []
    hashes = {}
    for relative in paths:
        payload = (ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        # Match the trainer's physical-line iteration and whitespace stripping.
        with (ROOT / relative).open(encoding="utf-8") as stream:
            lines.append([line.strip() for line in stream])
    if len(lines[0]) != len(lines[1]):
        raise ValueError(f"Unequal line counts: {paths}: {list(map(len, lines))}")
    pairs = list(zip(*lines))
    counts = Counter(pairs)
    return pairs, {
        "rows": len(pairs),
        "sha256": hashes,
        "both_blank_lines": [i for i, (s, t) in enumerate(pairs, 1) if not s and not t],
        "one_sided_blank_lines": [i for i, (s, t) in enumerate(pairs, 1) if bool(s) != bool(t)],
        "duplicate_extra_rows": sum(n - 1 for n in counts.values()),
        "identical_source_target_lines": [i for i, (s, t) in enumerate(pairs, 1) if s and s == t],
        "max_characters": [max(map(len, side), default=0) for side in lines],
        "replacement_character_lines": [i for i, (s, t) in enumerate(pairs, 1) if '\ufffd' in s + t],
    }


def split_like_trainer(pairs, seed, size):
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    held_out = set(indices[:size])
    return ([p for i, p in enumerate(pairs) if i not in held_out],
            [p for i, p in enumerate(pairs) if i in held_out])


def audit(baseline):
    corpora, statistics = {}, {}
    for name, paths in SOURCES.items():
        corpora[name], statistics[name] = read_source(paths)
    corpora["manual"] = corpora["tatoeba_de"] + corpora["additional"]
    report = {"sources": statistics, "legacy_row_split_overlap": {}}
    train, validation = [], []
    for name, seed, size in [("de", 42, 1000), ("tatoeba", 43, 200), ("dictionary_en", 44, 500)]:
        pairs = [(s, t) for s, t in corpora[name] if (s and t) or (name == "de" and not s and not t)]
        if name != "de":
            size = min(size, len(pairs) // 10)
        tr, va = split_like_trainer(pairs, seed, size)
        train_set = set(tr)
        report["legacy_row_split_overlap"][name] = {
            "train_rows": len(tr), "validation_rows": len(va),
            "validation_rows_also_in_same_source_train": sum(p in train_set for p in va),
        }
        train.extend(tr)
        validation.extend(va)
    train_frs = {f for _, f in train}
    report["legacy_row_split_overlap"]["validation_rows_sharing_frs_with_any_training_source"] = sum(f in train_frs for _, f in validation)
    de_set = set(corpora["de"])
    de_text = {g for g, _ in de_set}
    frs_text = {f for _, f in de_set}
    report["external_overlap_with_generated_de"] = {
        "exact_pair_lines": [i for i, p in enumerate(corpora["external_validation"], 1) if p in de_set],
        "source_text_lines": [i for i, (g, _) in enumerate(corpora["external_validation"], 1) if g in de_text],
        "target_text_lines": [i for i, (_, f) in enumerate(corpora["external_validation"], 1) if f in frs_text],
    }
    # Whole-row equality misses evaluation sentences inside augmented paragraphs.
    report["external_full_text_substring_matches"] = {}
    for name in ("de", "tatoeba", "dictionary_en"):
        matches = []
        for column, language in ((0, "de"), (1, "frs")):
            if column == 0 and name != "de":
                continue
            joined = "\n".join(pair[column] for pair in corpora[name])
            for line, pair in enumerate(corpora["external_validation"], 1):
                if pair[column] and pair[column] in joined:
                    matches.append({"evaluation_line": line, "language": language})
        report["external_full_text_substring_matches"][name] = matches
    legacy_paths = ("data/fan teksten/german.txt", "data/fan teksten/eastfrisian.txt")
    legacy_exists = subprocess.run(
        ["git", "cat-file", "-e", f"{baseline}:{legacy_paths[0]}"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    baseline_sources = [legacy_paths] if legacy_exists else [SOURCES["tatoeba_de"], SOURCES["additional"]]
    old = []
    for paths in baseline_sources:
        sides = []
        for path in paths:
            raw = subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=ROOT)
            sides.append([line.strip() for line in raw.decode("utf-8").splitlines()])
        if len(sides[0]) != len(sides[1]):
            raise ValueError("Git baseline text files are not aligned")
        old.extend(zip(*sides))
    current = corpora["manual"]
    old_set = set(old)
    # Do not mislabel edits to old rows as an append-only addition.
    prefix_unchanged = current[:len(old)] == old
    report["manual_vs_baseline"] = {
        "baseline": baseline, "baseline_rows": len(old),
        "net_added_rows": len(current) - len(old), "old_prefix_unchanged": prefix_unchanged,
    }
    if prefix_unchanged:
        new = current[len(old):]
        report["manual_vs_baseline"].update({
            "new_row_range": [len(old) + 1, len(current)] if new else [],
            "new_exact_pairs_already_generated": sum(p in de_set for p in new),
            "new_exact_pairs_already_in_old_manual": sum(p in old_set for p in new),
            "new_duplicate_extra_rows": len(new) - len(set(new)),
            "new_max_characters": [max((len(p[i]) for p in new), default=0) for i in (0, 1)],
        })
    report["manual_pairs_missing_from_generated"] = sum(p not in de_set for p in current)
    current_train, current_validation, manifest = prepare_split(
        ROOT / "data", tatoeba_path=ROOT / "data/tatoeba/training", db_eng_path=ROOT / "data",
        external_path=ROOT / "validation data"
    )
    overlap = overlapping_rows(current_train, current_validation)
    report["current_split"] = {
        "version": manifest["version"], "counts": manifest["counts"],
        "training_rows_overlapping_internal_validation": len(overlap),
    }
    if overlap:
        raise ValueError("Current split contains internal overlap")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline), indent=2, ensure_ascii=True))
