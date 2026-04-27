"""
create_tatoeba_export.py

Creates a TSV file of East Frisian (frs) translations of German (deu) Tatoeba
sentences, ready to be submitted to Tatoeba for bulk import.

Source files:
  - data/fan teksten/german.txt      – German sentences (one per line)
  - data/fan teksten/eastfrisian.txt – Corresponding frs translations (one per line)
  - data/krektüren/deu copy.txt      – All German Tatoeba sentences with IDs
    (format: English_text TAB German_text TAB CC-BY attribution with sentence IDs)

Output (all written to data/tatoeba/):
  tatoeba_frs_export.tsv    – Tatoeba bulk-import file
                              Format: german_id TAB german_text TAB frs_text
  eastfrisian_tatoeba.txt   – frs sentences only (one per line, aligned)
  german_tatoeba.txt        – deu sentences only (one per line, aligned)
  english_tatoeba.txt       – eng sentences only (one per line, aligned)
                              (for multi-language training)

Rules:
  - Only sentence pairs whose German text appears verbatim in deu copy.txt are
    included (those are guaranteed to be real Tatoeba sentences).
  - Each unique German sentence appears at most once (first East Frisian
    translation encountered wins).
  - Empty lines in either source file are skipped.
  - Output is sorted by German sentence ID.
  - All plain-text files are line-aligned with each other.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent          # data/tatoeba/
REPO_DIR = BASE_DIR.parent.parent         # repository root
FAN_TEKSTEN_DIR = REPO_DIR / "data" / "fan teksten"
KREKTÜREN_DIR   = REPO_DIR / "data" / "krektüren"

GERMAN_FILE         = FAN_TEKSTEN_DIR / "german.txt"
FRISIAN_FILE        = FAN_TEKSTEN_DIR / "eastfrisian.txt"
TATOEBA_GERMAN_FILE = KREKTÜREN_DIR / "deu copy.txt"

# Output files (all in the same directory as this script)
OUTPUT_FILE  = BASE_DIR / "tatoeba_frs_export.tsv"
FRISIAN_OUT  = BASE_DIR / "eastfrisian_tatoeba.txt"
GERMAN_OUT   = BASE_DIR / "german_tatoeba.txt"
ENGLISH_OUT  = BASE_DIR / "english_tatoeba.txt"

# ---------------------------------------------------------------------------
# Step 1 – Build a lookup: german_text → german_sentence_id
#          from the Tatoeba German reference file.
#
# deu copy.txt format per line:
#   English_text\tGerman_text\tCC-BY 2.0 (France) Attribution: tatoeba.org
#       #<english_id> (<user>) & #<german_id> (<user>)
# ---------------------------------------------------------------------------

# Regex to extract the German sentence ID from the attribution string.
# The German ID is the one after the ampersand.
GERMAN_ID_RE = re.compile(r"& #(\d+)")

print("Reading Tatoeba German reference file …")
german_to_id: dict[str, int] = {}      # german_text → sentence ID
german_to_english: dict[str, str] = {} # german_text → english sentence

with open(TATOEBA_GERMAN_FILE, encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.rstrip("\n")
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue  # malformed or empty line

        english_text = parts[0]         # column 1 = English sentence
        german_text  = parts[1]         # column 2 = German sentence
        attribution  = parts[2]

        m = GERMAN_ID_RE.search(attribution)
        if not m:
            continue  # no German ID found – skip

        german_id = int(m.group(1))

        # Keep the first occurrence for each unique German sentence text
        if german_text not in german_to_id:
            german_to_id[german_text]      = german_id
            german_to_english[german_text] = english_text

print(f"  Loaded {len(german_to_id):,} unique German Tatoeba sentences.")

# ---------------------------------------------------------------------------
# Step 2 – Walk the parallel fan-teksten files and collect matches
# ---------------------------------------------------------------------------

print("Scanning fan teksten parallel files …")

output_pairs: list[tuple[int, str, str]] = []   # (german_id, german_text, frs_text)
seen_german: set[str] = set()                    # deduplication key

skipped_empty = 0
skipped_not_tatoeba = 0
skipped_duplicate = 0

with (
    open(GERMAN_FILE, encoding="utf-8") as gf,
    open(FRISIAN_FILE, encoding="utf-8") as ff,
):
    for lineno, (g_raw, f_raw) in enumerate(zip(gf, ff), start=1):
        german_text = g_raw.rstrip("\n")
        frisian_text = f_raw.rstrip("\n")

        # Skip pairs where either side is blank
        if not german_text.strip() or not frisian_text.strip():
            skipped_empty += 1
            continue

        # Check whether the German sentence is a known Tatoeba sentence
        if german_text not in german_to_id:
            skipped_not_tatoeba += 1
            continue

        # Deduplicate by German sentence text
        if german_text in seen_german:
            skipped_duplicate += 1
            continue

        seen_german.add(german_text)
        german_id = german_to_id[german_text]
        output_pairs.append((german_id, german_text, frisian_text))

print(f"  Pairs scanned        : {lineno:,}")
print(f"  Skipped (empty)      : {skipped_empty:,}")
print(f"  Skipped (not Tatoeba): {skipped_not_tatoeba:,}")
print(f"  Skipped (duplicate)  : {skipped_duplicate:,}")
print(f"  Included in output   : {len(output_pairs):,}")

# ---------------------------------------------------------------------------
# Step 3 – Sort by German sentence ID and write output
# ---------------------------------------------------------------------------

output_pairs.sort(key=lambda x: x[0])

print(f"Writing output files to: {BASE_DIR}")

# --- Tatoeba bulk-import TSV ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("# East Frisian (frs) translations of German (deu) Tatoeba sentences\n")
    out.write("# Format: german_sentence_id<TAB>german_sentence_text<TAB>frs_translation_text\n")
    out.write("# Generated from: data/fan teksten/  (verified against data/krektüren/deu copy.txt)\n")
    out.write(f"# Total pairs: {len(output_pairs)}\n")
    out.write("#\n")
    for german_id, german_text, frisian_text in output_pairs:
        out.write(f"{german_id}\t{german_text}\t{frisian_text}\n")

# --- Plain-text aligned files ---
with (
    open(FRISIAN_OUT, "w", encoding="utf-8") as frs_out,
    open(GERMAN_OUT,  "w", encoding="utf-8") as deu_out,
    open(ENGLISH_OUT, "w", encoding="utf-8") as eng_out,
):
    for german_id, german_text, frisian_text in output_pairs:
        frs_out.write(frisian_text + "\n")
        deu_out.write(german_text  + "\n")
        eng_out.write(german_to_english.get(german_text, "") + "\n")

print("Done.")
print()
print("Summary")
print("-------")
print(f"  {OUTPUT_FILE.name:<30} {len(output_pairs):,} pairs")
print(f"  {FRISIAN_OUT.name:<30} {len(output_pairs):,} lines")
print(f"  {GERMAN_OUT.name:<30} {len(output_pairs):,} lines")
print(f"  {ENGLISH_OUT.name:<30} {len(output_pairs):,} lines")
print(f"  ID range: {output_pairs[0][0]:,} – {output_pairs[-1][0]:,}")
