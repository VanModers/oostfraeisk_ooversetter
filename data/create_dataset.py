"""
Python port of oostfraeiskorg_dataset_creator/Program.cs
Generates parallel training data from the WFDOT.db dictionary database.

Outputs (written to the same directory as this script, i.e. data/):
  german.txt                      - German training sentences
  eastfrisian.txt                 - East Frisian (line-aligned with german.txt)
  english_db.txt                  - English phrases from DB (only rows with English)
  eastfrisian_for_english_db.txt  - East Frisian aligned with english_db.txt

Usage:
  python data/create_dataset.py
"""

import sqlite3
import random
from pathlib import Path

# ── Constants matching Program.cs ──────────────────────────────────────────
MAX_TEXT_LENGTH     = 5000
MANUAL_DATASET_MUL  = 1   # how many times to write each fan-teksten pair
MANUAL_DATASET_COMB_MUL = 2  # how many shuffle+smash passes for fan teksten
SQL_SEED            = 42
NOUN_LIMIT          = 5000
TEXT_ABBREVIATIONS  = {"bzw"}

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
REPO_DIR     = SCRIPT_DIR.parent
DB_PATH      = (REPO_DIR / "oostfraeiskorg_dataset_creator" /
                "oostfraeiskorg_dataset_creator" / "bin" / "Debug" /
                "net7.0" / "WFDOT.db")
FAN_TEKSTEN  = SCRIPT_DIR / "fan teksten"
OUT_GER      = SCRIPT_DIR / "german.txt"
OUT_FRS      = SCRIPT_DIR / "eastfrisian.txt"
OUT_ENG_DB   = SCRIPT_DIR / "english_db.txt"
OUT_FRS_ENG  = SCRIPT_DIR / "eastfrisian_for_english_db.txt"

# ── Abbreviation filter (mirrors FilterAllAbbreviations) ───────────────────
ABBREVIATIONS = [
    # German abbreviations
    "geogr.: ", "zool.: ", "weller.: ", "fig.: ",
    "Asrf.: ", "fehnkult.: ", "kinderr.: ", "räts.: ",
    "zungenbr.: ", "scherzh.: ", "imk.: ", "friesensp.: ",
    "botan.: ", "myk.: ", "naut.: ", "ziegel.: ",
    "kinderspr.: ", "med.: ", "geh.: ", "ugs.: ",
    "vulg.: ", "fis.: ", "iron.: ", "anatom.: ",
    "schimpfw.: ", "chem.: ", "scherz.: ", "landw.: ",
    "deichbaul.: ", "mühlenkult.: ", "abw.: ", "Nn.: ",
    "m. Vn.: ", "w. Vn.: ", "wn. Vn.: ", "zool: ", "Tr.: ",
    "m. V.: ", "maurerh.: ", "schimfpw.: ", "mod.: ",
    "Asrf: ", "tischlerh.: ", "schimpfwort.: ", "m.: ",
    "mühlenkult.:: ", "phys.: ", "m. Vn.n: ", "w. Vn: ",
    "deichbaul..: ", "anaton.: ", "kindersp.: ", "w. Vm.: ",
    "m.V.: ",
    # English abbreviations
    "childrenr.: ", "childrent.: ", "rid.: ",
    "m. fn.: ", "f. fn.: ", "w. fn.: ", "m.fn.: ", "f.fn.: ",
    "ln.: ", "Ln.: ",
    "exc.: ",
    "bricky.: ", "brick.: ",
    "swearw.: ",
    "millcult.: ", "fehncult.: ",
    "dikeconstr.: ", "dikeconst.: ", "dikecontr.: ", "dikecon.: ",
    "jok.: ",
    "beek.: ",
    "frisiansp.: ",
    "agricult.: ",
    "masoncr.: ", "mansoncr.: ",
    "derog.: ",
    "myc.: ",
    "col.: ",
    "assr.: ", "asrf.: ",
    "cost.: ",
    "onomat.: ",
    "math.: ",
    "carp.: ",
    "kulin.: ",
    "kloot.: ",
    "intrans. metaph.: ",
    "scherzh. zool.: ", "onomat. zool.: ",
    "jok. geogr.: ", "geogr. jok.: ", "jok. botan.: ",
    "exc. childrent.: ", "exc. jok.: ",
    "frisiansp. jok.: ",
    "col. childrent.: ",
    "anatom. vulg.: ",
    "form.: ",
    "f. Vm.: ",
    "naut..: ", "swearw..: ", "jok..: ", "exc..: ",
    "dikeconstr..: ", "rid..: ",
    "geog.: ",
    "chase someone away id.: ",
]

def filter_all_abbreviations(s: str) -> str:
    for abbr in ABBREVIATIONS:
        parts = s.split(abbr)
        if len(parts) > 1:
            s = parts[1]
    return s

def is_to_be_filtered(s: str) -> bool:
    low = s.lower()
    return (len(s) > 8 and "jööed" in low) or "jööden" in low or len(s) <= 3

def first_char_to_upper(s: str) -> str:
    return s[0].upper() + s[1:] if s else s

def add_point(s: str) -> str:
    return s if (s and s[-1] in ".?!") else s + "."

def is_valid_pair(src: str, tgt: str, allow_both_blank: bool = False) -> bool:
    """Return whether a parallel pair is usable for generated training data."""
    src_blank = not src.strip()
    tgt_blank = not tgt.strip()
    if src_blank and tgt_blank:
        return allow_both_blank
    return not src_blank and not tgt_blank

def cut_sentence_if_necessary(frs: str, ger: str) -> str:
    if "," in ger:
        max_len = int(len(frs) * 1.3)
        if len(ger) > max_len:
            idx = ger.rfind(",")
            return ger[:idx].strip()
    return ger.strip()

def cut_brackets(ger: str) -> str:
    parts = ger.split("(")
    if len(parts) > 1 and ('Rdw.: "' in parts[1] or 'Spr.: "' in parts[1] or 'say.: "' in parts[1] or 'id.: "' in parts[1]):
        return parts[1].split('"')[1]
    return parts[0]

def split_sentences(frs: str, ger: str) -> list[tuple[str, str]]:
    """Split on '. ' boundaries only for very long texts (>MAX_TEXT_LENGTH chars)."""
    if len(ger) > MAX_TEXT_LENGTH:
        frs_sents = frs.split(". ")
        ger_sents = ger.split(". ")
        size = min(len(frs_sents), len(ger_sents))

        # Merge German abbreviation splits (e.g. "bzw")
        i = 0
        while i < len(ger_sents) - 1:
            words = ger_sents[i].split()
            last = words[-1].rstrip(".") if words else ""
            if last in TEXT_ABBREVIATIONS:
                ger_sents[i] = ger_sents[i] + ". " + ger_sents[i + 1]
                ger_sents.pop(i + 1)
            else:
                i += 1

        result = []
        for j in range(size):
            f = frs_sents[j] if j < len(frs_sents) else ""
            g = ger_sents[j] if j < len(ger_sents) else ""
            if j < size - 1:
                f += "."
                g += "."
            result.append((f, g))
        return result
    else:
        return [(frs, ger)]

def split_meanings(frs: str, ger: str) -> list[tuple[str, str]]:
    """Expand comma-separated German meanings, each paired with the same FRS word."""
    return [(frs, g) for g in ger.split(", ")]

NOUN_TEMPLATES = [
    ("[WORD] is mooj.",          "[WORD] ist schön."),
    ("[WORD] is läif.",          "[WORD] ist lieb."),
    ("[WORD] is interessant.",   "[WORD] ist interessant."),
    ("[WORD] kan ik näit säin.", "[WORD] kann ich nicht sehen."),
]

def insert_into_template(frs: str, ger: str) -> list[tuple[str, str]]:
    frs = first_char_to_upper(frs)
    return [(tmpl_f.replace("[WORD]", frs), tmpl_g.replace("[WORD]", ger))
            for tmpl_f, tmpl_g in NOUN_TEMPLATES]

def seeded_shuffle(lst: list, seed: int) -> None:
    """Fisher-Yates shuffle matching C# Random(seed) behaviour (same algorithm)."""
    rng = random.Random(seed)
    n = len(lst)
    while n > 1:
        n -= 1
        k = rng.randint(0, n)          # randint is inclusive → matches C# Next(n+1)
        lst[k], lst[n] = lst[n], lst[k]

def get_db_rows(query: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"DB: {DB_PATH}")
    assert DB_PATH.exists(), f"Database not found: {DB_PATH}"

    with (open(OUT_GER,     "w", encoding="utf-8") as ger_out,
          open(OUT_FRS,     "w", encoding="utf-8") as frs_out,
          open(OUT_ENG_DB,  "w", encoding="utf-8") as eng_out,
          open(OUT_FRS_ENG, "w", encoding="utf-8") as frs_eng_out):

        # ── Step 1: Phrases — individual, length-sorted ──────────────────────
        print("\nStep 1: Phrases (length-sorted, individual)...")
        rows = get_db_rows(
            "SELECT Ostfriesisch, Deutsch, Englisch FROM WB "
            "WHERE Deutsch != '-' AND Wortart = 'Phrase' "
            "ORDER BY LENGTH(Ostfriesisch) DESC"
        )
        ger_count = eng_count = 0
        for row in rows:
            frs = row["Ostfriesisch"].replace("\n", " ")
            ger = row["Deutsch"].replace("\n", " ")
            eng = (row["Englisch"] or "").replace("\n", " ")

            ger = filter_all_abbreviations(cut_brackets(ger.split(";")[0]))
            eng = filter_all_abbreviations(cut_brackets(eng.split(";")[0]))

            if is_to_be_filtered(frs):
                continue

            ger = cut_sentence_if_necessary(frs, ger)
            if not is_valid_pair(ger, frs):
                continue
            sentences = split_sentences(frs, ger)

            for frs_s, ger_s in sentences:
                if not is_valid_pair(ger_s, frs_s):
                    continue
                ger_out.write(ger_s + "\n")
                frs_out.write(frs_s + "\n")
                ger_count += 1

            # English: write the original (unsplit) phrase pair when available.
            # Splitting DE/FRS on '. ' doesn't apply to EN, so we use the full pair.
            eng_clean = cut_sentence_if_necessary(frs, eng.strip())
            if eng_clean != "-" and is_valid_pair(eng_clean, frs):
                eng_out.write(eng_clean + "\n")
                frs_eng_out.write(frs + "\n")
                eng_count += 1

        print(f"  {ger_count} German/FRS pairs, {eng_count} English/FRS pairs")

        # ── Step 2: Phrases — smashed in pairs, random order ─────────────────
        print("\nStep 2: Phrases (random-order, smashed in pairs)...")
        rows = get_db_rows(
            f"SELECT Ostfriesisch, Deutsch, "
            f"((ID * {SQL_SEED}) % 100000) AS RandomOrder "
            f"FROM WB WHERE Deutsch != '-' AND Wortart = 'Phrase' "
            f"ORDER BY RandomOrder"
        )
        acc_frs = acc_ger = ""
        accumulating = False
        smash_count = 0
        for row in rows:
            frs = row["Ostfriesisch"].replace("\n", " ")
            ger = row["Deutsch"].replace("\n", " ")
            ger = filter_all_abbreviations(cut_brackets(ger.split(";")[0]))

            if is_to_be_filtered(frs):
                continue

            ger = cut_sentence_if_necessary(frs, ger)
            if not is_valid_pair(ger, frs):
                continue
            frs = add_point(first_char_to_upper(frs))
            ger = add_point(first_char_to_upper(ger))

            if not accumulating:
                # Write accumulated pair and start fresh
                acc_frs += frs
                acc_ger += ger
                ger_out.write(acc_ger + "\n")
                frs_out.write(acc_frs + "\n")
                smash_count += 1
                acc_frs = acc_ger = ""
            else:
                acc_frs += frs + " "
                acc_ger += ger + " "

            accumulating = not accumulating

        print(f"  {smash_count} smashed phrase pairs")

        # ── Step 3: Nouns — template sentences ───────────────────────────────
        print("\nStep 3: Nouns (templates)...")
        rows = get_db_rows(
            f"SELECT Ostfriesisch, Deutsch, Artikel, "
            f"((ID * {SQL_SEED}) % 100000) AS RandomOrder "
            f"FROM WB WHERE Deutsch != '-' AND Wortart = 'Substantiv' "
            f"ORDER BY RandomOrder LIMIT {NOUN_LIMIT}"
        )
        tmpl_count = 0
        for row in rows:
            frs = row["Ostfriesisch"].replace("\n", " ")
            ger = row["Deutsch"].replace("\n", " ")
            ger = filter_all_abbreviations(ger.split(";")[0].split("(")[0])

            for frs_m, ger_m in split_meanings(frs, ger):
                if is_valid_pair(ger_m, frs_m) and len(ger_m.strip()) < 25:
                    for frs_t, ger_t in insert_into_template(frs_m.strip(), ger_m.strip()):
                        ger_out.write(ger_t + "\n")
                        frs_out.write(frs_t + "\n")
                        tmpl_count += 1

        print(f"  {tmpl_count} noun template pairs")

        # ── Step 4: Fan teksten — individual + smashed ────────────────────────
        print("\nStep 4: Fan teksten (manual texts)...")
        frs_texts: list[str] = []
        ger_texts: list[str] = []

        if (FAN_TEKSTEN / "german.txt").exists():
            with (open(FAN_TEKSTEN / "german.txt",     encoding="utf-8") as gf,
                  open(FAN_TEKSTEN / "eastfrisian.txt", encoding="utf-8") as ff):
                for g, f in zip(gf, ff):
                    g, f = g.rstrip("\n"), f.rstrip("\n")
                    if is_valid_pair(g, f, allow_both_blank=True):
                        ger_texts.append(g)
                        frs_texts.append(f)
                        for _ in range(MANUAL_DATASET_MUL):
                            ger_out.write(g + "\n")
                            frs_out.write(f + "\n")

            print(f"  {len(ger_texts)} source pairs written")

            pairs_list = [[f, g] for f, g in zip(frs_texts, ger_texts)]
            for pass_num in range(MANUAL_DATASET_COMB_MUL):
                seeded_shuffle(pairs_list, SQL_SEED)
                acc_frs = acc_ger = ""
                accumulating = False
                smash_count = 0
                for pair in pairs_list:
                    f, g = pair[0].replace("\n", " "), pair[1].replace("\n", " ")
                    if not f or not g:
                        continue
                    g = cut_sentence_if_necessary(f, g)
                    f = add_point(first_char_to_upper(f))
                    g = add_point(first_char_to_upper(g))
                    if not accumulating:
                        acc_frs += f
                        acc_ger += g
                        ger_out.write(acc_ger + "\n")
                        frs_out.write(acc_frs + "\n")
                        smash_count += 1
                        acc_frs = acc_ger = ""
                    else:
                        acc_frs += f + " "
                        acc_ger += g + " "
                    accumulating = not accumulating
                print(f"  Smash pass {pass_num + 1}: {smash_count} pairs")
        else:
            print("  Fan teksten files not found, skipping.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nDone. Line counts:")
    for path in [OUT_GER, OUT_FRS, OUT_ENG_DB, OUT_FRS_ENG]:
        n = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.name}: {n:,} lines")


if __name__ == "__main__":
    main()
