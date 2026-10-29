"""Decode feedback mail into aligned text files, oldest submission first.

Run: python data/additional/extract_feedback.py [mailbox.mbox ...]
Defaults to raw/*.mbox and writes raw/extracted/ for review before importing.
"""

import argparse
from datetime import timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import mailbox
from pathlib import Path
import re

RAW = Path(__file__).resolve().parent / "raw"


def parse_pair(body):
    match = re.fullmatch(
        r"\s*(Deutsch|Oostfräisk|English):\s*(.*?)\n\s*"
        r"(Deutsch|Oostfräisk|English):\s*(.*)", body, re.S,
    )
    if match:
        first, a, second, b = match.groups()
        fields = {first: a, second: b}
        language = "en" if "English" in fields else "de"
        source = fields["English" if language == "en" else "Deutsch"]
        target = fields["Oostfräisk"]
    else:
        parts = re.split(r"\n\s*OOVERSETTEN:\s*", body, maxsplit=1)
        if len(parts) != 2:
            raise ValueError("Unrecognized feedback format")
        language = "de"
        source, target = parts
    clean = lambda text: " ".join(text.replace("[FEELER]", "").split())
    return language, clean(source), clean(target)


def extract_feedback(paths):
    records = []
    parser = BytesParser(policy=policy.default)
    for path in paths:
        box = mailbox.mbox(path, create=False, factory=parser.parse)
        try:
            for message in box:
                if message["Subject"] not in ("Translation Success Report", "Translation Issue Report"):
                    continue
                body = message.get_body(preferencelist=("plain",))
                if body is None:
                    raise ValueError(f"No plain-text feedback in {path}")
                date = parsedate_to_datetime(message["Date"])
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                records.append((date, *parse_pair(body.get_content())))
        finally:
            box.close()
    records.sort(key=lambda row: row[0])
    seen, unique = set(), []
    for record in records:
        pair = record[1:]
        if pair not in seen:
            seen.add(pair)
            unique.append(record)
    return unique


def write_pairs(records, output):
    output.mkdir(parents=True, exist_ok=True)
    for language, source_name, target_name in [
        ("de", "german.txt", "eastfrisian.txt"),
        ("en", "english.txt", "eastfrisian_for_english.txt"),
    ]:
        pairs = [(s, t) for _, lang, s, t in records if lang == language]
        for name, column in [(source_name, 0), (target_name, 1)]:
            (output / name).write_text("".join(pair[column] + "\n" for pair in pairs), encoding="utf-8")
        print(f"{language}: {len(pairs)} pairs -> {output / source_name}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mailboxes", nargs="*", type=Path)
    cli.add_argument("--output", type=Path, default=RAW / "extracted")
    args = cli.parse_args()
    paths = args.mailboxes or sorted(RAW.glob("*.mbox"))
    if not paths:
        cli.error(f"No .mbox files found in {RAW}")
    write_pairs(extract_feedback(paths), args.output)
