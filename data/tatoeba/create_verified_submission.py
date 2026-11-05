"""Create a Tatoeba submission TSV containing only reviewed sentence pairs.

The verifier records one-based row indices in ``verifyer/verified_indices.txt``.
Rows which still need work are also listed in
``verifyer/sentences-to-correct.txt``. This script selects verified rows from
the corrected, aligned ``german_tatoeba.txt`` and
``eastfrisian_tatoeba.txt`` files, excludes every still-flagged row, and uses
``tatoeba_frs_export.tsv`` only to obtain the corresponding Tatoeba sentence
IDs. It writes a headerless UTF-8 TSV in Tatoeba's list-with-translations
format::

    German sentence ID<TAB>German sentence<TAB>East Frisian translation

The first non-comment row of ``tatoeba_frs_export.tsv`` is index 1, matching
``german_tatoeba.txt``, ``eastfrisian_tatoeba.txt``, and the verifier.

Run from any directory::

    python data/tatoeba/create_verified_submission.py

By default, exactly 3,000 pairs are written to
``data/tatoeba/tatoeba_frs_verified_3000.tsv``.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "tatoeba_frs_export.tsv"
DEFAULT_GERMAN = BASE_DIR / "german_tatoeba.txt"
DEFAULT_FRISIAN = BASE_DIR / "eastfrisian_tatoeba.txt"
DEFAULT_VERIFIED = BASE_DIR / "verifyer" / "verified_indices.txt"
DEFAULT_CORRECTIONS = BASE_DIR / "verifyer" / "sentences-to-correct.txt"
DEFAULT_OUTPUT = BASE_DIR / "tatoeba_frs_verified_3000.tsv"

CORRECTION_INDEX_RE = re.compile(r"^(?:Line|ID)\s+(\d+)\s+\|")


@dataclass(frozen=True)
class ExportRow:
    german_id: int
    german_text: str
    frisian_text: str


def read_export_rows(path: Path) -> list[ExportRow]:
    """Read the three-column project export, ignoring comments and blanks."""
    rows: list[ExportRow] = []
    with path.open(encoding="utf-8") as source:
        for file_line, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{file_line}: expected 3 tab-separated fields, "
                    f"found {len(parts)}"
                )

            german_id_text, german_text, frisian_text = parts
            try:
                german_id = int(german_id_text)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{file_line}: invalid German sentence ID "
                    f"{german_id_text!r}"
                ) from error

            if not german_text.strip() or not frisian_text.strip():
                raise ValueError(f"{path}:{file_line}: sentence text must not be empty")

            rows.append(ExportRow(german_id, german_text, frisian_text))

    if not rows:
        raise ValueError(f"{path}: no export rows found")
    return rows


def read_aligned_sentences(path: Path) -> list[str]:
    """Read an aligned sentence file without stripping meaningful spaces."""
    with path.open(encoding="utf-8") as source:
        return [line.rstrip("\r\n") for line in source]


def use_corrected_aligned_text(
    export_rows: list[ExportRow],
    german_sentences: list[str],
    frisian_sentences: list[str],
) -> list[ExportRow]:
    """Combine export IDs with the corrected aligned German/Frisian text."""
    counts = {
        "export": len(export_rows),
        "German": len(german_sentences),
        "East Frisian": len(frisian_sentences),
    }
    if len(set(counts.values())) != 1:
        details = ", ".join(f"{name}={count:,}" for name, count in counts.items())
        raise ValueError(f"aligned source files have different row counts: {details}")

    corrected_rows: list[ExportRow] = []
    for index, (export_row, german_text, frisian_text) in enumerate(
        zip(export_rows, german_sentences, frisian_sentences), start=1
    ):
        if export_row.german_text != german_text:
            raise ValueError(
                f"row {index}: German text does not match the Tatoeba ID export"
            )
        if not german_text.strip() or not frisian_text.strip():
            raise ValueError(f"row {index}: aligned sentence text must not be empty")
        corrected_rows.append(
            ExportRow(export_row.german_id, german_text, frisian_text)
        )
    return corrected_rows


def read_verified_indices(path: Path) -> set[int]:
    """Read positive integer indices, allowing blank and comment lines."""
    indices: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for file_line, raw_line in enumerate(source, start=1):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                index = int(value)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{file_line}: expected a row index, found {value!r}"
                ) from error
            if index < 1:
                raise ValueError(f"{path}:{file_line}: row indices must be positive")
            indices.add(index)
    return indices


def read_correction_indices(path: Path) -> set[int]:
    """Read unresolved row indices from the verifier's correction log."""
    if not path.exists():
        return set()

    indices: set[int] = set()
    with path.open(encoding="utf-8") as source:
        for file_line, raw_line in enumerate(source, start=1):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            match = CORRECTION_INDEX_RE.match(value)
            if not match:
                raise ValueError(
                    f"{path}:{file_line}: cannot read correction row index"
                )
            indices.add(int(match.group(1)))
    return indices


def select_rows(
    rows: list[ExportRow],
    verified_ids: set[int],
    correction_ids: set[int],
    count: int,
) -> tuple[list[ExportRow], list[int]]:
    """Return the first ``count`` verified, non-flagged rows by row order.

    ``verified_ids`` / ``correction_ids`` are Tatoeba German sentence IDs.
    Selection follows the (ID-sorted) row order of ``rows``.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    row_ids = {row.german_id for row in rows}
    unknown = sorted((verified_ids | correction_ids) - row_ids)
    if unknown:
        preview = ", ".join(map(str, unknown[:10]))
        raise ValueError(f"unknown Tatoeba sentence ID(s): {preview}")

    eligible_ids = verified_ids - correction_ids
    if len(eligible_ids) < count:
        raise ValueError(
            f"requested {count} corrected pairs, but only "
            f"{len(eligible_ids)} are currently eligible"
        )

    selected_rows: list[ExportRow] = []
    selected_ids: list[int] = []
    for row in rows:
        if row.german_id in eligible_ids:
            selected_rows.append(row)
            selected_ids.append(row.german_id)
            if len(selected_rows) >= count:
                break

    german_ids = [row.german_id for row in selected_rows]
    if len(german_ids) != len(set(german_ids)):
        raise ValueError("selected rows contain duplicate German sentence IDs")

    pairs = [(row.german_text, row.frisian_text) for row in selected_rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("selected rows contain duplicate sentence pairs")

    return selected_rows, selected_ids


def write_submission(path: Path, rows: list[ExportRow]) -> None:
    """Atomically write a headerless, three-column, UTF-8 TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(
                    f"{row.german_id}\t{row.german_text}\t{row.frisian_text}\n"
                )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3000, help="pairs to write")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="three-column export used for Tatoeba sentence IDs",
    )
    parser.add_argument(
        "--german",
        type=Path,
        default=DEFAULT_GERMAN,
        help="corrected aligned German sentences",
    )
    parser.add_argument(
        "--frisian",
        type=Path,
        default=DEFAULT_FRISIAN,
        help="corrected aligned East Frisian sentences",
    )
    parser.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_rows = read_export_rows(args.source)
    rows = use_corrected_aligned_text(
        export_rows,
        read_aligned_sentences(args.german),
        read_aligned_sentences(args.frisian),
    )
    verified_ids = read_verified_indices(args.verified)
    correction_ids = read_correction_indices(args.corrections)
    selected_rows, selected_ids = select_rows(
        rows, verified_ids, correction_ids, args.count
    )
    write_submission(args.output, selected_rows)

    excluded_verified = verified_ids & correction_ids
    print(f"Source rows: {len(rows):,}")
    print(f"Unique verified IDs: {len(verified_ids):,}")
    print(f"Unresolved correction IDs: {len(correction_ids):,}")
    print(f"Verified rows excluded as unresolved: {len(excluded_verified):,}")
    print(
        f"Wrote {len(selected_rows):,} corrected pairs "
        f"(IDs {selected_ids[0]:,}-{selected_ids[-1]:,})"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
