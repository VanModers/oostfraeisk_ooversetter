import unittest
import tempfile
import json
from pathlib import Path

from data_split import normalize, overlapping_rows, split_records


def row(source, target, language="deu_Latn", corpus="de", line=1):
    return (source, target, language, "frs_Latn", corpus, line)


class SplitTests(unittest.TestCase):
    def test_cross_language_reuse_and_punctuation(self):
        held = [row("Guten Morgen!", "Moin, Welt!")]
        candidates = [row("Good morning", "moin welt", "eng_Latn"),
                      row("GUTEN MORGEN", "Other translation"),
                      row("Fresh sentence", "Fresh target")]
        self.assertEqual(overlapping_rows(candidates, held), {0, 1})

    def test_augmentation_both_containment_directions_and_fragments(self):
        held = [row("A long protected sentence. A different held out sentence.", "Reserved target")]
        candidates = [row("A long protected sentence. Some unrelated ending.", "First"),
                      row("A different held out sentence.", "Second"),
                      row("Prefix. A long protected sentence. A different held out sentence. Suffix.", "Third"),
                      row("No shared content at all.", "Fourth")]
        self.assertEqual(overlapping_rows(candidates, held), {0, 1, 2})

    def test_token_boundaries_and_language_separation(self):
        held = [row("an", "unique")]
        self.assertEqual(overlapping_rows([row("another", "different"),
                                          row("an", "other", "eng_Latn")], held), set())

    def test_short_phrases_shared_by_two_different_concatenations(self):
        held = [row("Hi. Protected ending.", "Reserved target")]
        candidates = [row("Hi. Different ending.", "Other target")]
        self.assertEqual(overlapping_rows(candidates, held), {0})

    def test_split_preserves_empty_examples_and_removes_duplicates_and_overlap(self):
        rows = [row(f"source{i}", f"target{i}", line=i + 1) for i in range(20)]
        rows += [row("source0!", "TARGET0", line=21), row("", "", line=22)]
        external = [row("source19", "target19")]
        tr, va, manifest = split_records(rows, {"de": 3}, external)
        self.assertEqual(len(va), 3)
        self.assertEqual(manifest["counts"]["excluded"], {"duplicate": 1, "external_overlap": 1})
        self.assertIn(rows[-1], tr)
        self.assertNotIn(rows[-1], va)
        self.assertFalse(overlapping_rows(tr, va + external))
        tr2, va2, _ = split_records(list(reversed(rows[:20])), {"de": 3}, external)
        self.assertEqual({r[:4] for r in va}, {r[:4] for r in va2})

    def test_overlarge_quota_fails_instead_of_empty_training(self):
        with self.assertRaises(ValueError):
            split_records([row("one", "two")], {"de": 2})

    def test_unicode_normalization(self):
        self.assertEqual(normalize("CAFÉ!"), normalize("cafe\u0301"))

    def test_dataset_uses_purged_split_and_writes_manifest_before_reversing(self):
        from dataset_creator import get_dataset

        class Tokenizer:
            def __call__(self, text, text_target, **kwargs):
                return {"input_ids": [1], "labels": [2]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "german.txt").write_text("\n".join(f"source{i}" for i in range(20)), encoding="utf-8")
            (root / "eastfrisian.txt").write_text("\n".join(f"target{i}" for i in range(20)), encoding="utf-8")
            manifest_path = root / "output" / "split.json"
            dataset = get_dataset(root, Tokenizer(), val_size=3, manifest_path=manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(dataset["validation"]), 6)
            self.assertEqual(len(dataset["train"]), 34)
            self.assertEqual(manifest["counts"]["validation"], 3)
            self.assertEqual(len(manifest["input_sha256"]), 2)


if __name__ == "__main__":
    unittest.main()
