"""Regression checks for parallel-file data loss (no model download)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset_creator import read_aligned_pairs, load_parallel_pairs, load_tatoeba_eng_pairs, load_db_eng_pairs

SPEC = importlib.util.spec_from_file_location(
    "generator", Path(__file__).resolve().parents[1] / "data" / "create_dataset.py"
)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class AlignmentTests(unittest.TestCase):
    def test_evaluation_overlap_inside_paragraph_on_either_side(self):
        evaluation = (["Held out."], ["Reserved translation."])
        self.assertTrue(generator.overlaps_evaluation("Before. Held out. After.", "Other.", evaluation))
        self.assertTrue(generator.overlaps_evaluation("Other.", "Reserved translation. After.", evaluation))
        self.assertFalse(generator.overlaps_evaluation("Unrelated.", "Other.", evaluation))
        self.assertFalse(generator.overlaps_evaluation("Unrelated.", "Other.", ([], [])))

    def test_either_side_longer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.txt", Path(directory) / "b.txt"
            for left, right in [("one\ntwo\n", "one\n"), ("one\n", "one\ntwo\n")]:
                a.write_text(left, encoding="utf-8")
                b.write_text(right, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "line 2"):
                    read_aligned_pairs(a, b)

    def test_all_loaders_preserve_existing_blank_policy(self):
        cases = [
            (load_parallel_pairs, "german.txt", "eastfrisian.txt", True),
            (load_tatoeba_eng_pairs, "english_tatoeba.txt", "eastfrisian_tatoeba.txt", False),
            (load_db_eng_pairs, "english_db.txt", "eastfrisian_for_english_db.txt", False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for loader, src, tgt, keep_blanks in cases:
                (root / src).write_text(" Hallo \n\norphan\nend", encoding="utf-8")
                (root / tgt).write_text(" Moin \n\n\näen", encoding="utf-8")
                expected = [("Hallo", "Moin")] + ([("", "")] if keep_blanks else []) + [("end", "äen")]
                self.assertEqual(loader(root), expected)
                (root / tgt).write_text("Moin\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    loader(root)

    def test_generator_rejects_bad_input_before_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "german.txt").write_text("one\ntwo\n", encoding="utf-8")
            (root / "eastfrisian.txt").write_text("one\n", encoding="utf-8")
            output = root / "output.txt"
            output.write_text("keep me", encoding="utf-8")
            with patch.multiple(generator, TEXT_SOURCES=(root,), DB_PATH=output, OUT_GER=output):
                with self.assertRaises(ValueError):
                    generator.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_generator_requires_both_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                generator.load_text_pairs(root)
            (root / "eastfrisian.txt").write_text("Moin\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                generator.load_text_pairs(root)
            (root / "german.txt").write_text("Hallo\n", encoding="utf-8")
            self.assertEqual(generator.load_text_pairs(root), [("Hallo", "Moin")])

    def test_text_sources_preserve_duplicates_and_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "german.txt").write_text("Hallo\nHallo\norphan\n\n", encoding="utf-8")
            (root / "eastfrisian.txt").write_text("Moin\nMoin\n\n\n", encoding="utf-8")
            self.assertEqual(generator.load_text_pairs(root), [
                ("Hallo", "Moin"), ("Hallo", "Moin"), ("orphan", ""), ("", ""),
            ])

    def test_english_regeneration_preserves_translation_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            training.mkdir()
            (root / "reference.tsv").write_text("Hello\tHallo\tattribution\n", encoding="utf-8")
            (training / "german.txt").write_text("Hallo\nHallo\n", encoding="utf-8")
            (training / "eastfrisian.txt").write_text("Moin\nMoin!\n", encoding="utf-8")
            generator.regenerate_tatoeba_english(root)
            self.assertEqual((training / "english_tatoeba.txt").read_text(encoding="utf-8"), "Hello\nHello\n")
            self.assertEqual((training / "eastfrisian_tatoeba.txt").read_text(encoding="utf-8"), "Moin\nMoin!\n")


if __name__ == "__main__":
    unittest.main()
