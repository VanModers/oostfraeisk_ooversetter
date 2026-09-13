import os
import json
from pathlib import Path
from data_split import prepare_split
from itertools import zip_longest
import torch
from torch.utils.data import Dataset

# East Frisian gets its own language code. The embedding is initialised from
# nld_Latn (Dutch) — the closest available NLLB language — so Dutch stays intact.
FRS_LANG = "frs_Latn"
DEU_LANG = "deu_Latn"
ENG_LANG = "eng_Latn"
NLD_DONOR = "nld_Latn"

# Token budget: NLLB architecture supports up to 1024, but 512 comfortably covers ~1000 chars.
MAX_LENGTH = 512



def is_usable_pair(src, tgt, allow_both_blank=False):
    """Keep aligned blanks only when the caller explicitly wants them."""
    src_blank = not src.strip()
    tgt_blank = not tgt.strip()
    if src_blank and tgt_blank:
        return allow_both_blank
    return not src_blank and not tgt_blank


def add_frs_lang(tokenizer, model=None, random_init=False):
    """Register frs_Latn as a new NLLB language.

    NLLB's weight matrices have 256206 rows but only 256204 tokens in the
    vocab, leaving rows 256204-256205 as spare slots.  We assign frs_Latn
    to row 256204 by adding it as a special token (no embedding resize
    needed since the row already exists).

    To prevent 'missing keys' on checkpoint reload we sync
    model.config.vocab_size with the tokenizer length so the Trainer's
    saved config and the weight shapes stay consistent.

    * First call (with model):
        - If random_init is False (default): copies nld_Latn embedding weights into the
          frs_Latn row so it has a meaningful starting point.
        - If random_init is True: initializes the frs_Latn row randomly.
    * All calls: patches the tokenizer's language-code maps so
      src_lang / tgt_lang = "frs_Latn" works.
    """
    # --- Add frs_Latn to tokenizer vocab if needed ---
    if FRS_LANG not in tokenizer.get_vocab():
        tokenizer.add_tokens([FRS_LANG], special_tokens=True)

    frs_id = tokenizer.convert_tokens_to_ids(FRS_LANG)

    # --- Seed the embedding row from Dutch or randomly (only when we have the model) ---
    if model is not None:
        emb_rows = model.get_input_embeddings().weight.shape[0]
        assert frs_id < emb_rows, (
            f"frs slot {frs_id} out of range (embedding has {emb_rows} rows)"
        )
        with torch.no_grad():
            inp_emb = model.get_input_embeddings()
            out_emb = model.get_output_embeddings()
            if random_init:
                torch.nn.init.normal_(inp_emb.weight[frs_id])
                if out_emb is not inp_emb:
                    torch.nn.init.normal_(out_emb.weight[frs_id])
                print(f"Randomly initialized {FRS_LANG} (id={frs_id}), matrix rows={emb_rows}, vocab_size={len(tokenizer)}")
            else:
                nld_id = tokenizer.convert_tokens_to_ids(NLD_DONOR)
                inp_emb.weight[frs_id] = inp_emb.weight[nld_id].clone()
                if out_emb is not inp_emb:
                    out_emb.weight[frs_id] = out_emb.weight[nld_id].clone()
                print(f"Seeded {FRS_LANG} (id={frs_id}) from {NLD_DONOR} (id={nld_id}), matrix rows={emb_rows}, vocab_size={len(tokenizer)}")

        # Keep config.vocab_size in sync so checkpoint reload works
        model.config.vocab_size = emb_rows

    # --- Patch the lang_code_to_id / id_to_lang_code maps ---
    if hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id[FRS_LANG] = frs_id
        if hasattr(tokenizer, "id_to_lang_code"):
            tokenizer.id_to_lang_code[frs_id] = FRS_LANG
    else:
        import re
        lang_code_to_id = {}
        id_to_lang_code = {}
        for token, idx in tokenizer.get_vocab().items():
            if re.match(r"^[a-z]{2,3}_[A-Z][a-z]{3}$", token):
                lang_code_to_id[token] = idx
                id_to_lang_code[idx] = token
        lang_code_to_id[FRS_LANG] = frs_id
        id_to_lang_code[frs_id] = FRS_LANG
        tokenizer.lang_code_to_id = lang_code_to_id
        tokenizer.id_to_lang_code = id_to_lang_code

    return frs_id


def read_aligned_pairs(src_path, tgt_path):
    """Read every row; reject unequal lengths instead of silently losing data."""
    pairs = []
    with open(src_path, encoding="utf-8") as src, open(tgt_path, encoding="utf-8") as tgt:
        for line_number, (s, t) in enumerate(zip_longest(src, tgt), 1):
            if s is None or t is None:
                raise ValueError(
                    f"Parallel file length mismatch at line {line_number}: "
                    f"{src_path} / {tgt_path}"
                )
            pairs.append((s.strip(), t.strip()))
    return pairs


def load_parallel_pairs(path):
    """Load parallel german.txt / eastfrisian.txt and return list of (ger, frs) tuples."""
    ger_path = os.path.join(path, "german.txt")
    frs_path = os.path.join(path, "eastfrisian.txt")

    pairs = read_aligned_pairs(ger_path, frs_path)

    return [(g, f) for g, f in pairs if is_usable_pair(g, f, allow_both_blank=True)]


def load_tatoeba_eng_pairs(tatoeba_path):
    """Load parallel english_tatoeba.txt / eastfrisian_tatoeba.txt and return list of (eng, frs) tuples."""
    eng_path = os.path.join(tatoeba_path, "english_tatoeba.txt")
    frs_path = os.path.join(tatoeba_path, "eastfrisian_tatoeba.txt")

    pairs = read_aligned_pairs(eng_path, frs_path)

    return [(e, f) for e, f in pairs if is_usable_pair(e, f)]


def load_db_eng_pairs(path):
    """Load English-FRS pairs generated by data/create_dataset.py.

    Reads english_db.txt + eastfrisian_for_english_db.txt from *path*.
    These files are created by the Python dataset creator and contain
    ~26k dictionary phrases with English translations.
    """
    eng_path = os.path.join(path, "english_db.txt")
    frs_path = os.path.join(path, "eastfrisian_for_english_db.txt")

    pairs = read_aligned_pairs(eng_path, frs_path)

    return [(e, f) for e, f in pairs if is_usable_pair(e, f)]


def make_lang_pairs(raw_pairs, src_lang, tgt_lang, bidirectional=True):
    """Convert (src_text, tgt_text) raw pairs to (src, tgt, src_lang, tgt_lang) 4-tuples."""
    result = []
    for src, tgt in raw_pairs:
        result.append((src, tgt, src_lang, tgt_lang))
        if bidirectional:
            result.append((tgt, src, tgt_lang, src_lang))
    return result


class NLLBTranslationDataset(Dataset):
    """Pre-tokenized dataset for NLLB fine-tuning."""

    def __init__(self, pairs, tokenizer, max_length=MAX_LENGTH):
        """pairs: list of (src_text, tgt_text, src_lang, tgt_lang) tuples."""
        self.data = []
        total = len(pairs)

        for i, (src, tgt, src_lang, tgt_lang) in enumerate(pairs):
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            enc = tokenizer(
                src, text_target=tgt,
                truncation=True, max_length=max_length, return_tensors=None
            )
            self.data.append(enc)

            if (i + 1) % 20000 == 0:
                print(f"  Tokenized {i + 1}/{total} pairs...")

        print(f"Dataset: {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(path, tokenizer, val_size=1000, bidirectional=True,
                tatoeba_path=None, db_eng_path=None, external_path=None,
                manifest_path=None):
    """Create the shared purged split before reversing directions/tokenizing.

    The manifest records original corpus line numbers, content IDs, exclusions
    and input hashes. Generated files remain unchanged.
    """
    train, validation, manifest = prepare_split(
        path, val_size, tatoeba_path, db_eng_path, external_path
    )
    print("Split:", manifest["counts"])
    if manifest_path is not None:
        destination = Path(manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def expand(rows):
        result = []
        for src, tgt, src_lang, tgt_lang, _, _ in rows:
            result.extend(make_lang_pairs([(src, tgt)], src_lang, tgt_lang, bidirectional))
        return result

    print("Tokenizing training data...")
    train_ds = NLLBTranslationDataset(expand(train), tokenizer, max_length=MAX_LENGTH)
    print("Tokenizing validation data...")
    val_ds = NLLBTranslationDataset(expand(validation), tokenizer, max_length=MAX_LENGTH)
    return {"train": train_ds, "validation": val_ds}
