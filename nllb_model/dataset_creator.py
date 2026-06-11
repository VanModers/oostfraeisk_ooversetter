import os
import random
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


def load_parallel_pairs(path):
    """Load parallel german.txt / eastfrisian.txt and return list of (ger, frs) tuples."""
    ger_path = os.path.join(path, "german.txt")
    frs_path = os.path.join(path, "eastfrisian.txt")

    with open(ger_path, "r", encoding="utf-8") as gf, \
         open(frs_path, "r", encoding="utf-8") as ff:
        pairs = [(g.strip(), f.strip()) for g, f in zip(gf, ff)]

    return [(g, f) for g, f in pairs if is_usable_pair(g, f, allow_both_blank=True)]


def load_tatoeba_eng_pairs(tatoeba_path):
    """Load parallel english_tatoeba.txt / eastfrisian_tatoeba.txt and return list of (eng, frs) tuples."""
    eng_path = os.path.join(tatoeba_path, "english_tatoeba.txt")
    frs_path = os.path.join(tatoeba_path, "eastfrisian_tatoeba.txt")

    with open(eng_path, "r", encoding="utf-8") as ef, \
         open(frs_path, "r", encoding="utf-8") as ff:
        pairs = [(e.strip(), f.strip()) for e, f in zip(ef, ff)]

    return [(e, f) for e, f in pairs if is_usable_pair(e, f)]


def load_db_eng_pairs(path):
    """Load English-FRS pairs generated by data/create_dataset.py.

    Reads english_db.txt + eastfrisian_for_english_db.txt from *path*.
    These files are created by the Python dataset creator and contain
    ~26k dictionary phrases with English translations.
    """
    eng_path = os.path.join(path, "english_db.txt")
    frs_path = os.path.join(path, "eastfrisian_for_english_db.txt")

    with open(eng_path, "r", encoding="utf-8") as ef, \
         open(frs_path, "r", encoding="utf-8") as ff:
        pairs = [(e.strip(), f.strip()) for e, f in zip(ef, ff)]

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
                tatoeba_path=None, db_eng_path=None):
    """Load parallel data, split into train/val, then tokenize.

    Args:
        path: directory containing german.txt and eastfrisian.txt
        tokenizer: NLLB tokenizer with frs_Latn registered
        val_size: number of DEU-FRS pairs to hold out for validation
        bidirectional: also add reverse-direction pairs (FRS->DEU, FRS->ENG)
        tatoeba_path: optional directory with english_tatoeba.txt /
                      eastfrisian_tatoeba.txt (9k Tatoeba ENG-FRS pairs)
        db_eng_path: optional directory with english_db.txt /
                     eastfrisian_for_english_db.txt (~26k DB ENG-FRS pairs)
    """
    raw_pairs = load_parallel_pairs(path)
    print(f"Loaded {len(raw_pairs)} DEU-FRS parallel pairs from {path}")

    # Split raw pairs BEFORE creating bidirectional data to prevent leakage
    rng = random.Random(42)
    indices = list(range(len(raw_pairs)))
    rng.shuffle(indices)
    val_indices = set(indices[:val_size])

    deu_train = [raw_pairs[i] for i in range(len(raw_pairs)) if i not in val_indices]
    deu_val   = [raw_pairs[i] for i in range(len(raw_pairs)) if i in val_indices]
    print(f"DEU-FRS split: {len(deu_train)} train, {len(deu_val)} val")

    train_lang_pairs = make_lang_pairs(deu_train, DEU_LANG, FRS_LANG, bidirectional=bidirectional)
    val_lang_pairs   = make_lang_pairs(deu_val,   DEU_LANG, FRS_LANG, bidirectional=bidirectional)

    # Optionally add English <-> FRS pairs from Tatoeba
    if tatoeba_path is not None:
        eng_raw = load_tatoeba_eng_pairs(tatoeba_path)
        print(f"Loaded {len(eng_raw)} ENG-FRS Tatoeba pairs from {tatoeba_path}")

        eng_rng = random.Random(43)
        eng_indices = list(range(len(eng_raw)))
        eng_rng.shuffle(eng_indices)
        eng_val_size = min(200, len(eng_raw) // 10)
        eng_val_idx = set(eng_indices[:eng_val_size])

        eng_train = [eng_raw[i] for i in range(len(eng_raw)) if i not in eng_val_idx]
        eng_val   = [eng_raw[i] for i in range(len(eng_raw)) if i in eng_val_idx]
        print(f"ENG-FRS Tatoeba split: {len(eng_train)} train, {len(eng_val)} val")

        train_lang_pairs += make_lang_pairs(eng_train, ENG_LANG, FRS_LANG, bidirectional=bidirectional)
        val_lang_pairs   += make_lang_pairs(eng_val,   ENG_LANG, FRS_LANG, bidirectional=bidirectional)

    # Optionally add English <-> FRS pairs from the dictionary DB
    if db_eng_path is not None:
        db_eng_raw = load_db_eng_pairs(db_eng_path)
        print(f"Loaded {len(db_eng_raw)} ENG-FRS DB pairs from {db_eng_path}")

        db_rng = random.Random(44)
        db_indices = list(range(len(db_eng_raw)))
        db_rng.shuffle(db_indices)
        db_val_size = min(500, len(db_eng_raw) // 10)
        db_val_idx = set(db_indices[:db_val_size])

        db_eng_train = [db_eng_raw[i] for i in range(len(db_eng_raw)) if i not in db_val_idx]
        db_eng_val   = [db_eng_raw[i] for i in range(len(db_eng_raw)) if i in db_val_idx]
        print(f"ENG-FRS DB split: {len(db_eng_train)} train, {len(db_eng_val)} val")

        train_lang_pairs += make_lang_pairs(db_eng_train, ENG_LANG, FRS_LANG, bidirectional=bidirectional)
        val_lang_pairs   += make_lang_pairs(db_eng_val,   ENG_LANG, FRS_LANG, bidirectional=bidirectional)

    print("Tokenizing training data...")
    train_ds = NLLBTranslationDataset(train_lang_pairs, tokenizer, max_length=MAX_LENGTH)
    print("Tokenizing validation data...")
    val_ds   = NLLBTranslationDataset(val_lang_pairs,   tokenizer, max_length=MAX_LENGTH)

    return {"train": train_ds, "validation": val_ds}
