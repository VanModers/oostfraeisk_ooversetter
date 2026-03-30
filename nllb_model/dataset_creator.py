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

    return [(g, f) for g, f in pairs if g and f]


class NLLBTranslationDataset(Dataset):
    """Pre-tokenized dataset for NLLB fine-tuning with optional bidirectional pairs."""

    def __init__(self, pairs, tokenizer, max_length=256, bidirectional=True):
        self.data = []
        total = len(pairs)

        for i, (ger, frs) in enumerate(pairs):
            # de -> frs
            tokenizer.src_lang = DEU_LANG
            tokenizer.tgt_lang = FRS_LANG
            enc = tokenizer(
                ger, text_target=frs,
                truncation=True, max_length=max_length, return_tensors=None
            )
            self.data.append(enc)

            if bidirectional:
                # frs -> de
                tokenizer.src_lang = FRS_LANG
                tokenizer.tgt_lang = DEU_LANG
                enc = tokenizer(
                    frs, text_target=ger,
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


def get_dataset(path, tokenizer, val_size=1000, bidirectional=True):
    """Load parallel data, split into train/val, then tokenize bidirectionally."""
    raw_pairs = load_parallel_pairs(path)
    print(f"Loaded {len(raw_pairs)} parallel pairs from {path}")

    # Split raw pairs BEFORE creating bidirectional data to prevent leakage
    rng = random.Random(42)
    indices = list(range(len(raw_pairs)))
    rng.shuffle(indices)
    val_indices = set(indices[:val_size])

    train_pairs = [raw_pairs[i] for i in range(len(raw_pairs)) if i not in val_indices]
    val_pairs = [raw_pairs[i] for i in range(len(raw_pairs)) if i in val_indices]
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")

    print("Tokenizing training data...")
    train_ds = NLLBTranslationDataset(train_pairs, tokenizer, max_length=256, bidirectional=bidirectional)
    print("Tokenizing validation data...")
    val_ds = NLLBTranslationDataset(val_pairs, tokenizer, max_length=256, bidirectional=bidirectional)

    return {"train": train_ds, "validation": val_ds}