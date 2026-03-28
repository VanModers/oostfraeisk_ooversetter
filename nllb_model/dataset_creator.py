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


def add_frs_lang(tokenizer, model=None):
    """Register frs_Latn as a new NLLB language.

    * First call (with model): adds the token, resizes embeddings, copies
      nld_Latn weights into the new frs_Latn row.
    * Later calls (without model, e.g. inference): just patches the
      lang_code_to_id / id_to_lang_code maps so the tokenizer can use
      frs_Latn as src_lang / tgt_lang.
    """
    frs_id = tokenizer.convert_tokens_to_ids(FRS_LANG)

    if frs_id == tokenizer.unk_token_id:
        # Token does not exist yet — add it as a special token
        tokenizer.add_tokens([FRS_LANG], special_tokens=True)
        frs_id = tokenizer.convert_tokens_to_ids(FRS_LANG)

        if model is not None:
            nld_id = tokenizer.convert_tokens_to_ids(NLD_DONOR)
            old_size = old_shared.weight.shape[0]
            new_size = len(tokenizer)
            old_shared = model.model.shared

            # Do NOT use resize_token_embeddings — it replaces model.shared
            # with a new object but leaves encoder/decoder embed_tokens pointing
            # to the old one.  M2M100Model.forward then detects the mismatch
            # (shared is not decoder.embed_tokens) and passes BOTH
            # decoder_input_ids and decoder_inputs_embeds to the decoder,
            # which crashes.  Instead we create ONE new embedding and assign
            # it to all three places manually.
            new_shared = old_shared.__class__(
                new_size, old_shared.embedding_dim,
                old_shared.padding_idx, embed_scale=old_shared.embed_scale,
            )
            with torch.no_grad():
                new_shared.weight[:old_size] = old_shared.weight.data
                new_shared.weight[frs_id] = new_shared.weight[nld_id].clone()

            model.model.shared = new_shared
            model.model.encoder.embed_tokens = new_shared
            model.model.decoder.embed_tokens = new_shared

            # Resize lm_head separately
            old_lm = model.lm_head
            new_lm = torch.nn.Linear(old_lm.in_features, new_size, bias=False)
            with torch.no_grad():
                new_lm.weight[:old_size] = old_lm.weight.data
                new_lm.weight[frs_id] = new_lm.weight[nld_id].clone()
            model.lm_head = new_lm

            model.config.vocab_size = new_size

            assert model.model.shared is model.model.decoder.embed_tokens
            print(f"Added {FRS_LANG} (id={frs_id}), embedding copied from {NLD_DONOR} (id={nld_id})")

    # Patch the lang_code_to_id / id_to_lang_code mapping so src_lang works
    if hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id[FRS_LANG] = frs_id
        tokenizer.id_to_lang_code[frs_id] = FRS_LANG
    else:
        # Newer TokenizersBackend — build mapping from vocab and set it
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
