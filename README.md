# Ooversetter

The Ooversetter is a machine-translation project for East Frisian Low Saxon. The main model is [`VanModers114/East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model), a multilingual model fine-tuned from Meta's [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M).

The NLLB model translates in all four supported directions:

- German → East Frisian
- East Frisian → German
- English → East Frisian
- East Frisian → English

Model weights are published on Hugging Face and are not stored in this repository.

## Oostfräisk wersjoon

Dit repository daajt däi dóótensatsen un däi code för däi Oostfräisk Ooversetter undhollen. Dat hoovdmodel is nuu dat multilinguóóel [`East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model). Dit model wur fan Meta siin `nllb-200-distilled-600M` fine-tuned un kan tüsken Düütsk, Oostfräisk un Engelsk ooversetten.

Däi modelgewichten bünt up Hugging Face spaichert un näit meer direkt in dit repository t' finnen. Däi ollerder Marian-models för Düütsk → Oostfräisk un Oostfräisk → Düütsk bliivent as äigen modellen up Hugging Face beskikbor.

## Published models

| Model | Directions | Status |
| --- | --- | --- |
| [`East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model) | German ↔ East Frisian, English ↔ East Frisian | **Main model** |
| [`opus-mt-de-frs`](https://huggingface.co/VanModers114/opus-mt-de-frs) | German → East Frisian | Legacy Marian model |
| [`opus-mt-frs-de`](https://huggingface.co/VanModers114/opus-mt-frs-de) | East Frisian → German | Legacy Marian model |

## NLLB implementation

Training starts from `facebook/nllb-200-distilled-600M`. Because NLLB does not provide a dedicated East Frisian language token, `nllb_model/dataset_creator.py` registers `frs_Latn` and initializes its embedding from the Dutch `nld_Latn` embedding. German uses `deu_Latn`, and English uses `eng_Latn`.

The dataset is split before reverse-direction examples are created, preventing a pair in one direction from entering training while its reverse direction appears in validation. The split is reproducible, using fixed random seeds for each data source. Training uses a maximum sequence length of 512 tokens.

The current trainer is configured for:

- up to 40 epochs, with early stopping after three evaluations without improvement;
- an effective training batch size of 64;
- bfloat16 training and fused AdamW;
- evaluation and checkpointing after every epoch;
- restoration of the checkpoint with the lowest validation loss.

## Training data

At the current checked-in dataset revision, the NLLB pipeline loads:

| Parallel source | Raw aligned pairs | Training pairs | Validation pairs |
| --- | ---: | ---: | ---: |
| German–East Frisian generated corpus | 115,518 | 114,518 | 1,000 |
| Tatoeba English–East Frisian corpus | 10,018 | 9,818 | 200 |
| Dictionary English–East Frisian phrases | 27,164 | 26,664 | 500 |
| **Total** | **152,700** | **151,000** | **1,700** |

Every retained pair is added in both directions. The final tokenized datasets therefore contain **302,000 training examples** and **3,400 validation examples**.

These figures describe the checked-in generated files and the latest recorded NLLB training run.

The 115,518 German–East Frisian pairs are generated from:

| Component | Pairs |
| --- | ---: |
| Individual dictionary phrases | 39,573 |
| Combined dictionary phrases | 19,787 |
| Template sentences generated from dictionary nouns | 26,564 |
| Original manually translated/text-corpus rows | 14,798 |
| Combined text-corpus augmentation | 14,796 |
| **Total** | **115,518** |

The local WFDOT dictionary contains 111,379 entries. The generator uses 39,573 usable phrase entries and 4,855 contributing nouns from a deterministic 5,000-noun sample. The accepted noun meanings are inserted into four simple sentence templates. Dictionary phrases with English translations produce the additional 27,164 English–East Frisian pairs.

The Tatoeba English corpus reuses 10,018 East Frisian translations from the manually translated German corpus. It adds English alignments rather than 10,018 entirely new East Frisian sentences.

The loader does not deduplicate aligned pairs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `nllb_model/` | Main NLLB dataset loader, trainer, validator, CLI, and Gradio application |
| `data/` | Checked-in aligned corpora and dataset-generation utilities |
| `data/create_dataset.py` | Generates German/East Frisian and dictionary English/East Frisian files from `WFDOT.db` and `data/fan teksten/` |
| `data/tatoeba/` | Tatoeba-aligned German, English, and East Frisian data and verification utilities |
| `validation data/` | Separate 100-pair German–East Frisian evaluation set |
| `train_nllb.sbatch` | HPC/Slurm preflight, training, and validation workflow |
| `pretrained_de_frs/` | Legacy German → East Frisian Marian training code |
| `pretrained_frs_de/` | Legacy East Frisian → German Marian training code |

Directories such as `data/auto_translated/`, `data/krektüren/`, `data/fan LLM dóóten/`, and `data/minecraft/` are data-preparation or review workspaces. The NLLB trainer does not load them directly.

## Run the translator

Run commands from the repository root. The Gradio application downloads the main model from Hugging Face:

```bash
python nllb_model/app.py
```

The required Python environment must provide PyTorch, Transformers, Gradio, Hugging Face Hub, and the Hugging Face Spaces helper used by the application. A CUDA GPU is optional for inference.

## Train the NLLB model

The trainer reads the checked-in parallel text files directly:

```bash
python nllb_model/check_training_env.py
python nllb_model/trainer.py
```

PyTorch 2.6 or newer is required by the environment preflight. The configured bfloat16 and fused-optimizer training path requires a compatible GPU. Training downloads the base NLLB model and writes the result to `./nllb_frs_model`, which is ignored by Git.

On a Slurm cluster, `train_nllb.sbatch` runs the environment check, validates the currently published Hugging Face model, trains a new local model, and validates the result.

### Regenerate dictionary-derived data

`data/create_dataset.py` requires the local dictionary database at:

```text
oostfraeiskorg_dataset_creator/oostfraeiskorg_dataset_creator/bin/Debug/net7.0/WFDOT.db
```

The database is intentionally ignored by Git. Regeneration overwrites `data/german.txt`, `data/eastfrisian.txt`, `data/english_db.txt`, and `data/eastfrisian_for_english_db.txt`:

```bash
python data/create_dataset.py
```

## Validate a model

The validator evaluates German → East Frisian on the separate 100-pair validation corpus and reports teacher-forced token accuracy, average loss, and generated-text SacreBLEU:

```bash
python nllb_model/validator.py \
  --model-path VanModers114/East_Frisian_NLLB_Model \
  --model-label "Published East Frisian NLLB model"
```

The historical chart and `validation_results.txt` concern the legacy Marian German → East Frisian models; they are not NLLB evaluation results.

## NLLB citation

The main model is based on the NLLB-200 project:

> Costa-jussà, M. R., et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672).

Base model: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M).

## License

The repository source code is licensed under the [Mozilla Public License 2.0](LICENSE). Published models and third-party datasets may have separate licenses or terms; consult their respective model cards and source documentation.
