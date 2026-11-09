# Ooversetter

The Ooversetter is a machine-translation project for East Frisian Low Saxon. The main model is [`VanModers114/East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model), a multilingual model fine-tuned from Meta's [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M).

The NLLB model translates in all four supported directions:

- German → East Frisian
- East Frisian → German
- English → East Frisian
- East Frisian → English

Model weights are published on Hugging Face and are not stored in this repository.

## Oostfräisk wersjoon

Dit repository daajt däi dóótensatsen un däi code föör däi Oostfräisk Ooversetter undhollen. Dat hoovdmodel is nuu dat multilinguóóel [`East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model). Dit model wur fan Meta siin `nllb-200-distilled-600M` fine-tuned un kan tüsken Düütsk, Oostfräisk un Engelsk ooversetten.

Däi modelgewichten bünt up Hugging Face spaichert un näit meer direkt in dit repository t' finnen. Däi ollerder Marian-models föör Düütsk → Oostfräisk un Oostfräisk → Düütsk bliivent as äigen modellen up Hugging Face beskikbor.

## Published models

| Model | Directions | Status |
| --- | --- | --- |
| [`East_Frisian_NLLB_Model`](https://huggingface.co/VanModers114/East_Frisian_NLLB_Model) | German ↔ East Frisian, English ↔ East Frisian | **Main model** |
| [`opus-mt-de-frs`](https://huggingface.co/VanModers114/opus-mt-de-frs) | German → East Frisian | Legacy Marian model |
| [`opus-mt-frs-de`](https://huggingface.co/VanModers114/opus-mt-frs-de) | East Frisian → German | Legacy Marian model |

## NLLB implementation

Training starts from `facebook/nllb-200-distilled-600M`. Because NLLB does not provide a dedicated East Frisian language token, `nllb_model/dataset_creator.py` registers `frs_Latn` and initializes its embedding from the Dutch `nld_Latn` embedding. German uses `deu_Latn`, and English uses `eng_Latn`.

The dataset is split and overlapping training rows are excluded before reverse-direction examples are created. Selection uses deterministic content hashes and a fixed seed. Training uses a maximum sequence length of 512 tokens.

The current trainer is configured for:

- up to 40 epochs, with early stopping after three evaluations without improvement;
- an effective training batch size of 64;
- bfloat16 training and fused AdamW;
- evaluation and checkpointing after every epoch;
- restoration of the checkpoint with the lowest validation loss.

## Training data

The training data combines dictionary phrases and noun templates, Tatoeba
translations, additional parallel texts, and synthetic German translations of
original East Frisian passages.

| Parallel source | Aligned pairs | Unique nonblank pairs |
| --- | ---: | ---: |
| Generated German/East Frisian corpus | 117,774 | 117,267 |
| Tatoeba German/East Frisian | 10,699 | 10,694 |
| Tatoeba English/East Frisian | 10,699 | 10,546 |
| Dictionary English/East Frisian phrases | 27,501 | 27,396 |
| Additional German/East Frisian | 4,877 | 4,813 |
| Additional English/East Frisian | 8 | 8 |
| Synthetic German/East Frisian | 2,694 | 2,694 |

Unique counts use normalized bilingual pairs, excluding blanks. Sources overlap:
the generated German corpus already includes Tatoeba and additional translations,
so source counts are not additive.

The retained split contains **153,092 training pairs** and **1,700 internal
validation pairs**. Every pair is included in both directions: **306,184 training
examples** and **3,400 internal validation examples**. Of the training pairs,
**2,689 (1.76%)** come from the synthetic corpus. Synthetic pairs are training-only.
The separate reviewed set in `validation data/` contains **300 German/East Frisian
pairs**.

The shared splitter removes normalized duplicate pairs and excludes detected
validation overlap in either language, including inside concatenated paragraphs.
The trainer saves input hashes and exclusions in
`nllb_frs_model/data_split_manifest.json`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `nllb_model/` | Main NLLB dataset loader, trainer, validator, CLI, and Gradio application |
| `data/` | Checked-in aligned corpora and dataset-generation utilities |
| `data/create_dataset.py` | Generates German/East Frisian and dictionary English/East Frisian files from `WFDOT.db`, Tatoeba and additional texts |
| `data/tatoeba/` | Tatoeba-aligned German, English, and East Frisian data and verification utilities |
| `validation data/` | Separate 300-pair German–East Frisian evaluation set |
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

PyTorch 2.6 or newer is required by the environment preflight. The configured bfloat16 and fused-optimizer training path requires a compatible GPU. Training also requires Transformers and Accelerate. Training downloads the base NLLB model and writes the result to `./nllb_frs_model`, which is ignored by Git.

On a Slurm cluster, `train_nllb.sbatch` runs the environment check, validates the currently published Hugging Face model, trains a new local model, and validates the result.

### Regenerate dictionary-derived data

`data/create_dataset.py` requires the local dictionary database at:

```text
data/dictionary/WFDOT.db
```

The database is intentionally ignored by Git. Regeneration overwrites `data/german.txt`, `data/eastfrisian.txt`, `data/english_db.txt`, and `data/eastfrisian_for_english_db.txt`:

```bash
python data/create_dataset.py
```

## Validate a model

The validator evaluates both German/East Frisian directions on the reviewed
300-pair corpus and reports teacher-forced token accuracy, token-weighted loss,
SacreBLEU BLEU and chrF. It uses greedy decoding and requires SacreBLEU:

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
