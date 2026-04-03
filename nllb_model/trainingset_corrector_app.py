import zipfile
from pathlib import Path

import gradio as gr


PENDING_GERMAN_FILE = Path("auto_translated/german.txt")
PENDING_EASTFRISIAN_FILE = Path("auto_translated/eastfrisian.txt")
CORRECTED_DIR = Path("corrected")
CORRECTED_GERMAN_FILE = CORRECTED_DIR / "german_corrected.txt"
CORRECTED_EASTFRISIAN_FILE = CORRECTED_DIR / "eastfrisian_corrected.txt"
ZIP_FILE = CORRECTED_DIR / "corrected_translations.zip"


def read_lines(path: Path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if lines:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def load_pairs():
    german_lines = read_lines(PENDING_GERMAN_FILE)
    eastfrisian_lines = read_lines(PENDING_EASTFRISIAN_FILE)

    pair_count = min(len(german_lines), len(eastfrisian_lines))
    pairs = list(zip(german_lines[:pair_count], eastfrisian_lines[:pair_count]))

    # Keep files aligned if one side is longer.
    if len(german_lines) != len(eastfrisian_lines):
        write_lines(PENDING_GERMAN_FILE, german_lines[:pair_count])
        write_lines(PENDING_EASTFRISIAN_FILE, eastfrisian_lines[:pair_count])

    return pairs


def save_pending_pairs(pairs):
    german_lines = [g for g, _ in pairs]
    eastfrisian_lines = [f for _, f in pairs]
    write_lines(PENDING_GERMAN_FILE, german_lines)
    write_lines(PENDING_EASTFRISIAN_FILE, eastfrisian_lines)


def append_corrected(german_text: str, eastfrisian_text: str):
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    with CORRECTED_GERMAN_FILE.open("a", encoding="utf-8") as ger_file, CORRECTED_EASTFRISIAN_FILE.open(
        "a", encoding="utf-8"
    ) as frs_file:
        ger_file.write(german_text + "\n")
        frs_file.write(eastfrisian_text + "\n")


def build_zip_bundle():
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if CORRECTED_GERMAN_FILE.exists():
            archive.write(CORRECTED_GERMAN_FILE, arcname=CORRECTED_GERMAN_FILE.name)
        if CORRECTED_EASTFRISIAN_FILE.exists():
            archive.write(CORRECTED_EASTFRISIAN_FILE, arcname=CORRECTED_EASTFRISIAN_FILE.name)
        if PENDING_GERMAN_FILE.exists():
            archive.write(PENDING_GERMAN_FILE, arcname=PENDING_GERMAN_FILE.name)
        if PENDING_EASTFRISIAN_FILE.exists():
            archive.write(PENDING_EASTFRISIAN_FILE, arcname=PENDING_EASTFRISIAN_FILE.name)
    return str(ZIP_FILE)


def next_pair(pairs):
    if not pairs:
        return "No more pending pairs.", "", "Finished. All pending pairs are processed.", build_zip_bundle()

    german_text, eastfrisian_text = pairs[0]
    status = f"Pending pairs: {len(pairs)}"
    return german_text, eastfrisian_text, status, build_zip_bundle()


def approve_translation(german_text, eastfrisian_text, state_pairs):
    pairs = list(state_pairs or [])
    if not pairs:
        german_next, eastfrisian_next, status, zip_path = next_pair(pairs)
        return german_next, eastfrisian_next, status, zip_path, pairs

    current_german, _ = pairs[0]
    if german_text.strip() != current_german.strip():
        status = "Current German line was changed. Keep it unchanged and edit only East Frisian text."
        zip_path = build_zip_bundle()
        return current_german, eastfrisian_text, status, zip_path, pairs

    append_corrected(current_german, eastfrisian_text.strip())
    pairs.pop(0)
    save_pending_pairs(pairs)

    german_next, eastfrisian_next, status, zip_path = next_pair(pairs)
    return german_next, eastfrisian_next, status, zip_path, pairs


def skip_translation(state_pairs):
    pairs = list(state_pairs or [])
    if not pairs:
        german_next, eastfrisian_next, status, zip_path = next_pair(pairs)
        return german_next, eastfrisian_next, status, zip_path, pairs

    pairs.pop(0)
    save_pending_pairs(pairs)

    german_next, eastfrisian_next, status, zip_path = next_pair(pairs)
    return german_next, eastfrisian_next, status, zip_path, pairs


def initialize():
    pairs = load_pairs()
    german_text, eastfrisian_text, status, zip_path = next_pair(pairs)
    return german_text, eastfrisian_text, status, zip_path, pairs


with gr.Blocks(title="NLLB Trainingset Corrector") as demo:
    gr.Markdown("# German -> East Frisian Correction Tool")
    gr.Markdown("Review generated pairs, edit East Frisian, and export corrected files.")

    pairs_state = gr.State([])

    german_text = gr.Textbox(label="German sentence", interactive=False, lines=3)
    eastfrisian_text = gr.Textbox(label="East Frisian translation", lines=3)
    status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        approve_button = gr.Button("Approve and Next", variant="primary")
        skip_button = gr.Button("Skip", variant="secondary")

    zip_download = gr.File(label="Download corrected bundle")

    approve_button.click(
        fn=approve_translation,
        inputs=[german_text, eastfrisian_text, pairs_state],
        outputs=[german_text, eastfrisian_text, status_text, zip_download, pairs_state],
    )

    skip_button.click(
        fn=skip_translation,
        inputs=[pairs_state],
        outputs=[german_text, eastfrisian_text, status_text, zip_download, pairs_state],
    )

    demo.load(
        fn=initialize,
        inputs=None,
        outputs=[german_text, eastfrisian_text, status_text, zip_download, pairs_state],
    )


demo.launch()
