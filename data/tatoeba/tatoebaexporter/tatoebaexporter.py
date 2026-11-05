"""Semi-automatic insertion of East Frisian translations into Tatoeba.

The Tatoeba API is read-only. The app therefore copies the East Frisian sentence
to the clipboard, opens the Tatoeba sentence in the browser, and then uses the API
to check whether the translation has arrived. See README.md.
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import tatoeba_api

BASE_DIR = Path(__file__).resolve().parent
TATOEBA_DIR = BASE_DIR.parent
STATE_FILE = BASE_DIR / "exporter_state.json"
LANG = "frs"
PREFETCH_AHEAD = 5
PREFETCH_DELAY_SECONDS = 2.0

STATUS_INSERTED = "inserted"
STATUS_UNVERIFIED = "inserted (unverified)"
STATUS_EXISTING = "already present"
STATUS_SKIPPED = "skipped"
STATUS_READY = (
    "No East Frisian translation on Tatoeba yet. "
    "Step 1: “Copy & open Tatoeba”."
)

sys.path.insert(0, str(TATOEBA_DIR))
import create_verified_submission as submission


def read_pairs():
    """Only verified sentences that are not flagged for correction, as in the Tatoeba export."""
    export_rows = submission.read_export_rows(submission.DEFAULT_SOURCE)
    rows = submission.use_corrected_aligned_text(
        export_rows,
        submission.read_aligned_sentences(submission.DEFAULT_GERMAN),
        submission.read_aligned_sentences(submission.DEFAULT_FRISIAN),
    )
    verified_ids = submission.read_verified_indices(submission.DEFAULT_VERIFIED)
    correction_ids = submission.read_correction_indices(submission.DEFAULT_CORRECTIONS)
    selected_rows, _ = submission.select_rows(rows, verified_ids, correction_ids, None)
    return [(row.german_id, row.german_text, row.frisian_text) for row in selected_rows]


LEGACY_STATUS = {
    "bereits vorhanden": STATUS_EXISTING,
    "übersprungen": STATUS_SKIPPED,
}


def migrate_status(status):
    """Map status values written by the German version of the app."""
    if status in LEGACY_STATUS:
        return LEGACY_STATUS[status]
    if status.startswith("eingef"):
        return STATUS_UNVERIFIED if "(" in status else STATUS_INSERTED
    return status


def load_state(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        state = json.load(source)
    for item in state.values():
        item["status"] = migrate_status(item["status"])
    save_state(path, state)
    return state


def save_state(path, state):
    with path.open("w", encoding="utf-8") as target:
        json.dump(state, target, ensure_ascii=False, indent=2)


class ExporterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tatoeba Exporter (East Frisian)")
        self.geometry("840x440")
        self.minsize(660, 400)

        self.pairs = read_pairs()
        self.state = load_state(STATE_FILE)
        self.checked = set()
        self.current = None
        self.busy = False
        self.prefetching = False
        self.results = queue.Queue()

        self._build_ui()
        self._show_next()
        self.after(100, self._poll_results)

    # -- User interface --------------------------------------------------

    def _build_ui(self):
        self.stats_var = tk.StringVar()
        self.id_var = tk.StringVar()
        self.de_var = tk.StringVar()
        self.frs_var = tk.StringVar()
        self.status_var = tk.StringVar()

        tk.Label(
            self, textvariable=self.status_var, fg="#333", wraplength=800,
            justify="left", anchor="w",
        ).pack(side="bottom", fill="x", padx=12, pady=(0, 10))

        buttons = tk.Frame(self)
        buttons.pack(side="bottom", fill="x", padx=12, pady=6)
        self.copy_button = tk.Button(
            buttons, text="1. Copy & open Tatoeba", width=28,
            command=self._copy_and_open,
        )
        self.verify_button = tk.Button(
            buttons, text="2. Inserted – verify & next", width=28,
            command=self._verify,
        )
        self.skip_button = tk.Button(
            buttons, text="Skip", width=14, command=self._skip,
        )
        self.copy_button.pack(side="left", padx=(0, 6))
        self.verify_button.pack(side="left", padx=(0, 6))
        self.skip_button.pack(side="left")

        tk.Label(self, textvariable=self.stats_var, fg="#555").pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(self, textvariable=self.id_var, font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 0))

        tk.Label(self, text="German (Tatoeba):", fg="#555").pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(
            self, textvariable=self.de_var, font=("Segoe UI", 13), wraplength=800, justify="left", anchor="w",
        ).pack(anchor="w", padx=12)

        tk.Label(self, text="East Frisian (to insert):", fg="#555").pack(anchor="w", padx=12, pady=(10, 0))
        tk.Label(
            self, textvariable=self.frs_var, font=("Segoe UI", 13, "bold"), wraplength=800, justify="left", anchor="w",
        ).pack(anchor="w", padx=12)

    def _update_buttons(self):
        state = "normal" if (self.current is not None and not self.busy) else "disabled"
        for button in (self.copy_button, self.verify_button, self.skip_button):
            button.config(state=state)

    def _set_status(self, text):
        self.status_var.set(text)

    def _update_stats(self):
        pending = self._pending()
        inserted = sum(1 for item in self.state.values() if item["status"].startswith(STATUS_INSERTED))
        existing = sum(1 for item in self.state.values() if item["status"] == STATUS_EXISTING)
        skipped = sum(1 for item in self.state.values() if item["status"] == STATUS_SKIPPED)
        self.stats_var.set(
            f"Open: {len(pending)} of {len(self.pairs)}   ·   inserted: {inserted}   ·   "
            f"already present: {existing}   ·   skipped: {skipped}"
        )
        return pending

    # -- Workflow --------------------------------------------------------

    def _pending(self):
        return [pair for pair in self.pairs if str(pair[0]) not in self.state]

    def _show_next(self):
        pending = self._update_stats()
        if not pending:
            self.current = None
            self.id_var.set("")
            self.de_var.set("All sentences have been processed.")
            self.frs_var.set("")
            self._set_status("Done.")
            self._update_buttons()
            return

        self.current = pending[0]
        sentence_id, de_text, frs_text = self.current
        self.id_var.set(f"Tatoeba ID {sentence_id}")
        self.de_var.set(de_text)
        self.frs_var.set(frs_text)
        if sentence_id in self.checked:
            self._set_status(STATUS_READY)
        else:
            self._set_status("Checking this sentence on Tatoeba …")
            self._run_async(
                lambda: tatoeba_api.get_direct_translations(sentence_id, LANG),
                lambda texts, error: self._on_current_check(sentence_id, texts, error),
            )
        self._prefetch(pending[1:1 + PREFETCH_AHEAD])
        self._update_buttons()

    def _on_current_check(self, sentence_id, texts, error):
        if self.current is None or self.current[0] != sentence_id:
            return
        if error:
            self._set_status(f"Could not check Tatoeba ({error}). You can still copy.")
            return
        self.checked.add(sentence_id)
        if texts:
            self._finish(STATUS_EXISTING)
        else:
            self._set_status(STATUS_READY)

    def _copy_and_open(self):
        sentence_id, _, frs_text = self.current
        self.clipboard_clear()
        self.clipboard_append(frs_text.strip())
        self.update()
        webbrowser.open(tatoeba_api.sentence_url(sentence_id))
        self._set_status(
            "East Frisian sentence is in the clipboard. On Tatoeba: add a translation, "
            "choose the language “Eastern Frisian / frs”, paste it and save. "
            "Then click “Inserted – verify & next”."
        )

    def _verify(self):
        sentence_id = self.current[0]
        self._set_status("Checking Tatoeba …")
        self._run_async(
            lambda: tatoeba_api.get_direct_translations(sentence_id, LANG),
            self._on_verify,
        )

    def _on_verify(self, texts, error):
        if error:
            self._set_status(f"Verification failed ({error}). Please try again.")
        elif texts:
            self._finish(STATUS_INSERTED)
        elif messagebox.askyesno(
            "Not found yet",
            "There is no East Frisian translation of this sentence on Tatoeba yet.\n\n"
            "Mark it as done anyway?\n"
            "Yes = done, No = stays open.",
        ):
            self._finish(STATUS_UNVERIFIED)
        else:
            self._set_status(
                "Not found yet. Please save the translation on Tatoeba and check again."
            )

    def _skip(self):
        self._finish(STATUS_SKIPPED)

    def _finish(self, status):
        sentence_id = self.current[0]
        self.state[str(sentence_id)] = {
            "status": status,
            "date": datetime.now().isoformat(timespec="seconds"),
        }
        save_state(STATE_FILE, self.state)
        self._show_next()

    # -- Background work (network calls do not block the window) ---------

    def _run_async(self, func, callback):
        """Run `func` in the background; `callback` runs afterwards in the main thread."""
        def worker():
            try:
                result, error = func(), None
            except Exception as exc:
                result, error = None, str(exc) or exc.__class__.__name__

            def done(result, error):
                self.busy = False
                callback(result, error)

            self.results.put((done, result, error))

        self.busy = True
        self._update_buttons()
        threading.Thread(target=worker, daemon=True).start()

    def _prefetch(self, pairs):
        """Check the next open sentences in the background, slowly and only a few."""
        sentence_ids = [sentence_id for sentence_id, _, _ in pairs if sentence_id not in self.checked]
        if not sentence_ids or self.prefetching:
            return
        self.prefetching = True
        threading.Thread(target=self._prefetch_worker, args=(sentence_ids,), daemon=True).start()

    def _prefetch_worker(self, sentence_ids):
        for sentence_id in sentence_ids:
            time.sleep(PREFETCH_DELAY_SECONDS)
            try:
                texts = tatoeba_api.get_direct_translations(sentence_id, LANG)
            except tatoeba_api.TatoebaRateLimited:
                self.results.put((self._on_prefetch_done, True, None))
                return
            except Exception:
                texts = None
            self.results.put((self._on_prefetch_item, (sentence_id, texts), None))
        self.results.put((self._on_prefetch_done, False, None))

    def _on_prefetch_item(self, result, error):
        sentence_id, texts = result
        if texts is None:
            return
        self.checked.add(sentence_id)
        if texts and str(sentence_id) not in self.state:
            self.state[str(sentence_id)] = {
                "status": STATUS_EXISTING,
                "date": datetime.now().isoformat(timespec="seconds"),
            }
            save_state(STATE_FILE, self.state)
            if self.current and self.current[0] == sentence_id:
                self._show_next()
                return
        self._update_stats()

    def _on_prefetch_done(self, rate_limited, error):
        self.prefetching = False
        if rate_limited:
            self._set_status(
                "Tatoeba is throttling requests. The advance check was stopped; "
                "sentences will be checked one by one when shown."
            )

    def _poll_results(self):
        while True:
            try:
                callback, result, error = self.results.get_nowait()
            except queue.Empty:
                break
            callback(result, error)
            self._update_buttons()
        self.after(100, self._poll_results)


if __name__ == "__main__":
    ExporterApp().mainloop()
