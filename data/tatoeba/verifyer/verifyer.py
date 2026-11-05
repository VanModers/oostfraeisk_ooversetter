"""Interactive validator for German/East Frisian sentence pairs.

Each East Frisian sentence is read aloud automatically through the East
Frisian TTS Hugging Face Space (the same endpoint the oostfraeisk.org website
uses). Press [p] to repeat the audio, [m] to toggle the automatic playback
and [b] to go back to the previous sentence and undo its stored result.
"""

import json
import os
import re
import sys
import tempfile
import threading
import urllib.request

# --- Text-to-speech (Hugging Face Space) configuration ---
TTS_API_URL = "https://vanmoders114-east-frisian-tts.hf.space/gradio_api/call/predict"
TTS_SPACE_ID = "VanModers114/East_Frisian_TTS"
TTS_RUNTIME_URL = f"https://huggingface.co/api/spaces/{TTS_SPACE_ID}/runtime"
TTS_TIMEOUT_SECONDS = 300

# Playback speed. 1.0 = original tempo; lower values (e.g. 0.9) read slower.
PLAYBACK_SPEED = 1.0

# How many upcoming sentences to synthesize in the background ahead of time.
PREFETCH_AHEAD = 5

# Keep at most this many synthesized clips in memory.
CACHE_LIMIT = 50

# The Space is private, so a read token is required. Pass it as the first
# command-line argument, set HF_TOKEN, or create verifyer_config.json next to
# this script with {"hf_token": "hf_..."}.
TTS_TOKEN = ""
AUTO_READ = True


def use_safe_console():
    """Make sure East Frisian characters print on legacy (cp1252) consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def get_total_lines(file_path):
    """Helper to count the total number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def get_specific_line(file_path, line_number):
    """Helper to fetch a specific line number from a file on disk."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i == line_number:
                    return line.strip()
    except Exception as e:
        return f"Error reading file: {e}"
    return ""


def read_tsv_rows(file_path):
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for raw in f:
                line = raw.rstrip('\n').rstrip('\r')
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                try:
                    sid = int(parts[0].strip())
                except ValueError:
                    continue
                rows.append((sid, parts[1], parts[2]))
    except Exception as e:
        print(f"Error reading TSV: {e}")
    return rows


def read_verified_indices(log_path):
    """Load the verified line indices, ignoring comments and blank lines."""
    indices = set()
    if not os.path.exists(log_path):
        return indices
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                indices.add(int(line.strip()))
            except ValueError:
                continue
    return indices


def append_line(path, text):
    """Append one line to a log file and flush it right away."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"{text}\n")


def sort_verified_log(log_path):
    """Rewrite the verified log with its IDs in ascending order, comments on top."""
    if not os.path.exists(log_path):
        return
    comments, ids = [], set()
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                comments.append(stripped)
                continue
            try:
                ids.add(int(stripped))
            except ValueError:
                continue
    with open(log_path, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(f"{comment}\n")
        for sid in sorted(ids):
            f.write(f"{sid}\n")


def append_verified_id(log_path, index):
    """Record a verified Tatoeba ID and keep the log sorted."""
    append_line(log_path, index)
    sort_verified_log(log_path)


def remove_verified_index(log_path, index):
    """Remove `index` from the verified log, keeping comments and line order."""
    if not os.path.exists(log_path):
        return
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in lines:
            try:
                if int(line.strip()) == index:
                    continue
            except ValueError:
                pass
            f.write(line)


def remove_error_entry(error_path, index):
    """Remove the correction entry for `index` from the error log.

    Uses the same leading pattern as create_verified_submission.py
    (``Line <index> | ...``), so only the matching row is dropped.
    """
    if not os.path.exists(error_path):
        return
    pattern = re.compile(r"^(?:Line|ID)\s+(\d+)\s+\|")
    with open(error_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(error_path, 'w', encoding='utf-8') as f:
        for line in lines:
            match = pattern.match(line.strip())
            if match and int(match.group(1)) == index:
                continue
            f.write(line)


def undo_verified_index(log_path, error_path, verified_indices, index):
    """Undo the stored result for `index` so it can be reviewed again."""
    verified_indices.discard(index)
    remove_verified_index(log_path, index)
    remove_error_entry(error_path, index)

# ---------------------------------------------------------------------------
# Text-to-speech helpers
# ---------------------------------------------------------------------------

def load_tts_token():
    """Resolve the Hugging Face token from CLI, environment or config file."""
    global TTS_TOKEN
    if len(sys.argv) > 1:
        TTS_TOKEN = sys.argv[1].strip()
        return
    TTS_TOKEN = os.environ.get("HF_TOKEN", "").strip()
    if TTS_TOKEN:
        return
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "verifyer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                TTS_TOKEN = str(json.load(f).get("hf_token", "")).strip()
        except Exception as e:
            print(f"WARNING: Could not read {config_path}: {e}")


def _tts_request(url, data=None, accept="application/json"):
    headers = {"Accept": accept, "Authorization": f"Bearer {TTS_TOKEN}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers)


def _tts_ensure_running():
    """Wake a sleeping Space before synthesizing (same flow as the website)."""
    try:
        with urllib.request.urlopen(_tts_request(TTS_RUNTIME_URL), timeout=60) as response:
            stage = json.load(response).get("stage")
    except Exception as e:
        print(f"  TTS: runtime check failed ({e}); trying the Space anyway.")
        return
    if stage == "RUNNING":
        return
    print(f"  TTS: Space is '{stage}', requesting restart...")
    request = _tts_request(f"https://huggingface.co/api/spaces/{TTS_SPACE_ID}/restart")
    request.get_method = lambda: "POST"
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except Exception as e:
        print(f"  TTS: restart request failed ({e}).")


def extract_json_from_event_stream(text):
    """Return the JSON payload of the 'complete' event in a Gradio stream."""
    match = re.search(r"event: complete\s*\ndata: (.*)", text)
    return match.group(1).strip() if match else None


def fetch_tts_audio(text):
    """Synthesize `text` via the Space and return the WAV bytes, or None."""
    if not TTS_TOKEN:
        raise RuntimeError("No Hugging Face token configured for TTS.")

    _tts_ensure_running()

    payload = json.dumps({"data": [text]}).encode("utf-8")
    with urllib.request.urlopen(
            _tts_request(TTS_API_URL, data=payload), timeout=TTS_TIMEOUT_SECONDS) as response:
        event_id = json.load(response).get("event_id")
    if not event_id:
        raise RuntimeError("The TTS Space did not return an event_id.")

    request = _tts_request(f"{TTS_API_URL}/{event_id}", accept="text/event-stream")
    with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
        stream = response.read().decode("utf-8", errors="replace")
    if "event: error" in stream:
        raise RuntimeError("The TTS Space reported an inference error.")
    if "event: complete" not in stream:
        raise RuntimeError("The TTS Space closed without a complete event.")

    data = extract_json_from_event_stream(stream)
    if not data:
        raise RuntimeError("The TTS result did not contain JSON data.")
    file_url = json.loads(data)[0].get("url")
    if not file_url:
        raise RuntimeError("The TTS result did not contain an audio URL.")

    with urllib.request.urlopen(_tts_request(file_url), timeout=TTS_TIMEOUT_SECONDS) as response:
        audio = response.read()
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise RuntimeError("The TTS download was not a WAV file.")

    return audio


class TtsCache:
    """Cache of synthesized clips plus a background prefetch worker.

    `schedule(text)` queues a synthesis without blocking. The worker runs one
    request at a time, so it never floods the Space. `get(text)` returns ready
    audio immediately, waits for a still-running prefetch, or falls back to a
    direct synchronous fetch.
    """

    def __init__(self, ahead=PREFETCH_AHEAD):
        self.ahead = ahead
        self._entries = {}            # text -> {"event": Event, "audio": bytes|None, "error": str|None}
        self._order = []              # texts, oldest first, for eviction
        self._queue = []              # texts waiting to be synthesized
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- public API ------------------------------------------------------

    def schedule(self, texts):
        """Queue `texts` for background synthesis (already-cached ones skip)."""
        with self._lock:
            for text in texts:
                if not text or text in self._entries:
                    continue
                self._entries[text] = {"event": threading.Event(), "audio": None, "error": None}
                self._order.append(text)
                self._queue.append(text)
            self._evict_locked()
        self._wake.set()

    def get(self, text):
        """Return WAV bytes for `text`.

        If a prefetched clip is ready it is returned at once. If a background
        request is still running, we block until it finishes. Otherwise we
        synthesize synchronously. Raises on failure or missing token.
        """
        if not TTS_TOKEN:
            raise RuntimeError("No Hugging Face token configured for TTS.")

        with self._lock:
            entry = self._entries.get(text)
            if entry is not None and entry["event"].is_set():
                # Prefetch already finished (or failed).
                if entry["error"]:
                    self._forget(text)
                    raise RuntimeError(entry["error"])
                return entry["audio"]

        if entry is not None:
            # A background request is in flight; wait for it.
            entry["event"].wait()
            if entry["error"]:
                self._forget(text)
                raise RuntimeError(entry["error"])
            return entry["audio"]

        # No prefetch is running for this text: synthesize it now.
        audio = fetch_tts_audio(text)
        if audio is None:
            raise RuntimeError("The TTS Space returned no audio.")
        with self._lock:
            if entry is None:
                entry = {"event": threading.Event(), "audio": None, "error": None}
                self._entries[text] = entry
                self._order.append(text)
                self._evict_locked()
            entry["audio"] = audio
            entry["event"].set()
        return audio

    def stop(self):
        self._running = False
        self._wake.set()

    def ready(self):
        """Block until all currently queued/running prefetches have finished."""
        with self._lock:
            entries = list(self._entries.values())
        for entry in entries:
            entry["event"].wait()

    def pending_count(self):
        """Number of clips still queued or being synthesized right now."""
        with self._lock:
            return sum(1 for e in self._entries.values() if not e["event"].is_set())

    # -- internals -------------------------------------------------------

    def _forget(self, text):
        with self._lock:
            self._entries.pop(text, None)
            if text in self._order:
                self._order.remove(text)

    def _evict_locked(self):
        while len(self._order) > CACHE_LIMIT:
            oldest = self._order.pop(0)
            if oldest in self._queue:
                self._queue.remove(oldest)
            self._entries.pop(oldest, None)

    def _run(self):
        while self._running:
            self._wake.wait()
            self._wake.clear()
            while self._running:
                with self._lock:
                    if not self._queue:
                        break
                    text = self._queue.pop(0)
                    entry = self._entries.get(text)
                if entry is None:
                    continue
                try:
                    audio = fetch_tts_audio(text)
                    if audio is None:
                        raise RuntimeError("The TTS Space returned no audio.")
                    entry["audio"] = audio
                except Exception as e:
                    entry["error"] = str(e) or e.__class__.__name__
                entry["event"].set()


def slow_down_wav(audio_bytes, speed=PLAYBACK_SPEED):
    """Return a copy of the WAV with its sample rate scaled by `speed`.

    Lowering the declared sample rate makes every player read the audio slower
    (like slowing a tape). No resampling is needed and it works with winsound,
    afplay, paplay and aplay alike.
    """
    if speed == 1.0 or len(audio_bytes) < 44:
        return audio_bytes
    data = bytearray(audio_bytes)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return audio_bytes
    # Walk the chunk list to find "fmt ", which holds the sample rate at +12.
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = bytes(data[offset:offset + 4])
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        if chunk_id == b"fmt " and offset + 8 + 16 <= len(data):
            rate_offset = offset + 8 + 4  # after audio_format + channels
            rate = int.from_bytes(data[rate_offset:rate_offset + 4], "little")
            new_rate = max(1, int(round(rate * speed)))
            data[rate_offset:rate_offset + 4] = new_rate.to_bytes(4, "little")
            return bytes(data)
        offset += 8 + chunk_size + (chunk_size & 1)
    return audio_bytes


def play_wav_bytes(audio_bytes):
    """Play WAV bytes through the system's default audio device, slowed down."""
    audio_bytes = slow_down_wav(audio_bytes, PLAYBACK_SPEED)
    if os.name == "nt":
        import winsound
        path = os.path.join(tempfile.gettempdir(), "verifyer_tts.wav")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return

    # Linux / macOS fallback: afplay, paplay or aplay.
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        for player in (["afplay"], ["paplay"], ["aplay", "-q"]):
            try:
                subprocess.run(player + [path], check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        print("  TTS: No audio player found (tried afplay, paplay, aplay).")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def speak(text, cache=None):
    """Play the East Frisian TTS audio. Never aborts the review."""
    if not TTS_TOKEN:
        print("  TTS: No token configured; skipping audio. "
              "Pass a token, set HF_TOKEN or create verifyer_config.json.")
        return
    try:
        if cache is not None:
            audio = cache.get(text)
            label = "playing"
        else:
            print("  TTS: synthesizing...", end="\r", flush=True)
            audio = fetch_tts_audio(text)
            label = "playing"
    except Exception as e:
        print(f"  TTS: failed ({e})")
        return
    print(f"\r  TTS: {label}...        ", flush=True)
    try:
        play_wav_bytes(audio)
    except Exception as e:
        print(f"  TTS: playback failed ({e})")


def count_verified(verified_ids, id_to_index):
    """Number of Tatoeba rows that have already been verified."""
    return sum(1 for sid in id_to_index if sid in verified_ids)


def show_and_speak(verified_count, total_lines, current_id, de_text, frs_text, cache=None, read_aloud=True):
    """Print the sentence pair and read the East Frisian sentence aloud."""
    percentage = (verified_count / total_lines) * 100
    print(f"ID {current_id}  ({verified_count} / {total_lines} verifiziert, {percentage:.2f}%)")
    print(f"DE:  {de_text}")
    print(f"FRS: {frs_text}")

    if AUTO_READ and read_aloud:
        speak(frs_text, cache=cache)


def run_translation_check(tsv_path, log_path, error_path):
    global AUTO_READ
    # Load all previously verified Tatoeba IDs into a set for O(1) lookups
    verified_ids = read_verified_indices(log_path)

    rows = read_tsv_rows(tsv_path)
    if not rows:
        print(f"\nERROR: Could not read source file or file is empty: {tsv_path}")
        return

    total_rows = len(rows)
    # Map each Tatoeba ID to its position (0-based) in the (ID-sorted) row list.
    id_to_index = {sid: i for i, (sid, _, _) in enumerate(rows)}

    # Find the first unverified row to display as the resume point
    start_index = 0
    for i in range(total_rows + 1):
        if i >= total_rows:
            start_index = total_rows
            break
        sid = rows[i][0]
        if sid not in verified_ids:
            start_index = i
            break

    print(f"--- East Frisian Translation Validator ---")
    print(f"Total Rows in File: {total_rows}")
    print(f"Resuming at the first unverified sentence: "
          f"ID {rows[start_index][0] if start_index < total_rows else '-'}")
    if TTS_TOKEN:
        print("Audio: East Frisian sentences are read aloud via the HF TTS Space")
    else:
        print("Audio: disabled (no Hugging Face token configured)")
    print("Commands: [y] = Correct, [n] = Incorrect, [p] = Play again, "
          "[m] = Mute/Unmute, [r] = Reload from Disk, "
          "[b] = Back (undo previous), [exit] = Stop\n")

    try:
        # Background synthesis cache, active only when a token is available.
        cache = TtsCache() if TTS_TOKEN else None

        # Prime the first sentences so the audio is ready when we get there.
        if cache is not None and start_index < total_rows:
            warmup = [rows[i][2].strip() for i in range(start_index, min(start_index + PREFETCH_AHEAD, total_rows))]
            warmup = [t for t in warmup if t]
            if warmup:
                print(f"Filling audio cache ({len(warmup)} sentences)...", flush=True)
                cache.schedule(warmup)
                cache.ready()
                print("Audio cache ready.\n")

        try:
            # Index-based loop so [b] can jump back to earlier sentences.
            current_index = start_index
            while current_index < total_rows:
                current_id = rows[current_index][0]

                # Dynamically skip any row whose ID is already verified
                if current_id in verified_ids:
                    current_index += 1
                    continue

                de_text = rows[current_index][1].strip()
                frs_text = rows[current_index][2].strip()

                if not de_text or not frs_text:
                    append_verified_id(log_path, current_id)
                    verified_ids.add(current_id)
                    current_index += 1
                    continue

                if cache is not None:
                    nxt = [rows[i][2].strip() for i in range(current_index, min(current_index + PREFETCH_AHEAD + 1, total_rows))]
                    cache.schedule(nxt)

                show_and_speak(count_verified(verified_ids, id_to_index), total_rows, current_id, de_text, frs_text, cache=cache)

                while True:
                    user_input = input("Valid? (y/n/p/m/r/b/exit): ").lower().strip()

                    if user_input == 'p':
                        speak(frs_text, cache=cache)
                        continue

                    if user_input == 'm':
                        AUTO_READ = not AUTO_READ
                        state = "on" if AUTO_READ else "off"
                        print(f"Automatic playback: {state}\n")
                        if AUTO_READ:
                            speak(frs_text, cache=cache)
                        continue

                    if user_input == 'r':
                        print("\n--- Reloading row from disk... ---")
                        fresh = read_tsv_rows(tsv_path)
                        if fresh and current_index < len(fresh):
                            rows[current_index] = fresh[current_index]
                        current_id = rows[current_index][0]
                        de_text = rows[current_index][1].strip()
                        frs_text = rows[current_index][2].strip()
                        show_and_speak(count_verified(verified_ids, id_to_index), total_rows, current_id, de_text, frs_text,
                                       cache=cache, read_aloud=False)
                        continue

                    if user_input == 'b':
                        # Jump to the most recently handled sentence (by position)
                        # and undo its stored result, so it can be reviewed again.
                        earlier = [rid for rid in verified_ids
                                   if id_to_index.get(rid, -1) < current_index]
                        if not earlier:
                            print("No earlier sentence to go back to.\n")
                            continue
                        target_id = max(earlier, key=lambda rid: id_to_index.get(rid, -1))
                        undo_verified_index(log_path, error_path, verified_ids, target_id)
                        print(f"\nStatus: Back to ID {target_id}. "
                              f"Its previous result was undone.\n")
                        current_index = id_to_index[target_id]
                        break

                    if user_input == 'y':
                        append_verified_id(log_path, current_id)
                        verified_ids.add(current_id)
                        print("Status: Verified.\n")
                        current_index += 1
                        break

                    elif user_input == 'exit':
                        print(f"\nExiting. Next time you will resume at the next unverified sentence.")
                        return

                    else:
                        append_line(error_path,
                                    f"ID {current_id} | DE: {de_text} | FY: {frs_text}")
                        append_verified_id(log_path, current_id)
                        verified_ids.add(current_id)
                        print(f"Status: Sent to {error_path}\n")
                        current_index += 1
                        break

            print("--- All sentences have been reviewed! ---")
        finally:
            if cache is not None:
                cache.stop()

    except FileNotFoundError:
        print(f"\nERROR: Could not find the source files.")

if __name__ == "__main__":
    TSV_FILE      = "../tatoeba_frs_export.tsv"
    LOG_FILE      = "verified_indices.txt"
    CORRECT_FILE  = "sentences-to-correct.txt"

    load_tts_token()
    use_safe_console()
    run_translation_check(TSV_FILE, LOG_FILE, CORRECT_FILE)
