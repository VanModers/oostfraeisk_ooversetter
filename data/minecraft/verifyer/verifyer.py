import json
import os
import sys
import tkinter as tk

# Basisverzeichnis des Skripts (./verifyer)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_abs_path(relative_path):
    """Erstellt einen absoluten Pfad basierend auf dem Skript-Standort."""
    return os.path.normpath(os.path.join(SCRIPT_DIR, relative_path))

def load_json(file_path):
    """Lädt eine JSON-Datei sicher von der Festplatte."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"HINWEIS: Datei '{file_path}' wurde nicht gefunden (wird übersprungen).")
        return {}
    except json.JSONDecodeError as e:
        print(f"FEHLER: Ungültiges JSON-Format in '{file_path}': {e}")
        return {}

def save_json(file_path, data):
    """Speichert Daten sauber formatiert in einer JSON-Datei ab."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"FEHLER beim Speichern der JSON-Datei: {e}")

def edit_text_popup(de_text, en_text, nl_text, prefill_text):
    """Öffnet ein großes Tkinter-Fenster mittig auf dem Bildschirm mit Referenzen."""
    result = {"text": None}

    root = tk.Tk()
    root.title("FRS Übersetzung bearbeiten")
    
    # Größe des Fensters angepasst für zusätzliche Zeilen
    width, height = 750, 450
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.attributes("-topmost", True)
    root.resizable(True, True)

    # Deutsch-Referenzanzeige
    lbl_de = tk.Label(
        root, 
        text=f"DE (Original): {de_text}", 
        wraplength=710, 
        justify="left", 
        fg="#222222", 
        font=("Arial", 11, "bold")
    )
    lbl_de.pack(anchor="w", padx=20, pady=(15, 5))

    # Englisch-Referenzanzeige
    lbl_en = tk.Label(
        root, 
        text=f"EN (Englisch): {en_text if en_text else '--- [FEHLT] ---'}", 
        wraplength=710, 
        justify="left", 
        fg="#555555", 
        font=("Arial", 10, "italic")
    )
    lbl_en.pack(anchor="w", padx=20, pady=(0, 5))

    # Niederländisch-Referenzanzeige
    lbl_nl = tk.Label(
        root, 
        text=f"NL (Niederländisch): {nl_text if nl_text else '--- [FEHLT] ---'}", 
        wraplength=710, 
        justify="left", 
        fg="#555555", 
        font=("Arial", 10, "italic")
    )
    lbl_nl.pack(anchor="w", padx=20, pady=(0, 10))

    lbl_frs = tk.Label(root, text="Ostfriesisch bearbeiten:", font=("Arial", 11, "bold"))
    lbl_frs.pack(anchor="w", padx=20, pady=(0, 2))

    # Mehrzeiliges Textfeld
    text_box = tk.Text(root, font=("Arial", 14), height=4, wrap="word", relief="groove", bd=2)
    text_box.insert("1.0", prefill_text)
    text_box.pack(fill="both", expand=True, padx=20, pady=5)
    text_box.focus_set()
    
    text_box.tag_add("sel", "1.0", "end-1c")

    def save_and_close(event=None):
        result["text"] = text_box.get("1.0", "end-1c")
        root.destroy()

    def cancel(event=None):
        root.destroy()

    root.bind("<Return>", save_and_close)
    root.bind("<Escape>", cancel)

    btn_frame = tk.Frame(root)
    btn_frame.pack(side="bottom", fill="x", pady=15, padx=20)

    btn_cancel = tk.Button(btn_frame, text="Abbrechen (Esc)", command=cancel, width=15, font=("Arial", 10))
    btn_cancel.pack(side="right", padx=(10, 0))

    btn_save = tk.Button(
        btn_frame, 
        text="Übernehmen (Enter)", 
        command=save_and_close, 
        width=20, 
        bg="#4CAF50", 
        fg="white", 
        font=("Arial", 10, "bold")
    )
    btn_save.pack(side="right")

    root.mainloop()
    return result["text"]

def run_minecraft_json_validator(de_path, frs_path, en_path, nl_path, log_path, error_path, corrected_path):
    de_abs = get_abs_path(de_path)
    frs_abs = get_abs_path(frs_path)
    en_abs = get_abs_path(en_path) if en_path else None
    nl_abs = get_abs_path(nl_path) if nl_path else None
    
    log_abs = get_abs_path(log_path)
    error_abs = get_abs_path(error_path)
    corrected_abs = get_abs_path(corrected_path)

    # 1. Daten laden
    de_data = load_json(de_abs)
    frs_data = load_json(frs_abs)
    en_data = load_json(en_abs) if en_abs else {}
    nl_data = load_json(nl_abs) if nl_abs else {}

    if not de_data:
        print("Abbruch: Keine Daten in der Quelldatei.")
        return

    # 2. Bereits überprüfte Keys laden
    verified_keys = set()
    if os.path.exists(log_abs):
        with open(log_abs, 'r', encoding='utf-8') as f:
            for line in f:
                verified_keys.add(line.strip())

    total_keys = len(de_data)
    
    # 3. Header anzeigen
    print("=" * 65)
    print("    MINECRAFT ÜBERSETZUNGS-VALIDATOR (JSON / LIVE-CORRECT)")
    print("=" * 65)
    print(f"Gesamt-Keys     : {total_keys}")
    print(f"Bereits geprüft : {len(verified_keys)} / {total_keys}")
    print("-" * 65)
    print("Befehle:")
    print("  [y]    = Richtig / Validiert")
    print("  [c]    = Korrigieren (Öffnet großes Eingabefenster)")
    print("  [n]    = Falsch (In Fehlerdatei protokollieren)")
    print("  [r]    = Neu von Festplatte laden")
    print("  [exit] = Beenden & Fortschritt speichern")
    print("=" * 65 + "\n")

    with open(log_abs, 'a', encoding='utf-8') as log_file, \
         open(error_abs, 'a', encoding='utf-8') as err_file, \
         open(corrected_abs, 'a', encoding='utf-8') as corr_file:

        current_index = 0

        for key, de_text in de_data.items():
            current_index += 1

            if key in verified_keys:
                continue

            frs_text = frs_data.get(key, "")
            en_text = en_data.get(key, "")
            nl_text = nl_data.get(key, "")

            percentage = (current_index / total_keys) * 100

            while True:
                print(f"--- Key {current_index} / {total_keys} ({percentage:.1f}%) ---")
                print(f"KEY: {key}")
                print(f"DE : {de_text}")
                print(f"EN : {en_text if en_text else '--- [FEHLT] ---'}")
                print(f"NL : {nl_text if nl_text else '--- [FEHLT] ---'}")
                print(f"FRS: {frs_text if frs_text else '--- [LEER / FEHLT] ---'}")

                user_input = input("\n[y/c/n/r/exit] > ").lower().strip()

                if user_input == 'r':
                    print("\n--> Lade Dateien neu von Festplatte...")
                    de_data = load_json(de_abs)
                    frs_data = load_json(frs_abs)
                    if en_abs: en_data = load_json(en_abs)
                    if nl_abs: nl_data = load_json(nl_abs)
                    de_text = de_data.get(key, "")
                    frs_text = frs_data.get(key, "")
                    en_text = en_data.get(key, "")
                    nl_text = nl_data.get(key, "")
                    continue

                elif user_input == 'y':
                    log_file.write(f"{key}\n")
                    log_file.flush()
                    print("Status: ✅ Verifiziert.\n")
                    break

                elif user_input == 'c':
                    default_fill = frs_text if frs_text else de_text
                    
                    # Pop-up mit zusätzlichen Sprachen aufrufen
                    new_frs_text = edit_text_popup(de_text, en_text, nl_text, default_fill)

                    if new_frs_text is not None and new_frs_text.strip() != "":
                        new_frs_text = new_frs_text.strip()
                        old_val = frs_text
                        frs_text = new_frs_text 
                        
                        frs_data[key] = new_frs_text
                        save_json(frs_abs, frs_data)

                        corr_file.write(f"Key: {key} | DE: {de_text} | FRS_ALT: {old_val} | FRS_NEU: {new_frs_text}\n")
                        corr_file.flush()

                        log_file.write(f"{key}\n")
                        log_file.flush()

                        print(f"Status: ✏️ Übernommen & verifiziert (Neu: {new_frs_text}).\n")
                        # GEÄNDERT: 'continue' sorgt dafür, dass sofort sauber zum nächsten Key gesprungen wird
                        continue
                    else:
                        print("❌ Bearbeitung abgebrochen. Zurück zum Eintrag...\n")

                elif user_input == 'n':
                    err_file.write(f"Key: {key} | DE: {de_text} | FRS: {frs_text}\n")
                    err_file.flush()
                    
                    log_file.write(f"{key}\n")
                    log_file.flush()
                    print(f"Status: ❌ In Fehlerdatei ({error_path}) notiert.\n")
                    break

                elif user_input == 'exit':
                    print("\nBeendet. Dein Fortschritt wurde gespeichert.")
                    return

                else:
                    print("Ungültige Eingabe! Bitte y, c, n, r oder exit eingeben.\n")

        print("\n🎉 Sämtliche Einträge wurden erfolgreich durchgegangen!")


if __name__ == "__main__":
    GERMAN_FILE    = "../de_de.json"
    FRISIAN_FILE   = "../frs_de.json"
    ENGLISH_FILE   = "../en_us.json"  # Passe hier den Pfad an (oder '' lassen, falls nicht vorhanden)
    DUTCH_FILE     = "../nl_nl.json"  # Passe hier den Pfad an (oder '' lassen, falls nicht vorhanden)
    
    LOG_FILE       = "verified_keys.txt"
    ERROR_FILE     = "keys-to-correct.txt"
    CORRECTED_FILE = "keys-directly-corrected.txt"

    run_minecraft_json_validator(GERMAN_FILE, FRISIAN_FILE, ENGLISH_FILE, DUTCH_FILE, LOG_FILE, ERROR_FILE, CORRECTED_FILE)