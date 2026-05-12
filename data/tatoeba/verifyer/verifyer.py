import os

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

def run_translation_check(de_path, frs_path, log_path, error_path):
    start_line_number = 1 
    
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                try:
                    start_line_number = int(lines[-1].strip()) + 1
                except ValueError:
                    start_line_number = 1

    print(f"--- East Frisian Translation Validator ---")
    print(f"Resuming at Line: {start_line_number}")
    print("Commands: [y] = Correct, [n] = Incorrect, [r] = Reload from Disk, [exit] = Stop\n")

    try:
        # We still open the files to iterate, but we will re-read them if 'r' is pressed
        with open(de_path, 'r', encoding='utf-8') as de_file, \
             open(frs_path, 'r', encoding='utf-8') as frs_file, \
             open(log_path, 'a', encoding='utf-8') as log_file, \
             open(error_path, 'a', encoding='utf-8') as err_file:

            for current_line, (de_line, frs_line) in enumerate(zip(de_file, frs_file), 1):
                if current_line < start_line_number:
                    continue

                de_text = de_line.strip()
                frs_text = frs_line.strip()

                if not de_text or not frs_text:
                    log_file.write(f"{current_line}\n")
                    continue

                while True:
                    print(f"Line {current_line}")
                    print(f"DE:  {de_text}")
                    print(f"FRS: {frs_text}")

                    user_input = input("Valid? (y/n/r/exit): ").lower().strip()

                    if user_input == 'r':
                        print("\n--- Reloading file from disk... ---")
                        # Fetch the latest version of these lines specifically
                        de_text = get_specific_line(de_path, current_line)
                        frs_text = get_specific_line(frs_path, current_line)
                        continue 
                    
                    if user_input == 'y':
                        log_file.write(f"{current_line}\n")
                        log_file.flush() 
                        print("Status: Verified.\n")
                        break

                    elif user_input == 'exit':
                        print(f"\nExiting. Next time you will start at line {current_line}.")
                        return
                    
                    else:
                        err_file.write(f"Line {current_line} | DE: {de_text} | FY: {frs_text}\n")
                        err_file.flush()
                        log_file.write(f"{current_line}\n")
                        log_file.flush()
                        print(f"Status: Sent to {error_path}\n")
                        break

            print("--- All lines have been reviewed! ---")

    except FileNotFoundError:
        print(f"\nERROR: Could not find the source files.")

if __name__ == "__main__":
    GERMAN_FILE   = "../german_tatoeba.txt"
    FRISIAN_FILE  = "../eastfrisian_tatoeba.txt"
    LOG_FILE      = "verified_indices.txt"
    CORRECT_FILE  = "sentences-to-correct.txt"

    run_translation_check(GERMAN_FILE, FRISIAN_FILE, LOG_FILE, CORRECT_FILE)