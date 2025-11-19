import os
import difflib
import string

def load_dictionary(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        return set(w.strip() for w in f if w.strip())


def save_dictionary(dict_path, dictionary):
    with open(dict_path, "w", encoding="utf-8") as f:
        for w in sorted(dictionary):
            f.write(w + "\n")


def clean_word(word):
    """Remove punctuation around words, e.g. pad, → pad"""
    return word.strip(string.punctuation)


def word_in_dictionary(word, dictionary):
    """Check dictionary including lowercase fallback, ignoring punctuation."""
    w = clean_word(word)
    if w in dictionary:
        return True
    if w.lower() in dictionary:
        return True
    return False


def suggest_replacement(word, dictionary):
    """Suggest closest match based on lowercase comparison, ignoring punctuation."""
    base_word = clean_word(word).lower()
    matches = difflib.get_close_matches(base_word, dictionary, n=1, cutoff=0.6)
    return matches[0] if matches else None


def main():
    base_path = "data/fan teksten"
    val_path = "validation data"
    ger_path = os.path.join(base_path, "german.txt")
    frs_path = os.path.join(base_path, "eastfrisian.txt")
    dict_path = os.path.join(base_path, "frs.dic")
    out_path = os.path.join(base_path, "eastfrisian_updated.txt")

    # Load dictionary
    dictionary = load_dictionary(dict_path)

    # Load German and Frisian text
    with open(ger_path, "r", encoding="utf-8") as ger_file, \
         open(frs_path, "r", encoding="utf-8") as frs_file:

        german = [line.strip() for line in ger_file]
        frisian = [line.strip() for line in frs_file]

    assert len(german) == len(frisian), "Line count mismatch!"

    # Load existing progress if exist
    if os.path.exists(out_path):
        print("Resuming from existing updated file...")
        with open(out_path, "r", encoding="utf-8") as out:
            updated_frisian = [line.strip() for line in out]
    else:
        updated_frisian = frisian.copy()

    total = len(frisian)

    for idx, (ger, original_frs) in enumerate(zip(german, frisian), start=1):

        # If sentence already processed, skip it
        if idx - 1 < len(updated_frisian) and updated_frisian[idx - 1] != original_frs:
            continue

        words = original_frs.split()

        # detect unknown words
        unknown_words = [w for w in words if not word_in_dictionary(w, dictionary)]

        if not unknown_words:
            continue

        print("\n-------------------------------------")
        print(f"Sentence {idx}/{total}")
        print("German:      ", ger)
        print("East Frisian:", original_frs)
        print("Unknown words:", unknown_words)

        do_correct = input("\nCorrect this sentence? (y/n): ").strip().lower()
        if do_correct != "y":
            continue

        new_words = words[:]

        for uw in unknown_words:
            cleaned = clean_word(uw)

            print(f"\nUnknown word: {uw} (cleaned: {cleaned})")

            suggestion = suggest_replacement(uw, dictionary)
            if suggestion:
                print(f"Suggested replacement: {suggestion}")
            else:
                print("No suggestion available.")

            action = input("Replace with suggestion? (y=yes, n=enter manually, s=skip and add to dictionary): ").strip().lower()

            # If using suggestion
            if action == "y" and suggestion:
                replacement = suggestion
                dictionary.add(replacement)

            # Manual correction
            elif action == "n":
                replacement = input("Enter correct replacement: ").strip()
                dictionary.add(replacement)

            # Skip → but add cleaned word to dictionary
            elif action == "s":
                dictionary.add(cleaned)
                continue

            else:
                continue

            # Perform replacement in the sentence
            new_words = [replacement if w == uw else w for w in new_words]

        updated_frisian[idx - 1] = " ".join(new_words)
        print("Updated sentence:", updated_frisian[idx - 1])

        # Save updated file after each sentence
        with open(out_path, "w", encoding="utf-8") as out:
            for line in updated_frisian:
                out.write(line + "\n")

        # Save updated dictionary
        save_dictionary(dict_path, dictionary)

    print("\nDone!")
    print("Updated file written to:", out_path)
    print("Dictionary saved to:", dict_path)


if __name__ == "__main__":
    main()