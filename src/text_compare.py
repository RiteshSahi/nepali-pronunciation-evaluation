import re
from jiwer import wer, cer


# -----------------------------------------------------
# Normalize Text
# -----------------------------------------------------

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[।,!?;:\"'()\-\[\]{}]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -----------------------------------------------------
# Compare Text
# -----------------------------------------------------

def compare_text(reference, recognized):

    reference = normalize_text(reference)
    recognized = normalize_text(recognized)

    word_error = wer(
        reference,
        recognized
    )

    char_error = cer(
        reference,
        recognized
    )

    return (
        word_error,
        char_error
    )