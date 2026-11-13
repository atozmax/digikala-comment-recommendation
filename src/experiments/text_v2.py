"""Character rules for the 12 November 2026 preprocessing experiment.

The input is the already cleaned, already Hazm-stemmed comments file.
This pass does not stem again, does not drop Latin letters, and does not
remove Persian stopwords. Stages run in order: letters, then digits,
then emoji.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_LETTERS = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ۀ": "ه",
        "ھ": "ه",
        "ؤ": "و",
    }
)

_DIGITS = str.maketrans(
    {chr(ord("0") + i): chr(0x06F0 + i) for i in range(10)}
    | {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}
)

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_INVISIBLE = re.compile(r"[\u200B\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2066-\u2069]")
_REPEATED_ZWNJ = re.compile(r"\u200C{2,}")
_EDGE_ZWNJ = re.compile(r"(?<!\S)\u200C|\u200C(?!\S)")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CHAR_RUN = re.compile(r"(.)\1{2,}")
_LINE_BREAKS = re.compile(r"[\r\n\t]+")
_SPACES = re.compile(r"\s+")
_EMOJI = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0E\U0000FE0F"
    "]+"
)
_PURE_PERSIAN_DIGITS = re.compile(r"^[۰-۹]+$")


def _squash(text: str) -> str:
    text = _LINE_BREAKS.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return _EDGE_ZWNJ.sub("", text)


def apply_letters(text: str) -> str:
    """NFKC, Persian letters, one interior half-space, no URLs, no long runs.

    Digits, emoji, and Latin letters are left for later stages. Nothing
    is stemmed.
    """
    if not isinstance(text, str) or not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_ARABIC_LETTERS)
    out = _DIACRITICS.sub("", out)
    out = _INVISIBLE.sub("", out)
    out = _REPEATED_ZWNJ.sub("\u200C", out)
    out = _EDGE_ZWNJ.sub("", out)
    out = _URL.sub(" ", out)
    out = _EMAIL.sub(" ", out)
    out = _CHAR_RUN.sub(r"\1\1", out)
    return _squash(out)


def apply_digits(text: str) -> str:
    """Fold ASCII and Arabic-Indic digits to Persian, then drop pure numbers.

    A token that mixes digits and letters, such as ``A54``, stays.
    """
    if not isinstance(text, str) or not text:
        return ""
    folded = text.translate(_DIGITS)
    kept = [token for token in folded.split(" ") if token and _PURE_PERSIAN_DIGITS.fullmatch(token) is None]
    return " ".join(kept)


def apply_emoji(text: str) -> str:
    """Delete emoji characters. The caller drops a body that becomes empty."""
    if not isinstance(text, str) or not text:
        return ""
    return _squash(_EMOJI.sub(" ", text))


def confusion_metrics(matrix: list[list[int]], names: tuple[str, ...]) -> dict[str, float | int]:
    """Accuracy, macro F1, micro F1, and per-class precision, recall, and F1.

    ``matrix[true][pred]`` is a count. Micro F1 equals accuracy for a
    single-label problem; both are stored so later reports do not mix them.
    """
    size = len(names)
    total = 0
    correct = 0
    per_f1: list[float] = []
    report: dict[str, float | int] = {}
    for index, name in enumerate(names):
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(size))
        true_positive = matrix[index][index]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_f1.append(f1)
        total += support
        correct += true_positive
        report[f"{name}_precision"] = float(precision)
        report[f"{name}_recall"] = float(recall)
        report[f"{name}_f1"] = float(f1)
        report[f"{name}_support"] = int(support)
    accuracy = correct / total if total else 0.0
    report["accuracy"] = float(accuracy)
    report["f1_macro"] = float(sum(per_f1) / size) if size else 0.0
    report["f1_micro"] = float(accuracy)
    report["rows"] = int(total)
    return report
