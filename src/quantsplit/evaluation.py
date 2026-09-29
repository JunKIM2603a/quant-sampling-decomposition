"""Strict numeric parser: no eval(), no recovery from reasoning-only output."""
from dataclasses import dataclass
from fractions import Fraction
import re

PARSER_VERSION = "boxed-after-think-v1"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


def numeric_value(text):
    text = text.strip().replace("−", "-")
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    # Accept commas only as valid thousands separators.
    if "," in text:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
            raise ValueError("invalid thousands separators")
        text = text.replace(",", "")
    frac = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\s*\{(" + NUMBER +
                        r")\}\s*\{(" + NUMBER + r")\}", text)
    if frac:
        value = Fraction(frac[2]) / Fraction(frac[3])
        return -value if frac[1] == "-" else value
    if re.fullmatch(NUMBER, text):
        return Fraction(text)
    frac = re.fullmatch(r"(" + NUMBER + r")\s*/\s*(" + NUMBER + r")", text)
    if frac:
        return Fraction(frac[1]) / Fraction(frac[2])
    raise ValueError("not a supported numeric answer")


def last_boxed(text):
    # A later incomplete box makes the output incomplete, not an earlier answer.
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i]
    return None


@dataclass(frozen=True)
class Score:
    correct: bool
    reasoning_complete: bool
    final_answer_complete: bool
    parsed: str | None
    failure: str | None


def score_gsm8k(generated_text, gold_answer):
    # Dataset schema errors are infrastructure errors, never model mistakes.
    if "####" not in gold_answer:
        raise ValueError("GSM8K answer must contain ####")
    gold = numeric_value(gold_answer.rsplit("####", 1)[1])
    if "</think>" not in generated_text:
        return Score(False, False, False, None, "missing_reasoning_end")
    final = generated_text.rsplit("</think>", 1)[1]
    boxed = last_boxed(final)
    if boxed is None:
        return Score(False, True, False, None, "missing_or_incomplete_box")
    try:
        value = numeric_value(boxed)
    except (ValueError, ZeroDivisionError):
        return Score(False, True, True, None, "invalid_numeric_box")
    return Score(value == gold, True, True, str(value), None)
