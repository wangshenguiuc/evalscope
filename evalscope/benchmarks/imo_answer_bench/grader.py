"""
Math answer grader for IMO-AnswerBench.

Uses math-verify + sympy for robust mathematical equivalence checking,
adapted from MiroVerifier's RuleBasedAnswerGrader logic.
"""

import math
import re
from typing import Any

import sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from sympy.parsing.latex import parse_latex as parse_latex_core
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]

_UNICODE_REPLACEMENTS = {
    "²": "^{2}",
    "³": "^{3}",
    "ⁿ": "^{n}",
    "π": "\\pi ",
    "∞": "\\infty ",
    "–": "-",
    "−": "-",
    "∪": "\\cup ",
    "∩": "\\cap ",
    "·": "\\cdot ",
    "×": "\\times ",
    "⁄": "/",
    "\xa0": " ",
    "½": "\\frac{1}{2}",
}


def _normalize_for_sympy(expression: str) -> str:
    normalized = expression.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = re.sub(r"(?<!\d),(?=\d{3}\b)", "", normalized)
    normalized = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1)/(\2)",
        normalized,
    )
    normalized = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = normalized.replace("{", "(").replace("}", ")")
    return normalized.replace("\\", "")


def _parse_sympy_expression(expression: str) -> Any:
    return parse_expr(
        _normalize_for_sympy(expression),
        transformations=_SYMPY_TRANSFORMATIONS,
        evaluate=True,
    )


def _fix_unicode(string: str) -> str:
    string = re.sub(
        r"√(\([^()]*\)|[A-Za-z0-9]+)",
        lambda m: r"\sqrt{" + m.group(1) + "}",
        string,
    )
    for unicode_char, latex_equiv in _UNICODE_REPLACEMENTS.items():
        string = string.replace(unicode_char, latex_equiv)
    return string


def _strip_string(string: Any) -> str:
    """Normalize a math answer string (adapted from MiroVerifier)."""
    string = str(string).strip().replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = re.sub(r"(?<!\\)\\ ", "", string)
    string = string.replace("\\,", "").replace("\\:", "").replace("\\;", "")
    string = string.replace("\\quad", "")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")

    stripped_text_suffix = re.sub(r"\\text{.*?}$", "", string).strip()
    if stripped_text_suffix:
        string = stripped_text_suffix

    string = string.replace("^{\\circ}", "").replace("^\\circ", "").strip()

    string = _fix_unicode(string)
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "%").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("\\mathbf", "").replace("\\mathrm", "")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if string and string[0] == ".":
        string = "0" + string

    # Normalize spaces around commas: "3, 7" → "3,7"
    string = re.sub(r"\s*,\s*", ",", string)

    # Normalize LaTeX relation aliases: \geq → \ge, \leq → \le, \neq → \ne
    string = string.replace("\\geq", "\\ge").replace("\\leq", "\\le").replace("\\neq", "\\ne")

    # Normalize spaces around comparison / relation operators
    string = re.sub(r"\s*([<>])\s*", r"\1", string)
    string = re.sub(r"\s*([=])\s*", r"\1", string)
    string = re.sub(r"\s*(\\le|\\ge|\\ne)\s*", r"\1", string)

    # Normalize subscript commas: a_{i,j} → a_{ij}
    string = re.sub(r"_\{([^}]*?),([^}]*?)\}", lambda m: "_{" + m.group(1) + m.group(2) + "}", string)

    string = re.sub(r"(\\|,|\.)+$", "", string).strip()

    # Collapse remaining spaces (implicit multiplication, formatting variations)
    string = string.replace(" ", "")

    return string


def _parse_digits(num: Any) -> float | None:
    num = re.sub(r"\{,\}", "", str(num))
    num = re.sub(r",", "", str(num))
    try:
        return float(num)
    except Exception:
        if str(num).endswith("%"):
            try:
                return float(str(num)[:-1].rstrip("\\")) / 100
            except Exception:
                return None
    return None


def _verify_math_verify(reference: str, prediction: str) -> bool | None:
    """Try equivalence check via math-verify library."""
    try:
        gold = parse(reference, extraction_config=_EXTRACTION_CONFIG, parsing_timeout=None)
        predicted = parse(prediction, extraction_config=_EXTRACTION_CONFIG, parsing_timeout=None)
    except Exception:
        return None
    if not gold or not predicted:
        return None
    try:
        return bool(verify(gold, predicted, float_rounding=6, numeric_precision=15))
    except Exception:
        return None


def _numeric_equivalence(left: str, right: str, tol: float = 0.0) -> bool | None:
    """Try numeric equivalence via sympy evaluation."""
    try:
        left_value = float(sympy.N(_parse_sympy_expression(left)))
        right_value = float(sympy.N(_parse_sympy_expression(right)))
    except Exception:
        return None
    return math.isclose(left_value, right_value, abs_tol=tol)


def _symbolic_equal(reference: str, prediction: str, tol: float = 0.0) -> bool | None:
    """Try symbolic equivalence: numeric → math-verify → sympy simplify."""
    numeric = _numeric_equivalence(reference, prediction, tol)
    if numeric is not None:
        return numeric

    mv = _verify_math_verify(reference, prediction)
    if mv is True:
        return True

    try:
        ref_expr = _parse_sympy_expression(reference)
        pred_expr = _parse_sympy_expression(prediction)
        if sympy.simplify(ref_expr - pred_expr) == 0:
            return True
        if tol > 0:
            return math.isclose(float(sympy.N(ref_expr)), float(sympy.N(pred_expr)), abs_tol=tol)
        return False
    except Exception:
        return None


def _split_set_items(answer: str) -> list[str] | None:
    if not answer:
        return None
    if "\\cup" in answer:
        parts = []
        for component in answer.split("\\cup"):
            items = _split_set_items(component.strip())
            if items is None:
                return None
            parts.extend(items)
        return parts
    stripped = answer.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    return [part.strip() for part in inner.split(",") if part.strip()]


def _split_bare_list(answer: str) -> list[str] | None:
    """Split a bare comma-separated list (no brackets) into items.

    Returns None if the answer doesn't look like a bare list.
    Only triggers when commas are present and the answer is not wrapped
    in brackets, braces, or parentheses.
    """
    if not answer or "," not in answer:
        return None
    stripped = answer.strip()
    # Skip if wrapped in any kind of brackets — those are handled elsewhere
    if (stripped[0] in "({[" and stripped[-1] in ")}]"):
        return None
    # Skip if it looks like a single expression with commas in subscripts/args
    # (e.g. "f(a,b)" or "\\binom{n,k}") — require at least one comma outside
    # of all bracket pairs
    depth = 0
    has_top_level_comma = False
    for ch in stripped:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            has_top_level_comma = True
            break
    if not has_top_level_comma:
        return None
    # Split on top-level commas only
    parts = []
    current = []
    depth = 0
    for ch in stripped:
        if ch in "({":
            depth += 1
            current.append(ch)
        elif ch in ")}":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    parts = [p for p in parts if p]
    return parts if len(parts) >= 2 else None


def _math_equal(prediction: str, reference: str, tol: float = 0.0) -> bool | None:
    """Core math equality check (single expression)."""
    if prediction == reference:
        return True

    # Numeric parse
    p_num = _parse_digits(prediction)
    r_num = _parse_digits(reference)
    if p_num is not None and r_num is not None:
        if math.isclose(p_num, r_num, abs_tol=tol):
            return True

    if not prediction:
        return False

    # Sequence/tuple match
    if (prediction[:1] in "([" and prediction[-1:] in ")]"
            and reference[:1] == prediction[:1] and reference[-1:] == prediction[-1:]):
        pred_parts = [p.strip() for p in prediction[1:-1].split(",")]
        ref_parts = [p.strip() for p in reference[1:-1].split(",")]
        if len(pred_parts) != len(ref_parts):
            return False
        return all(_math_equal(pp, rp, tol) for pp, rp in zip(pred_parts, ref_parts))

    # Equation match (a=b vs c=d)
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pl, pr = prediction.split("=")
        rl, rr = reference.split("=")
        pred_expr = f"{pl.strip()} - ({pr.strip()})"
        ref_expr = f"{rl.strip()} - ({rr.strip()})"
        direct = _symbolic_equal(ref_expr, pred_expr, tol)
        if direct is True:
            return True
        inverse = _symbolic_equal(ref_expr, f"-({pred_expr})", tol)
        if inverse is True:
            return True

    # Assignment match (x=5 vs 5)
    if prediction.count("=") == 1 and reference.count("=") == 0:
        left, right = prediction.split("=")
        if len(left.strip()) <= 2:
            result = _math_equal(right.strip(), reference, tol)
            if result is not None:
                return result
    if reference.count("=") == 1 and prediction.count("=") == 0:
        left, right = reference.split("=")
        if len(left.strip()) <= 2:
            result = _math_equal(prediction, right.strip(), tol)
            if result is not None:
                return result

    return _symbolic_equal(reference, prediction, tol)


def grade_answer(prediction: str, reference: str) -> bool:
    """Grade a predicted answer against a reference answer.

    Uses multi-layered equivalence checking:
    1. String normalization + exact match
    2. Numeric parsing
    3. math-verify library (LaTeX-aware)
    4. Sympy symbolic simplification
    5. Set/sequence/equation decomposition
    """
    pred = _strip_string(prediction)
    ref = _strip_string(reference)

    if pred == ref:
        return True

    # Set-level matching (unordered)
    pred_set = _split_set_items(pred)
    ref_set = _split_set_items(ref)
    if pred_set is not None and ref_set is not None:
        if len(pred_set) != len(ref_set):
            return False
        pred_matched = set()
        ref_matched = set()
        for i, p in enumerate(pred_set):
            for j, r in enumerate(ref_set):
                if j not in ref_matched and _math_equal(p, r):
                    pred_matched.add(i)
                    ref_matched.add(j)
                    break
        return len(pred_matched) == len(pred_set) and len(ref_matched) == len(ref_set)

    # Bare comma-separated list matching (unordered, no brackets)
    pred_bare = _split_bare_list(pred)
    ref_bare = _split_bare_list(ref)
    if pred_bare is not None and ref_bare is not None:
        if len(pred_bare) != len(ref_bare):
            return False
        ref_matched = set()
        for p in pred_bare:
            for j, r in enumerate(ref_bare):
                if j not in ref_matched and _math_equal(p, r):
                    ref_matched.add(j)
                    break
        if len(ref_matched) == len(ref_bare):
            return True

    result = _math_equal(pred, ref)
    return bool(result)
