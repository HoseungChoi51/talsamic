"""Guard for LLM-written explanations.

The LLM may rephrase a reason, but every number in its text must match a number the
engine produced (``facts``, plus numbers in the engine's own template text). Otherwise
the engine's fixed template is shown instead.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_NUM = re.compile(r"[-+−]?\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text):
        tok = tok.replace("−", "-").replace(",", "")
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _flatten(v: Any) -> Iterable[float]:
    if isinstance(v, bool):
        return
    if isinstance(v, (int, float)):
        yield float(v)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)


def allowed_numbers(facts: dict, template: str = "") -> set[float]:
    return set(_flatten(facts)) | set(numbers_in(template))


def unsupported_numbers(text: str, facts: dict, template: str = "", tol: float = 0.051) -> list[float]:
    ok = allowed_numbers(facts, template)
    return [x for x in numbers_in(text) if not any(abs(abs(x) - abs(a)) <= tol for a in ok)]


def checked_reason(llm_text: str | None, facts: dict, template: str) -> tuple[str, bool]:
    """Returns (text to show, whether the LLM text was used)."""
    if llm_text and not unsupported_numbers(llm_text, facts, template):
        return llm_text, True
    return template, False
