"""Small helpers for Korean sentences (particles depend on the last syllable)."""

from __future__ import annotations

PAIRS = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"),
         "과": ("과", "와"), "와": ("과", "와"), "을": ("을", "를"), "를": ("을", "를")}


def _has_final(word: str) -> bool | None:
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch.isdigit():
            return ch in "013678"  # 영 일 삼 육 칠 팔
        if ch.isascii() and ch.isalpha():
            return ch.lower() not in "aeiouy"  # rough: "lot" -> lot이
    return None


def josa(word: str, particle: str) -> str:
    """``josa("압력", "과") -> "압력과"``; ``josa("온도", "은") -> "온도는"``."""
    with_final, without = PAIRS[particle]
    f = _has_final(word)
    if f is None:
        return f"{word}{with_final}({without})"
    return word + (with_final if f else without)


def fmt_pp(x: float) -> str:
    return f"{abs(x):.1f}%p"
