"""Mathematical add-ons/shortcuts: compress hashes and arithmetic value representations.

Provides short base62-encoded hashes (and packed ids) and compact arithmetic value
forms to shorten length of hash/arithmatic values while keeping deterministic mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import math as _math
except Exception:  # pragma: no cover
    _math = None


_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def b62_encode(n: int) -> str:
    """Shorten an integer (e.g. 64-bit hash) to base62 string."""
    x = int(n)
    if x == 0:
        return "0"
    neg = x < 0
    x = abs(x)
    out = []
    while x:
        x, r = divmod(x, 62)
        out.append(_ALPHABET[r])
    s = "".join(reversed(out))
    return "-" + s if neg else s


def b62_decode(s: str) -> int:
    """Decode base62 string back to integer."""
    if not s:
        return 0
    neg = s.startswith("-")
    t = s[1:] if neg else s
    n = 0
    for ch in t:
        idx = _ALPHABET.find(ch)
        if idx < 0:
            raise ValueError(f"invalid base62 char: {ch!r}")
        n = n * 62 + idx
    return -n if neg else n


def shorten_hash(h: int, width: int = 11) -> str:
    """Deterministic short hash string for 64-bit hashes (default ~11 chars for u64)."""
    return b62_encode(int(h) & 0xFFFFFFFFFFFFFFFF).rjust(max(1, width), "0")[-max(1, width):]


def pack_u64(a: int, b: int) -> int:
    """Shortcut: pack two 32-bit values into one 64-bit id (shorter than storing pair)."""
    return ((int(a) & 0xFFFFFFFF) << 32) | (int(b) & 0xFFFFFFFF)


def unpack_u64(v: int) -> Tuple[int, int]:
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    return (v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF


def compress_arith_value(value: int) -> Dict[str, Any]:
    """Shorten arithmetic value representation while preserving full value."""
    v = int(value)
    sign = -1 if v < 0 else (1 if v > 0 else 0)
    mag = abs(v)
    # scientific-like compact form via log10 (shortcut length vs large decimal strings)
    if mag == 0:
        mant, exp10 = 0.0, 0
    elif _math is not None:
        exp10 = int(_math.floor(_math.log10(mag)))
        mant = mag / (10 ** exp10) if exp10 >= 0 else mag * (10 ** (-exp10))
    else:
        s = str(mag)
        exp10 = len(s) - 1
        mant = mag / (10 ** exp10) if exp10 else float(mag)
    short = b62_encode(v)
    return {
        "v": v,
        "sign": sign,
        "mag": mag,
        "b62": short,
        "len_dec": len(str(v)),
        "len_b62": len(short),
        "sci": {"mantissa": mant, "exp10": exp10},
    }


def simplify_log(x: float, base: float = 10.0) -> Dict[str, Any]:
    """Simplify log_base(x) using magnitude decomposition (log shortcut)."""
    if _math is None:
        raise RuntimeError("math module unavailable")
    xf = float(x)
    if xf <= 0:
        raise ValueError("log undefined for non-positive x")
    if base <= 0 or base == 1:
        raise ValueError("log base must be > 0 and != 1")
    if base == 10.0:
        exp10 = int(_math.floor(_math.log10(xf)))
        mant = xf / (10.0 ** exp10)
        value = exp10 + _math.log10(mant)
        expr = f"log10({xf}) = {exp10} + log10({mant})"
        return {"x": xf, "base": base, "value": value, "exp10": exp10, "mantissa": mant, "expr": expr}
    ln_x = _math.log(xf)
    ln_b = _math.log(base)
    value = ln_x / ln_b
    # decompose x = mantissa * 10^exp10 for a compact form
    exp10 = int(_math.floor(_math.log10(xf)))
    mant = xf / (10.0 ** exp10)
    expr = f"log_{base}({xf}) = ln({xf})/ln({base}) = {exp10}*log_{base}(10) + log_{base}({mant})"
    return {
        "x": xf,
        "base": base,
        "value": value,
        "exp10": exp10,
        "mantissa": mant,
        "ln_x": ln_x,
        "ln_base": ln_b,
        "expr": expr,
    }


def shortcut_mul_small(a: int, b: int) -> Optional[int]:
    """Exact shortcuts for common cases to reduce arithmetic work/representation."""
    a = int(a); b = int(b)
    if a == 0 or b == 0:
        return 0
    if a == 1:
        return b
    if b == 1:
        return a
    if a == -1:
        return -b
    if b == -1:
        return -a
    # power-of-two shortcut
    def _pow2(x: int) -> Optional[int]:
        ax = abs(x)
        if ax != 0 and (ax & (ax - 1)) == 0:
            shift = ax.bit_length() - 1
            return shift if x > 0 else shift  # sign handled outside
        return None
    sa = _pow2(a)
    sb = _pow2(b)
    if sb is not None:
        return (b // abs(b)) * (a << sb)
    if sa is not None:
        return (a // abs(a)) * (b << sa)
    return None


def shortcut_div_small(a: int, b: int) -> Optional[int]:
    a = int(a); b = int(b)
    if b == 0:
        return None
    if b == 1:
        return a
    if b == -1:
        return -a
    if a == 0:
        return 0
    # power-of-two divisor shortcut (trunc toward zero)
    ab = abs(b)
    if ab & (ab - 1) == 0:
        shift = ab.bit_length() - 1
        q = abs(a) >> shift
        sign_neg = (a < 0) ^ (b < 0)
        return -q if sign_neg else q
    return None


@dataclass
class ShortAddons:
    """Add-ons bundle for a hash/value pair."""
    hash_short: str
    value_compact: Dict[str, Any]

    @classmethod
    def from_hash_value(cls, h: int, v: int) -> "ShortAddons":
        return cls(hash_short=shorten_hash(h), value_compact=compress_arith_value(v))
