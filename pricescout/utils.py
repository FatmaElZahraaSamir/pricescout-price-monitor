"""Small parsing helpers shared by site adapters and fetchers."""

from __future__ import annotations

import re

CURRENCY_SYMBOLS = {
    "£": "GBP",
    "$": "USD",
    "€": "EUR",
    "¥": "JPY",
    "₹": "INR",
    "AED": "AED",
    "SAR": "SAR",
    "EGP": "EGP",
}

RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_INT_RE = re.compile(r"\d+")


def parse_price(text: str | None) -> tuple[float | None, str | None]:
    """Turn ``"£51.77"`` (or a mis-decoded ``"Â£51.77"``) into ``(51.77, "GBP")``."""
    if not text:
        return None, None
    currency = None
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            currency = code
            break
    match = _NUMBER_RE.search(text)
    if not match:
        return None, currency
    return float(match.group(0).replace(",", "")), currency


def parse_rating_word(text: str | None) -> int | None:
    """Map a CSS class such as ``"star-rating Three"`` to ``3``."""
    if not text:
        return None
    for token in text.replace("-", " ").split():
        value = RATING_WORDS.get(token.lower())
        if value:
            return value
    return None


def first_int(text: str | None) -> int | None:
    """Return the first integer in ``"In stock (22 available)"`` -> ``22``."""
    if not text:
        return None
    match = _INT_RE.search(text)
    return int(match.group(0)) if match else None


def is_in_stock(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "out of stock" in lowered or "unavailable" in lowered:
        return False
    return "in stock" in lowered or "available" in lowered


def decode_html(content: bytes, declared_charset: str | None = None) -> str:
    """Decode a response body robustly.

    Some servers (books.toscrape.com included) send ``Content-Type: text/html``
    without a charset. Naive clients then fall back to ISO-8859-1 and the pound
    sign becomes ``Â£``. We honour a declared charset, otherwise try UTF-8 and
    only then fall back to Windows-1252.
    """
    if declared_charset:
        try:
            return content.decode(declared_charset)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
