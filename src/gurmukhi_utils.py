"""Gurmukhi Unicode detection, validation, and normalization utilities."""

import html
import re
import unicodedata

# Gurmukhi Unicode block: U+0A00 to U+0A7F
GURMUKHI_START = 0x0A00
GURMUKHI_END = 0x0A7F


def is_gurmukhi_char(ch: str) -> bool:
    """Check if a single character falls in the Gurmukhi Unicode block."""
    cp = ord(ch)
    return GURMUKHI_START <= cp <= GURMUKHI_END


def gurmukhi_ratio(text: str) -> float:
    """Return the fraction of non-whitespace characters that are Gurmukhi."""
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    gurmukhi_count = sum(1 for ch in non_space if is_gurmukhi_char(ch))
    return gurmukhi_count / len(non_space)


def is_gurmukhi_text(text: str, min_ratio: float = 0.5) -> bool:
    """Return True if text contains sufficient Gurmukhi content."""
    return gurmukhi_ratio(text) >= min_ratio


def normalize_gurmukhi(text: str) -> str:
    """
    Normalize Gurmukhi text for ASR training:
    - Strip HTML tags (from VTT formatting)
    - Remove VTT positioning cues like <c>, </c>, <00:00:01.234>
    - Unicode NFC normalization
    - Collapse multiple spaces into one
    - Strip leading/trailing whitespace
    """
    # Decode HTML entities (&lt; -> <, &amp; -> &, etc.)
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove VTT timestamp tags like <00:00:01.234>
    text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_non_gurmukhi(text: str) -> str:
    """Keep only Gurmukhi characters and spaces."""
    return "".join(ch for ch in text if is_gurmukhi_char(ch) or ch.isspace()).strip()
