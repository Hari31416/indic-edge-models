"""Transliteration utilities for generating synthetic Romanized queries."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Basic fallback mapping if indic_transliteration is unavailable
_DEV_TO_LATIN_FALLBACK = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ा": "aa",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "oo",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "्": "",
    "ं": "n",
    "ः": "h",
}


def transliterate_devanagari_to_roman(text: str, scheme: str = "itrans") -> str:
    """Transliterate Devanagari text to Roman/Latin script.

    Args:
        text: Input string containing Devanagari text.
        scheme: Transliteration scheme ('itrans', 'hk', or 'iast').

    Returns:
        Romanized text string, lowercased and stripped of excessive whitespace.
    """
    if not text:
        return ""

    try:
        from indic_transliteration import sanscript

        target_scheme = sanscript.ITRANS
        if scheme == "hk":
            target_scheme = sanscript.HK
        elif scheme == "iast":
            target_scheme = sanscript.IAST

        romanized = sanscript.transliterate(text, sanscript.DEVANAGARI, target_scheme)
    except ImportError:
        logger.warning(
            "indic_transliteration not available; using simple character replacement fallback"
        )
        chars = []
        for char in text:
            chars.append(_DEV_TO_LATIN_FALLBACK.get(char, char))
        romanized = "".join(chars)

    # Normalize whitespace and lowercase
    romanized = romanized.lower()
    romanized = re.sub(r"\s+", " ", romanized).strip()
    return romanized
