"""Unit tests for transliteration and data utilities."""

import pytest
from indic_edge_embeddings.data.transliterate import (
    transliterate_devanagari_to_roman,
)


def test_transliterate_devanagari_to_roman():
    """Verify Hindi Devanagari text converts to clean lowercased Latin string."""
    text = "भारत की राजधानी क्या है?"
    roman = transliterate_devanagari_to_roman(text)
    assert len(roman) > 0
    assert roman.islower() or not any(c.isupper() for c in roman)
    assert "bharat" in roman.lower() or "bhaarata" in roman.lower()


def test_transliterate_empty_string():
    """Verify empty or whitespace strings return empty output."""
    assert transliterate_devanagari_to_roman("") == ""
    assert transliterate_devanagari_to_roman("   ") == ""
