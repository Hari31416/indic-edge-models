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
    assert "bhaarat" in roman


def test_transliterate_empty_string():
    """Verify empty or whitespace strings return empty output."""
    assert transliterate_devanagari_to_roman("") == ""
    assert transliterate_devanagari_to_roman("   ") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("भारत", "bhaarat"),
        ("कितने", "kitne"),
        ("वैज्ञानिक", "vaigyaanik"),
        ("अलीगढ़", "aleegarh"),
        ("अलीगढ़", "aleegarh"),
        ("बढ़ना", "barhnaa"),
        ("फ़ैज़", "faiz"),
        ("ग़ज़ल", "ghazal"),
        ("भारत। १२३", "bhaarat. 123"),
        ("API में COVID-19 और iPhone", "API men COVID-19 aur iPhone"),
        ("आ माता न", "aa maataa na"),
    ],
)
def test_hindi_ascii_regressions(text, expected):
    assert transliterate_devanagari_to_roman(text) == expected


def test_legacy_is_explicit():
    assert transliterate_devanagari_to_roman("भारत", scheme="itrans") == "bharata"
    with pytest.raises(ValueError):
        transliterate_devanagari_to_roman("भारत", scheme="typo")
    with pytest.raises(ValueError):
        transliterate_devanagari_to_roman("़्रामक")


def test_split_keeps_variants_together():
    from indic_edge_embeddings.data.build_train_data import split_and_augment_records

    records = [
        dict(
            query_id=str(i),
            query=f"भारत {i}",
            positive="passage",
            negatives=[],
            script="native",
            language="hi",
        )
        for i in range(20)
    ]
    train, val = split_and_augment_records(records, 29, 9, 1.0, 42)
    assert len(train) <= 29 and len(val) <= 9
    assert len(val) == 8
    assert {r["source_query_id"] for r in train}.isdisjoint(
        {r["source_query_id"] for r in val}
    )
    for split in (train, val):
        for row in split:
            assert (
                sum(
                    other["source_query_id"] == row["source_query_id"]
                    for other in split
                )
                == 2
            )
    assert (train, val) == split_and_augment_records(records, 29, 9, 1.0, 42)


def test_duplicate_query_text_stays_in_one_split():
    from indic_edge_embeddings.data.build_train_data import split_and_augment_records

    records = [
        dict(
            query_id=str(i),
            query=q,
            positive="passage",
            negatives=[],
            script="native",
            language="hi",
        )
        for i, q in enumerate(["भारत", "  भारत  ", "देश", "देश", "राज्य", "राजधानी"])
    ]
    train, val = split_and_augment_records(records, 20, 4, 1.0, 42)
    assert {r["split_group_id"] for r in train}.isdisjoint(
        {r["split_group_id"] for r in val}
    )
    assert len(train) + len(val) == 12


def test_missing_dependency_has_no_fallback(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "indic_transliteration", None)
    with pytest.raises(ImportError):
        transliterate_devanagari_to_roman("भारत")
