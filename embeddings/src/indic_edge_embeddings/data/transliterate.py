"""Versioned, deterministic synthetic Hindi Romanization.

The default is readable ASCII, not a pronunciation model or natural Hinglish.
Explicit ``itrans``, ``hk`` and ``iast`` modes retain the historical behavior.
"""

import re
import unicodedata

ROMANIZATION_VERSION = "hindi_ascii_v2"
_DEV_RUN = re.compile(r"[\u0900-\u0963\u0971-\u097f\u200c\u200d]+")
# Tokenize before ASCII folding so long vowels cannot be mistaken for schwas.
_IAST_TOKEN = re.compile(r"rh|kh|gh|ch|jh|ṭh|ḍh|th|dh|ph|bh|ai|au|.")
_VOWELS = set("aāiīuūṛṝḷḹeēoō") | {"ai", "au"}
_FOLD = {
    "ġ": "gh",
    "ẏ": "y",
    "c": "ch",
    "ch": "chh",
    "ṭh": "th",
    "ḍh": "dh",
    "ā": "aa",
    "ī": "ee",
    "ū": "oo",
    "ṛ": "ri",
    "ṝ": "ri",
    "ḷ": "li",
    "ḹ": "li",
    "ē": "e",
    "ō": "o",
    "ṅ": "n",
    "ñ": "n",
    "ṭ": "t",
    "ḍ": "d",
    "ṇ": "n",
    "ś": "sh",
    "ṣ": "sh",
    "ṃ": "n",
    "ṁ": "n",
    "ḥ": "h",
    "~": "n",
}


def _hindi_word(word: str) -> str:
    from indic_transliteration import sanscript

    word = (
        unicodedata.normalize("NFD", word).replace("\u200c", "").replace("\u200d", "")
    )
    word = re.sub("़+", "़", word)
    # Nukta letters not consistently handled by sanscript (notably ड़ and ढ़).
    word = word.replace("ड़", "र").replace("ढ़", "र्" + "ह")
    for source, target in {"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े"}.items():
        word = word.replace(source, target)
    roman = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)
    roman = roman.replace("jñ", "gy").replace("k͟h", "kh")
    roman = re.sub("[ṃṁ](?=[pbm])", "m", roman)
    tokens = _IAST_TOKEN.findall(roman)
    # Drop only an inherent final schwa, never an explicit अ/आ or a long vowel.
    # Single-syllable forms such as न retain their vowel.
    if (
        tokens
        and tokens[-1] == "a"
        and word[-1] not in "अआ"
        and sum(t in _VOWELS for t in tokens) > 1
    ):
        tokens.pop()
    # Conservative medial deletion before common inflectional endings.
    # Full Hindi schwa deletion needs lexical/morphological information.
    if (
        len(tokens) >= 5
        and tokens[-2] in {"n", "t", "k"}
        and tokens[-1] in {"ā", "ī", "e", "o"}
    ):
        if tokens[-3] == "a" and tokens[-4] not in _VOWELS and tokens[-5] in _VOWELS:
            del tokens[-3]
    roman = "".join(_FOLD.get(t, t) for t in tokens)
    roman = roman.replace("।", ".").replace("॥", ".").replace("ऽ", "'")
    for digit in range(10):
        roman = roman.replace(chr(0x0966 + digit), str(digit))
    # Canonical decompositions of supported vowels/consonants are folded above.
    # Unknown symbols must fail visibly, never leak into a supposedly Latin slice.
    if not roman.isascii():
        raise ValueError(f"Unsupported Hindi Romanization: {word!r} -> {roman!r}")
    return roman


def transliterate_devanagari_to_roman(
    text: str, scheme: str = ROMANIZATION_VERSION
) -> str:
    """Convert Hindi runs, preserving existing English spelling/case and punctuation.

    Missing dependencies and unsupported characters raise errors. There is no
    silent character-replacement fallback. Legacy schemes are explicit opt-ins.
    """
    if scheme not in {ROMANIZATION_VERSION, "itrans", "hk", "iast"}:
        raise ValueError(f"Unsupported Romanization scheme: {scheme}")
    if not text.strip():
        return ""
    if scheme != ROMANIZATION_VERSION:
        from indic_transliteration import sanscript

        target = {
            "itrans": sanscript.ITRANS,
            "hk": sanscript.HK,
            "iast": sanscript.IAST,
        }[scheme]
        roman = sanscript.transliterate(text, sanscript.DEVANAGARI, target).lower()
    else:
        roman = _DEV_RUN.sub(lambda match: _hindi_word(match.group()), text)
        roman = roman.translate(str.maketrans("०१२३४५६७८९।॥॰", "0123456789..."))
        if re.search(r"[\u0900-\u097f]", roman):
            raise ValueError("Residual Devanagari after Romanization")
    return re.sub(r"\s+", " ", roman).strip()
