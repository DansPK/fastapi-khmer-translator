"""
TranslationValidator — post-processing quality gate for all providers.

Validation strategy
-------------------
Validation is intentionally lightweight and heuristic-based. Its job is to
catch the two most common LLM failure modes for Khmer translation:

1. Script contamination
   Models that do not know Khmer well sometimes substitute Chinese, Japanese,
   or Korean characters (all visually "foreign" to the model). Any CJK / Hangul
   / Hiragana / Katakana in a Khmer-target response is always wrong.

2. Silent non-translation
   The model echoes the English source unchanged when it should have translated
   it. Detected by checking that the output contains at least one Khmer Unicode
   character when the source contains translatable natural-language words.

The validator intentionally does NOT try to measure translation accuracy — that
requires bilingual reference data we don't have. These two checks catch the
failure modes that actually occur in practice.

Unicode reference
-----------------
  Khmer block:           U+1780 – U+17FF
  Khmer Symbols:         U+19E0 – U+19FF
  CJK Unified:           U+4E00 – U+9FFF
  CJK Ext-A:             U+3400 – U+4DBF
  Hiragana:              U+3040 – U+309F
  Katakana:              U+30A0 – U+30FF
  Hangul Syllables:      U+AC00 – U+D7A3
  Hangul Jamo:           U+1100 – U+11FF
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Characters that are valid Khmer output
_KHMER = re.compile(r"[ក-៿᧠-᧿]")

# Characters that must NEVER appear in a Khmer-target translation
_INVALID_IN_KHMER = re.compile(
    r"[一-鿿"   # CJK Unified Ideographs
    r"㐀-䶿"    # CJK Extension A
    r"぀-ゟ"    # Hiragana
    r"゠-ヿ"    # Katakana
    r"가-힣"    # Hangul Syllables
    r"ᄀ-ᇿ"    # Hangul Jamo
    r"]"
)

# Technical tokens that legitimately stay in English in any translation.
# Used to strip "expected English" before deciding whether Khmer is missing.
_TECH_STRIP = re.compile(
    r"\b(?:"
    r"api|sdk|jwt|oauth|token|url|http|https|ssl|ssh|ui|ux|id|uuid|"
    r"docker|kubernetes|redis|nginx|linux|git|github|ci|cd|"
    r"fastapi|spring|boot|django|react|vue|angular|nextjs|nodejs|"
    r"python|javascript|typescript|java|go|rust|kotlin|swift|php|ruby|"
    r"openrouter|gemini|openai|google|aws|azure|vercel|supabase|"
    r"backend|frontend|database|cache|server|client|webhook|endpoint|"
    r"email|password|username|json|xml|rest|graphql|grpc|"
    r"html|css|npm|pip|yarn|mvn|gradle|bash|shell|cli|"
    r"v\d+(?:\.\d+)*"   # version strings: v2, v1.0.3
    r")\b",
    re.IGNORECASE,
)

# Language codes treated as Khmer targets
_KHMER_TARGETS = frozenset({"kh", "khm", "km"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


_OK = ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TranslationValidator:
    """
    Validates translation output for language correctness.

    Currently enforces strict rules for Khmer targets.
    For all other targets only the structural (count / empty) checks run.
    Language-specific rules for Vietnamese, Thai, etc. can be added to
    _validate_item() following the same pattern as _check_khmer().
    """

    def validate_batch(
        self,
        sources: list[str],
        translations: list[str],
        tgt_lang: str,
    ) -> ValidationResult:
        """Validate an entire batch. Returns the first failure found."""
        if len(translations) != len(sources):
            return ValidationResult(
                valid=False,
                reason=(
                    f"count mismatch: expected {len(sources)}, "
                    f"got {len(translations)}"
                ),
            )

        for i, (src, tgt) in enumerate(zip(sources, translations)):
            result = self._validate_item(i, src, tgt, tgt_lang)
            if not result.valid:
                return result

        return _OK

    # ------------------------------------------------------------------

    def _validate_item(
        self,
        idx: int,
        source: str,
        translation: str,
        tgt_lang: str,
    ) -> ValidationResult:
        if not translation.strip():
            return ValidationResult(valid=False, reason=f"item[{idx}]: empty translation")

        lang = tgt_lang.lower()

        if lang in _KHMER_TARGETS:
            return _check_khmer(idx, source, translation)

        return _OK


# ---------------------------------------------------------------------------
# Language-specific checks
# ---------------------------------------------------------------------------

def _check_khmer(idx: int, source: str, translation: str) -> ValidationResult:
    # Rule 1 — CJK / Japanese / Korean must not appear in Khmer output
    if _INVALID_IN_KHMER.search(translation):
        sample = _INVALID_IN_KHMER.search(translation).group()
        return ValidationResult(
            valid=False,
            reason=(
                f"item[{idx}]: unexpected CJK/Japanese/Korean character "
                f"'{sample}' (U+{ord(sample):04X}) detected in Khmer output"
            ),
        )

    # Rule 2 — output must contain Khmer Unicode when source has translatable words
    if _has_translatable_content(source) and not _KHMER.search(translation):
        return ValidationResult(
            valid=False,
            reason=(
                f"item[{idx}]: no Khmer Unicode characters in output "
                f"but source contains translatable content: {source!r}"
            ),
        )

    return _OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_translatable_content(text: str) -> bool:
    """
    Return True if text contains natural-language words that should be
    translated (i.e. it is not entirely composed of technical terms,
    numbers, or very short function words).

    Strips known technical tokens, then counts remaining alphabetic
    words of length >= 3. One such word is enough to expect a translation.
    """
    stripped = _TECH_STRIP.sub(" ", text)
    stripped = re.sub(r"[^a-zA-Z\s]", " ", stripped)
    meaningful_words = [w for w in stripped.split() if len(w) >= 3]
    return len(meaningful_words) >= 1
