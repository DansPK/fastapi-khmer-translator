"""
PromptBuilder — centralized prompt engineering for all translation providers.

Design principles
-----------------
1. Silent-engine behavior
   The system prompt configures the model to behave like Google Translate /
   DeepL — output only, never conversational. Every sentence in the system
   prompt is an instruction, not a description.

2. Structured numbered input
   Numbered lists give models unambiguous per-item boundaries. This reduces
   off-by-one count errors on batches and improves ordering fidelity compared
   to JSON-array input prompts.

3. Technical-term preservation
   Explicitly listing term categories (not an exhaustive word list) lets the
   model generalize correctly to unseen terms, while category examples anchor
   the boundary between "keep in English" and "translate naturally".

4. Language-specific quality hints
   A short, focused hint per target language shapes the register (modern app
   Khmer vs. literary Khmer; Vietnamese software copy vs. formal Vietnamese)
   without bloating the base system prompt with rules the model ignores when
   it doesn't apply to the current request.

5. PromptPair output
   Providers consume a (system, user) pair. Gemini uses system_instruction +
   contents; OpenAI-compatible APIs (OpenRouter, DeepSeek, Qwen …) use
   role=system + role=user messages. The split maps cleanly to both.

Example generated prompts
--------------------------
System (tgt=kh):
    You are a translation engine for software localization …
    TECHNICAL TERMS — keep these in English: API, Backend …
    Khmer quality standard: use modern, natural Khmer …

User (texts=["Login to your account", "Connect to the Redis server"]):
    Target language: Khmer
    Source language: English

    Input texts:
    1. Login to your account
    2. Connect to the Redis server
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Language display-name table
# Maps TranslateKH-style codes and ISO codes to full language names used in
# the prompt. Unknown codes fall back to the raw code string so the model
# still receives actionable guidance.
# ---------------------------------------------------------------------------

_LANG_NAMES: dict[str, str] = {
    # Khmer
    "kh": "Khmer",
    "khm": "Khmer",
    "km": "Khmer",
    # English
    "eng": "English",
    "en": "English",
    # South-East Asia
    "vi": "Vietnamese",
    "viet": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "my": "Burmese",
    "lo": "Lao",
    # East Asia
    "zh": "Chinese (Simplified)",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    # European
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    # Other
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    # Special
    "auto": "the detected source language",
}

# ---------------------------------------------------------------------------
# Language-specific quality hints
# Injected into the system prompt only when translating to that language.
# Kept short and concrete — long hints get deprioritized by the model.
# ---------------------------------------------------------------------------

_QUALITY_HINTS: dict[str, str] = {
    "kh": (
        "Khmer quality: write natural, modern Khmer as used in Cambodian mobile "
        "apps, websites, and digital products. "
        "Avoid literary Khmer, archaic vocabulary, and formal government phrasing."
    ),
    "vi": (
        "Vietnamese quality: write natural Vietnamese as used in Vietnamese "
        "software products and mobile applications."
    ),
    "th": (
        "Thai quality: write natural Thai as used in Thai digital products "
        "and mobile apps."
    ),
    "id": (
        "Indonesian quality: write natural Bahasa Indonesia as used in "
        "Indonesian software products."
    ),
    "ms": (
        "Malay quality: write natural Bahasa Melayu as used in Malaysian "
        "digital products."
    ),
}

# ---------------------------------------------------------------------------
# Technical-term preserve list (illustrative categories + anchor examples)
# The model generalizes from these examples — this is not an exhaustive list.
# ---------------------------------------------------------------------------

_TECH_TERMS = (
    "API, SDK, JWT, OAuth, Token, URL, HTTP, HTTPS, SSL, SSH, UI, UX, "
    "Backend, Frontend, Database, Cache, Server, Client, Webhook, "
    "Docker, Kubernetes, Redis, Nginx, Linux, Git, GitHub, CI/CD, "
    "FastAPI, Spring Boot, Django, React, Vue, Angular, Next.js, "
    "Node.js, Python, JavaScript, TypeScript, Java, Go, Rust, "
    "OpenRouter, Gemini, OpenAI, Google, AWS, Azure, Vercel, Supabase"
)

# ---------------------------------------------------------------------------
# System prompt template
# Uses str.format() — keep braces in literal text escaped as {{ }}
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a translation engine for software localization. Translate silently — output only JSON.

OUTPUT CONTRACT (never deviate):
- Respond with ONLY this structure: {{"translate_text": ["..."]}}
- Array length MUST equal the number of input items exactly
- Preserve original order
- No markdown, no code blocks, no explanations, no extra keys

TRANSLATION QUALITY:
- Produce natural, fluent {tgt_name} as used in modern apps and websites
- Match the source tone exactly (formal / informal / technical)
- Do not add, remove, or paraphrase meaning
- Preserve all formatting, line breaks, and punctuation
{quality_hint}
TECHNICAL TERMS — keep ALL of the following in English (do not translate):
- General: {tech_terms}
- All framework, library, and programming language names
- All product names, company names, brand names, and API names
- Any technical noun that would sound unnatural if translated
Translate only the surrounding natural language, not the technical nouns.\
"""

# Appended to the system prompt on retry attempts to enforce script correctness.
_STRICT_ADDENDUM = """

CRITICAL SCRIPT ENFORCEMENT (previous attempt failed validation):
- The target language is {tgt_name}. Every translated word MUST use {tgt_name} script.
- FORBIDDEN: any Chinese (CJK), Japanese (Hiragana/Katakana/Kanji), or Korean (Hangul) characters.
- Outputting any such character causes the entire response to be rejected.
- When in doubt, use a simpler {tgt_name} word — never substitute characters from other scripts.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptPair:
    """Ready-to-use system + user prompt pair for any LLM backend."""
    system: str
    user: str


class PromptBuilder:
    """
    Builds structured translation prompts for LLM providers.

    Usage:
        builder = PromptBuilder()
        pair = builder.build(texts, src_lang="eng", tgt_lang="kh")

        # OpenAI-compatible (OpenRouter, DeepSeek, Qwen, …):
        messages = [
            {"role": "system", "content": pair.system},
            {"role": "user",   "content": pair.user},
        ]

        # Gemini (system_instruction + contents):
        payload["system_instruction"] = {"parts": [{"text": pair.system}]}
        payload["contents"]           = [{"parts": [{"text": pair.user}]}]
    """

    def build(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        strict: bool = False,
    ) -> PromptPair:
        return PromptPair(
            system=self._system(tgt_lang, strict=strict),
            user=self._user(texts, src_lang, tgt_lang),
        )

    # ------------------------------------------------------------------

    def _system(self, tgt_lang: str, strict: bool = False) -> str:
        hint = _QUALITY_HINTS.get(tgt_lang.lower(), "")
        quality_hint = f"\n{hint}\n" if hint else "\n"
        base = _SYSTEM_TEMPLATE.format(
            tgt_name=_lang_name(tgt_lang),
            quality_hint=quality_hint,
            tech_terms=_TECH_TERMS,
        )
        if strict:
            return base + _STRICT_ADDENDUM.format(tgt_name=_lang_name(tgt_lang))
        return base

    def _user(self, texts: list[str], src_lang: str, tgt_lang: str) -> str:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return (
            f"Target language: {_lang_name(tgt_lang)}\n"
            f"Source language: {_lang_name(src_lang)}\n"
            f"\n"
            f"Input texts:\n{numbered}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.lower(), code)
