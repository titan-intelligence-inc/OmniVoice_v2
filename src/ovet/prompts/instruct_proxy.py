"""Map free-form emotion labels to OmniVoice's fixed instruct vocabulary.

OmniVoice's ``instruct`` parameter only accepts a closed vocabulary
(gender / age / pitch / accent / whisper). Any other token raises
``ValueError`` at generate-time. This module restricts itself to that
vocabulary.

Reference (English): see OmniVoice ``_resolve_instruct`` source.
"""
from __future__ import annotations
from dataclasses import dataclass


# Closed vocabulary tokens accepted by OmniVoice (English).
# Must match strings exactly (case + whitespace) per upstream parser.
VALID_GENDER = {"male", "female"}
VALID_AGE = {"child", "teenager", "young adult", "middle-aged", "elderly"}
VALID_PITCH = {
    "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
}
VALID_STYLE = {"whisper"}
VALID_ACCENT = {
    "american accent", "australian accent", "british accent", "canadian accent",
    "chinese accent", "indian accent", "japanese accent", "korean accent",
    "portuguese accent", "russian accent",
}
VALID_TOKENS = VALID_GENDER | VALID_AGE | VALID_PITCH | VALID_STYLE | VALID_ACCENT


# Coarse mapping from emotion label to instruct proxy.
# Limited by OmniVoice's vocabulary — these are best-effort approximations.
DEFAULT_EMOTION_PROXY: dict[str, list[str]] = {
    "anger":     ["high pitch"],
    "angry":     ["high pitch"],
    "sad":       ["low pitch"],
    "sadness":   ["low pitch"],
    "fear":      ["high pitch", "whisper"],
    "fearful":   ["high pitch", "whisper"],
    "happy":     ["high pitch"],
    "happiness": ["high pitch"],
    "joy":       ["high pitch"],
    "surprise":  ["high pitch"],
    "surprised": ["high pitch"],
    "disgust":   ["low pitch"],
    "disgusted": ["low pitch"],
    "calm":      ["moderate pitch"],
    "neutral":   ["moderate pitch"],
}


@dataclass
class SpeakerAttrs:
    gender: str | None = None     # "male" / "female"
    age:    str | None = None     # one of VALID_AGE


class InstructProxyComposer:
    """Compose an OmniVoice-vocabulary instruct string for a given emotion label."""

    def __init__(
        self,
        mapping: dict[str, list[str]] | None = None,
        prepend_attrs: bool = False,
    ):
        self.mapping = mapping or DEFAULT_EMOTION_PROXY
        self.prepend_attrs = prepend_attrs

    def compose(
        self,
        emotion_label: str | None,
        speaker_attrs: SpeakerAttrs | None = None,
    ) -> str | None:
        """Return an instruct string, or None if no proxy is applicable.

        The returned string is guaranteed to use only OmniVoice-valid tokens
        and the standard ``"a, b, c"`` comma-space separator.
        """
        tokens: list[str] = []

        if self.prepend_attrs and speaker_attrs is not None:
            if speaker_attrs.gender and speaker_attrs.gender in VALID_GENDER:
                tokens.append(speaker_attrs.gender)
            if speaker_attrs.age and speaker_attrs.age in VALID_AGE:
                tokens.append(speaker_attrs.age)

        if emotion_label:
            key = emotion_label.lower().strip()
            for tok in self.mapping.get(key, []):
                if tok in VALID_TOKENS and tok not in tokens:
                    tokens.append(tok)

        return ", ".join(tokens) if tokens else None

    @staticmethod
    def is_valid(instruct: str) -> bool:
        """True iff every comma-separated token is in OmniVoice's vocabulary."""
        if not instruct:
            return True
        parts = [p.strip() for p in instruct.split(",")]
        return all(p in VALID_TOKENS for p in parts)
