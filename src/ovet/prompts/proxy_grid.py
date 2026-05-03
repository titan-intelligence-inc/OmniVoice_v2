"""Proxy grid: enumerate instruct candidates per emotion.

Reads from a config-supplied dict; each value is a list of instruct
strings (or ``None`` = baseline = no instruct).
"""
from __future__ import annotations
from .instruct_proxy import InstructProxyComposer, SpeakerAttrs


class ProxyGrid:
    """Generate ``CandidateSpec``-friendly instruct strings for an emotion."""

    def __init__(
        self,
        grid: dict[str, list[str | None]],
        composer: InstructProxyComposer | None = None,
    ):
        self.grid = grid
        self.composer = composer or InstructProxyComposer()

    def for_emotion(
        self,
        emotion_label: str | None,
        speaker_attrs: SpeakerAttrs | None = None,
    ) -> list[tuple[str, str | None]]:
        """Return a list of ``(tag, instruct_str_or_None)`` pairs.

        Tag is a short, filename-safe identifier suitable for output paths.
        Always includes a ``baseline`` (None) entry so we can compare.
        """
        key = (emotion_label or "neutral").lower().strip()
        raw = self.grid.get(key) or [None, self.composer.compose(emotion_label, speaker_attrs)]

        out: list[tuple[str, str | None]] = []
        seen: set[str | None] = set()
        for entry in raw:
            instruct = self._validate(entry)
            if instruct in seen:
                continue
            seen.add(instruct)
            tag = self._tag(instruct)
            out.append((tag, instruct))

        # Guarantee at least a baseline.
        if not any(t == "baseline" for t, _ in out):
            out.insert(0, ("baseline", None))
        return out

    @staticmethod
    def _validate(entry: str | None) -> str | None:
        if entry is None or entry == "":
            return None
        # Reject anything that contains tokens outside OmniVoice's vocabulary.
        if not InstructProxyComposer.is_valid(entry):
            raise ValueError(f"proxy_grid contains invalid instruct: {entry!r}")
        return entry

    @staticmethod
    def _tag(instruct: str | None) -> str:
        if instruct is None:
            return "baseline"
        # filename-safe-ish — strip spaces and commas
        return (
            instruct.replace(",", "_").replace(" ", "_")
            .replace("__", "_").strip("_")
        )
