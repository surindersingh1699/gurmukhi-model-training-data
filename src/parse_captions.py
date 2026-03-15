"""Parse YouTube VTT caption files into structured segments."""

import logging
import re
from dataclasses import dataclass, asdict

import webvtt

from gurmukhi_utils import gurmukhi_ratio, normalize_gurmukhi, strip_non_gurmukhi

logger = logging.getLogger(__name__)


@dataclass
class CaptionSegment:
    start_ms: int
    end_ms: int
    text: str
    text_normalized: str
    is_gurmukhi: bool


class CaptionParser:
    def __init__(self, min_gurmukhi_ratio: float = 0.5):
        self.min_gurmukhi_ratio = min_gurmukhi_ratio

    def parse_vtt(self, vtt_path: str) -> list[dict]:
        """
        Parse a VTT file and return list of segment dicts.
        Handles YouTube's rolling caption deduplication.
        """
        try:
            captions = webvtt.read(vtt_path)
        except Exception as e:
            logger.error(f"Failed to parse VTT file {vtt_path}: {e}")
            return []

        # Convert to CaptionSegments
        raw_segments = []
        for caption in captions:
            start_ms = self._timestamp_to_ms(caption.start)
            end_ms = self._timestamp_to_ms(caption.end)
            text = caption.text
            normalized = normalize_gurmukhi(text)
            # Strip all non-Gurmukhi characters (removes stray <, numbers, latin chars)
            normalized = strip_non_gurmukhi(normalized)

            if not normalized:
                continue

            raw_segments.append(
                CaptionSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    text_normalized=normalized,
                    is_gurmukhi=gurmukhi_ratio(normalized) >= self.min_gurmukhi_ratio,
                )
            )

        # Deduplicate YouTube's rolling captions
        deduped = self._deduplicate(raw_segments)

        # Filter to Gurmukhi-only segments
        gurmukhi_segments = [s for s in deduped if s.is_gurmukhi]

        logger.info(
            f"Parsed {vtt_path}: {len(raw_segments)} raw -> "
            f"{len(deduped)} deduped -> {len(gurmukhi_segments)} Gurmukhi"
        )

        return [asdict(s) for s in gurmukhi_segments]

    def _timestamp_to_ms(self, timestamp: str) -> int:
        """Convert 'HH:MM:SS.mmm' to milliseconds."""
        # Handle both HH:MM:SS.mmm and MM:SS.mmm formats
        parts = timestamp.split(":")
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = "0"
            m, s = parts
        else:
            return 0

        seconds, _, millis = s.partition(".")
        millis = millis.ljust(3, "0")[:3]  # Ensure exactly 3 digits

        return (
            int(h) * 3600000
            + int(m) * 60000
            + int(seconds) * 1000
            + int(millis)
        )

    def _deduplicate(self, segments: list[CaptionSegment]) -> list[CaptionSegment]:
        """
        Deduplicate YouTube's rolling captions.

        YouTube auto-generated VTT uses a rolling display pattern:
          00:00:01.000 --> 00:00:03.000
          ਵਾਹਿਗੁਰੂ

          00:00:02.500 --> 00:00:05.000
          ਵਾਹਿਗੁਰੂ
          ਜੀ ਕਾ ਖਾਲਸਾ

        This produces duplicate text. We detect repeated lines between
        adjacent cues and extract only the new content.
        """
        if not segments:
            return []

        result = []
        prev_lines = set()

        for seg in segments:
            current_lines = seg.text_normalized.split("\n")
            # Find lines that are new (not in previous cue)
            new_lines = [line for line in current_lines if line.strip() not in prev_lines]

            if new_lines:
                new_text = " ".join(line.strip() for line in new_lines if line.strip())
                if new_text:
                    result.append(
                        CaptionSegment(
                            start_ms=seg.start_ms,
                            end_ms=seg.end_ms,
                            text=seg.text,
                            text_normalized=new_text,
                            is_gurmukhi=gurmukhi_ratio(new_text) >= self.min_gurmukhi_ratio,
                        )
                    )

            prev_lines = {line.strip() for line in current_lines if line.strip()}

        return result
