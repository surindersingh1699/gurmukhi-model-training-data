"""Quality control: filter segments and generate statistics."""

import logging
from collections import Counter
from pathlib import Path

from gurmukhi_utils import gurmukhi_ratio

logger = logging.getLogger(__name__)


class QualityController:
    def __init__(self, config: dict):
        self.min_duration = config["quality"]["min_duration_sec"]
        self.max_duration = config["quality"]["max_duration_sec"]
        self.min_gurmukhi_ratio = config["quality"]["min_gurmukhi_ratio"]
        self.min_text_length = config["quality"]["min_text_length"]

    def check_segment(self, segment: dict) -> tuple[bool, str]:
        """
        Validate a single segment.
        Returns (passed, reason).
        """
        transcript = segment.get("transcript", "")
        duration_sec = segment.get("duration_ms", 0) / 1000.0

        # Check text is not empty
        if not transcript or not transcript.strip():
            return False, "empty_text"

        # Check minimum text length
        if len(transcript.strip()) < self.min_text_length:
            return False, "text_too_short"

        # Check duration bounds
        if duration_sec < self.min_duration:
            return False, "too_short"
        if duration_sec > self.max_duration:
            return False, "too_long"

        # Check Gurmukhi content ratio
        ratio = gurmukhi_ratio(transcript)
        if ratio < self.min_gurmukhi_ratio:
            return False, "low_gurmukhi"

        # Check WAV file exists
        wav_path = segment.get("wav_path", "")
        if wav_path and not Path(wav_path).exists():
            return False, "missing_wav"

        return True, "ok"

    def filter_segments(self, segments: list[dict]) -> tuple[list[dict], dict]:
        """
        Filter a list of segments.
        Returns (passed_segments, stats_dict).
        """
        passed = []
        rejection_reasons = Counter()

        for seg in segments:
            ok, reason = self.check_segment(seg)
            if ok:
                passed.append(seg)
            else:
                rejection_reasons[reason] += 1

        # Calculate stats
        durations = [s["duration_ms"] / 1000.0 for s in passed]
        stats = {
            "total_segments": len(segments),
            "passed_segments": len(passed),
            "rejected_segments": len(segments) - len(passed),
            "rejection_reasons": dict(rejection_reasons),
            "total_duration_sec": sum(durations) if durations else 0,
            "avg_duration_sec": sum(durations) / len(durations) if durations else 0,
            "min_duration_sec": min(durations) if durations else 0,
            "max_duration_sec": max(durations) if durations else 0,
            "total_characters": sum(len(s["transcript"]) for s in passed),
        }

        logger.info(
            f"QC: {stats['passed_segments']}/{stats['total_segments']} passed "
            f"({stats['total_duration_sec']:.1f}s total). "
            f"Rejected: {dict(rejection_reasons)}"
        )

        return passed, stats
