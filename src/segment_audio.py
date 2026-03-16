"""Segment audio files based on caption timestamps."""

import logging
from pathlib import Path

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioSegmenter:
    def __init__(self, config: dict):
        self.sample_rate = config["processing"]["sample_rate"]
        self.channels = config["processing"]["channels"]
        self.sample_width = config["processing"]["sample_width"]
        self.padding_ms = config["processing"]["padding_ms"]

    def segment_audio(
        self,
        audio_path: str,
        segments: list[dict],
        output_dir: str,
        video_id: str,
    ) -> list[dict]:
        """
        Slice audio into segments based on caption timestamps.
        Returns list of segment dicts with audio_path and transcript.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load and convert audio once
        audio = self._load_and_convert(audio_path)
        if audio is None:
            return []

        audio_duration_ms = len(audio)
        results = []

        for i, seg in enumerate(segments):
            start_ms = self._clamp(
                seg["start_ms"] - self.padding_ms, 0, audio_duration_ms
            )
            end_ms = self._clamp(
                seg["end_ms"] + self.padding_ms, 0, audio_duration_ms
            )

            if end_ms <= start_ms:
                continue

            # Slice audio
            clip = audio[start_ms:end_ms]

            # Export FLAC (lossless, ~50% smaller than WAV)
            filename = f"{video_id}_{i:05d}.flac"
            audio_path = output_path / filename
            clip.export(str(audio_path), format="flac")

            results.append(
                {
                    "audio_path": str(audio_path),
                    "filename": filename.replace(".flac", ""),
                    "transcript": seg["text_normalized"],
                    "duration_ms": end_ms - start_ms,
                    "source_video_id": video_id,
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                }
            )

        logger.info(
            f"Segmented {audio_path}: {len(results)} clips from {len(segments)} captions"
        )
        return results

    def _load_and_convert(self, audio_path: str) -> AudioSegment | None:
        """Load audio and convert to ASR-standard format (16kHz, mono, 16-bit)."""
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(self.channels)
            audio = audio.set_sample_width(self.sample_width)
            return audio
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(value, max_val))
