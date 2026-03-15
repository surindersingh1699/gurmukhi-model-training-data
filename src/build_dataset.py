"""Assemble the final LJSpeech-format dataset."""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, config: dict):
        self.wavs_dir = Path(config["paths"]["dataset_wavs"])
        self.metadata_path = Path(config["paths"]["dataset_metadata"])
        self.metadata_ext_path = Path(config["paths"]["dataset_metadata_extended"])
        self.stats_path = Path(config["paths"]["dataset_stats"])

    def build(self, segments: list[dict], stats: dict):
        """
        Assemble the final dataset:
        1. Copy WAVs to dataset/wavs/
        2. Write metadata.csv (LJSpeech format)
        3. Write metadata_extended.csv (full provenance)
        4. Write stats.json
        """
        self.wavs_dir.mkdir(parents=True, exist_ok=True)

        metadata_lines = []
        extended_lines = []
        extended_lines.append(
            "id|transcript|duration_sec|source_video_id|start_ms|end_ms"
        )

        copied = 0
        for seg in segments:
            src = Path(seg["wav_path"])
            if not src.exists():
                logger.warning(f"WAV not found, skipping: {src}")
                continue

            dst = self.wavs_dir / src.name
            if src != dst:
                shutil.copy2(str(src), str(dst))

            filename = seg["filename"]
            transcript = seg["transcript"]

            # LJSpeech format: filename|transcript (no header, pipe-delimited)
            metadata_lines.append(f"{filename}|{transcript}")

            # Extended format with provenance
            duration_sec = seg["duration_ms"] / 1000.0
            extended_lines.append(
                f"{filename}|{transcript}|{duration_sec:.3f}|"
                f"{seg['source_video_id']}|{seg['start_ms']}|{seg['end_ms']}"
            )
            copied += 1

        # Write metadata.csv
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")

        # Write metadata_extended.csv
        with open(self.metadata_ext_path, "w", encoding="utf-8") as f:
            f.write("\n".join(extended_lines) + "\n")

        # Write stats.json
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Dataset built: {copied} segments -> {self.wavs_dir}, "
            f"metadata -> {self.metadata_path}"
        )
