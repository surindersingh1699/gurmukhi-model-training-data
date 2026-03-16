"""Assemble the final dataset in HuggingFace datasets format."""

import json
import logging
from pathlib import Path

from datasets import Audio, Dataset, Features, Value

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, config: dict):
        self.dataset_dir = Path(config["paths"]["dataset_dir"])
        self.stats_path = Path(config["paths"]["dataset_stats"])
        self.sample_rate = config["processing"]["sample_rate"]

    def build(self, segments: list[dict], stats: dict):
        """
        Assemble the final HuggingFace dataset:
        1. Build dataset from QC-passed segments
        2. Save to disk in Arrow/Parquet format
        3. Write stats.json
        """
        # Collect valid segments
        records = []
        for seg in segments:
            audio_path = seg.get("audio_path", "")
            if not audio_path or not Path(audio_path).exists():
                logger.warning(f"Audio not found, skipping: {audio_path}")
                continue

            records.append(
                {
                    "audio": audio_path,
                    "transcription": seg["transcript"],
                    "speaker_id": seg.get("speaker_id", "unknown"),
                    "source_video_id": seg.get("source_video_id", ""),
                    "duration_sec": seg["duration_ms"] / 1000.0,
                }
            )

        if not records:
            logger.warning("No valid segments to build dataset from")
            return

        # Create HuggingFace Dataset
        features = Features(
            {
                "audio": Audio(sampling_rate=self.sample_rate),
                "transcription": Value("string"),
                "speaker_id": Value("string"),
                "source_video_id": Value("string"),
                "duration_sec": Value("float64"),
            }
        )

        ds = Dataset.from_dict(
            {
                "audio": [r["audio"] for r in records],
                "transcription": [r["transcription"] for r in records],
                "speaker_id": [r["speaker_id"] for r in records],
                "source_video_id": [r["source_video_id"] for r in records],
                "duration_sec": [r["duration_sec"] for r in records],
            },
            features=features,
        )

        # Save to disk
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(self.dataset_dir))

        # Write stats.json
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Dataset built: {len(records)} segments -> {self.dataset_dir} "
            f"(HuggingFace format, loadable with load_from_disk)"
        )
