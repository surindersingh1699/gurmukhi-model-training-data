"""End-to-end pipeline orchestrator for Gurbani ASR training data."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from discover import YouTubeDownloader
from parse_captions import CaptionParser
from segment_audio import AudioSegmenter
from quality_control import QualityController
from build_dataset import DatasetBuilder


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/pipeline.log", encoding="utf-8"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_manifest(data, filename: str):
    """Save intermediate manifest to data/ directory."""
    path = Path("data") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved manifest: {path}")


def load_manifest(filename: str):
    """Load intermediate manifest from data/ directory."""
    path = Path("data") / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_directories(config: dict):
    """Create all required data directories."""
    for key, path in config["paths"].items():
        p = Path(path)
        if not p.suffix:  # It's a directory, not a file
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)


def stage_download(config: dict, sources: dict, video_id: str | None = None):
    """Stage 1: Download audio + captions from YouTube."""
    downloader = YouTubeDownloader(config)
    results = []

    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"
        result = downloader.download_video(url, speaker_id="debug")
        if result:
            results.append(result)
    else:
        for source_type in ["sehaj_path", "kirtan"]:
            source_config = sources.get(source_type, {})
            allow_auto = source_config.get("allow_auto_captions", True)

            for playlist in source_config.get("playlists", []):
                playlist_results = downloader.download_playlist(
                    playlist["url"],
                    playlist.get("speaker_id", "unknown"),
                    allow_auto_captions=allow_auto,
                )
                results.extend(playlist_results)

            for video in source_config.get("videos", []):
                result = downloader.download_video(
                    video["url"],
                    video.get("speaker_id", "unknown"),
                    allow_auto_captions=allow_auto,
                )
                if result:
                    results.append(result)

    save_manifest(results, "download_manifest.json")
    logging.info(f"Download complete: {len(results)} videos")
    return results


def stage_parse(config: dict):
    """Stage 2: Parse VTT captions into segments."""
    download_results = load_manifest("download_manifest.json")
    parser = CaptionParser(config["quality"]["min_gurmukhi_ratio"])

    all_parsed = []
    for result in download_results:
        if not result.get("caption_path"):
            continue

        caption_path = result["caption_path"]
        if not Path(caption_path).exists():
            logging.warning(f"Caption file not found: {caption_path}")
            continue

        segments = parser.parse_vtt(caption_path)
        if segments:
            all_parsed.append(
                {
                    "video_id": result["video_id"],
                    "audio_path": result["audio_path"],
                    "segments": segments,
                    "speaker_id": result.get("speaker_id", "unknown"),
                    "caption_lang": result.get("caption_lang"),
                    "is_auto_caption": result.get("is_auto_caption", False),
                }
            )

    save_manifest(all_parsed, "parsed_manifest.json")
    logging.info(f"Parsing complete: {len(all_parsed)} videos with captions")
    return all_parsed


def stage_segment(config: dict):
    """Stage 3: Segment audio based on caption timestamps."""
    all_parsed = load_manifest("parsed_manifest.json")
    segmenter = AudioSegmenter(config)

    all_segments = []
    for video_data in all_parsed:
        segs = segmenter.segment_audio(
            video_data["audio_path"],
            video_data["segments"],
            config["paths"]["processed_segments"],
            video_data["video_id"],
        )
        # Add extra metadata to each segment
        for seg in segs:
            seg["speaker_id"] = video_data.get("speaker_id", "unknown")
            seg["caption_lang"] = video_data.get("caption_lang")
            seg["is_auto_caption"] = video_data.get("is_auto_caption", False)
        all_segments.extend(segs)

    save_manifest(all_segments, "segments_manifest.json")
    logging.info(f"Segmentation complete: {len(all_segments)} clips")
    return all_segments


def stage_qc(config: dict):
    """Stage 4: Quality control filtering."""
    all_segments = load_manifest("segments_manifest.json")
    qc = QualityController(config)

    passed, stats = qc.filter_segments(all_segments)

    save_manifest(passed, "qc_passed_manifest.json")
    save_manifest(stats, "qc_stats.json")
    logging.info(f"QC complete: {len(passed)}/{len(all_segments)} passed")
    return passed, stats


def stage_build(config: dict):
    """Stage 5: Build final LJSpeech-format dataset."""
    passed = load_manifest("qc_passed_manifest.json")
    stats = load_manifest("qc_stats.json")

    builder = DatasetBuilder(config)
    builder.build(passed, stats)
    logging.info("Dataset build complete")


def main():
    parser = argparse.ArgumentParser(
        description="Gurbani ASR Training Data Pipeline"
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml", help="Pipeline config path"
    )
    parser.add_argument(
        "--sources", default="config/sources.yaml", help="Sources config path"
    )
    parser.add_argument(
        "--stage",
        choices=["download", "parse", "segment", "qc", "build", "all"],
        default="all",
        help="Run a specific pipeline stage or all stages",
    )
    parser.add_argument(
        "--video-id", help="Process a single video ID (for testing/debugging)"
    )
    args = parser.parse_args()

    # Ensure data directory exists for logging
    Path("data").mkdir(exist_ok=True)
    setup_logging()

    config = load_config(args.config)
    sources = load_config(args.sources)
    ensure_directories(config)

    logging.info(f"Running pipeline stage: {args.stage}")

    if args.stage in ("download", "all"):
        stage_download(config, sources, video_id=args.video_id)

    if args.stage in ("parse", "all"):
        stage_parse(config)

    if args.stage in ("segment", "all"):
        stage_segment(config)

    if args.stage in ("qc", "all"):
        stage_qc(config)

    if args.stage in ("build", "all"):
        stage_build(config)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
