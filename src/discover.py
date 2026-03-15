"""YouTube discovery and download: audio + captions via yt-dlp."""

import json
import logging
import os
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_audio_dir = Path(config["paths"]["raw_audio"])
        self.raw_captions_dir = Path(config["paths"]["raw_captions"])
        self.raw_metadata_dir = Path(config["paths"]["raw_metadata"])
        self.caption_langs = config["captions"]["preferred_langs"]
        self.allow_auto = config["captions"]["allow_auto_generated"]

    def download_playlist(
        self, playlist_url: str, speaker_id: str, allow_auto_captions: bool = True
    ) -> list[dict]:
        """Download all videos from a YouTube playlist. Returns list of result dicts."""
        results = []
        # First, extract playlist info to get video URLs
        extract_opts = {
            "quiet": True,
            "extract_flat": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(extract_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            if not info or "entries" not in info:
                logger.warning(f"No entries found in playlist: {playlist_url}")
                return results

            entries = list(info["entries"])
            logger.info(
                f"Found {len(entries)} videos in playlist: {info.get('title', 'Unknown')}"
            )

        for entry in entries:
            video_id = entry.get("id") or entry.get("url")
            if not video_id:
                continue
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            result = self.download_video(video_url, speaker_id, allow_auto_captions)
            if result:
                results.append(result)

        return results

    def download_video(
        self, video_url: str, speaker_id: str, allow_auto_captions: bool = True
    ) -> dict | None:
        """
        Download audio + captions for a single video.
        Returns result dict or None if download fails.
        """
        # Extract video ID from URL
        video_id = self._extract_video_id(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url}")
            return None

        # Skip if already downloaded
        audio_path = self.raw_audio_dir / f"{video_id}.wav"
        if audio_path.exists():
            logger.info(f"Skipping {video_id} — already downloaded")
            # Still try to return metadata if available
            return self._load_existing_metadata(video_id)

        logger.info(f"Downloading: {video_url}")

        # Step 1: Download audio only (no subtitles — avoids rate limit errors)
        audio_opts = self._build_audio_opts()
        audio_opts["outtmpl"] = {
            "default": str(self.raw_audio_dir / "%(id)s.%(ext)s"),
        }

        try:
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Failed to download audio {video_url}: {e}")
            return None

        # Step 2: Download subtitles separately, one language at a time
        self._download_subtitles(video_url, video_id, allow_auto_captions)

        # Move caption files to captions directory
        caption_path, caption_lang, is_auto = self._find_and_move_captions(video_id)

        if not caption_path:
            logger.warning(f"No captions found for {video_id}")
            self._log_skipped(video_id, video_url)
            # Still keep the audio — captions might be added manually later
            if not allow_auto_captions:
                return None

        # Save metadata
        result = {
            "video_id": video_id,
            "title": info.get("title", ""),
            "url": video_url,
            "audio_path": str(audio_path),
            "caption_path": str(caption_path) if caption_path else None,
            "caption_lang": caption_lang,
            "is_auto_caption": is_auto,
            "speaker_id": speaker_id,
            "duration": info.get("duration"),
            "uploader": info.get("uploader", ""),
        }

        # Save metadata JSON
        meta_path = self.raw_metadata_dir / f"{video_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def _build_audio_opts(self) -> dict:
        """Construct yt-dlp options for audio-only download (no subtitles)."""
        return {
            "format": self.config["audio"]["format"],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "writesubtitles": False,
            "writeautomaticsub": False,
            "writeinfojson": False,
            "quiet": False,
            "no_warnings": False,
            "noprogress": False,
        }

    def _download_subtitles(
        self, video_url: str, video_id: str, allow_auto_captions: bool
    ):
        """Download Gurmukhi (pa) subtitles separately from audio."""
        sub_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": allow_auto_captions and self.allow_auto,
            "subtitleslangs": ["pa"],
            "subtitlesformat": self.config["captions"]["format"],
            "outtmpl": {"default": str(self.raw_audio_dir / "%(id)s.%(ext)s")},
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                ydl.extract_info(video_url, download=True)
            logger.info(f"Downloaded Gurmukhi subtitles for {video_id}")
        except yt_dlp.utils.DownloadError as e:
            logger.warning(f"Could not download subtitles for {video_id}: {e}")

    def _extract_video_id(self, url: str) -> str | None:
        """Extract YouTube video ID from various URL formats."""
        import re

        patterns = [
            r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",  # bare video ID
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _find_and_move_captions(
        self, video_id: str
    ) -> tuple[Path | None, str | None, bool]:
        """
        Find downloaded caption files and move to captions directory.
        yt-dlp names them like: VIDEO_ID.pa.vtt or VIDEO_ID.pa-IN.vtt
        Returns (caption_path, language_code, is_auto_generated).
        """
        # Check for captions in audio directory (where yt-dlp puts them)
        for lang in self.caption_langs:
            # Manual captions first
            for suffix in [f".{lang}.vtt", f".{lang}-orig.vtt"]:
                src = self.raw_audio_dir / f"{video_id}{suffix}"
                if src.exists():
                    dst = self.raw_captions_dir / f"{video_id}.{lang}.vtt"
                    src.rename(dst)
                    return dst, lang, False

        # Then auto-generated
        for lang in self.caption_langs:
            src = self.raw_audio_dir / f"{video_id}.{lang}.vtt"
            if src.exists():
                dst = self.raw_captions_dir / f"{video_id}.{lang}.vtt"
                src.rename(dst)
                return dst, lang, True

        # Check for any VTT file matching the video ID
        for f in self.raw_audio_dir.glob(f"{video_id}.*.vtt"):
            lang = f.stem.split(".", 1)[1] if "." in f.stem else "unknown"
            dst = self.raw_captions_dir / f.name
            f.rename(dst)
            return dst, lang, True

        return None, None, False

    def _load_existing_metadata(self, video_id: str) -> dict | None:
        """Load previously saved metadata for an already-downloaded video."""
        meta_path = self.raw_metadata_dir / f"{video_id}.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _log_skipped(self, video_id: str, url: str):
        """Log videos skipped due to missing captions."""
        log_path = Path(self.config["paths"]["raw_metadata"]) / "skipped_no_captions.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{video_id}\t{url}\n")
