# Gurbani ASR Training Data Pipeline

## Context

We're building a pipeline to create training data for a Gurmukhi ASR (Speech-to-Text) model. The pipeline downloads Gurbani recordings from YouTube, extracts their captions as Gurmukhi transcripts, segments the audio into clips aligned with caption timestamps, and outputs WAV + transcript pairs in LJSpeech format.

**Two data sources, one pipeline:**

- **Sehaj Path** — clean spoken recitation, auto-generated or manual YouTube captions
- **Kirtan** — singing with instruments, **manual captions strongly preferred** (auto-generated captions are unusable for singing)

Both use the same YouTube CC-based pipeline. They are configured as separate source groups in `sources.yaml` and produce separate dataset outputs so they can be used independently or combined for training. We build Sehaj Path first, then add Kirtan sources.

## Project Structure

```text
gurmukhi-model-training-data/
├── config/
│   ├── pipeline.yaml          # Processing parameters (sample rate, thresholds, paths)
│   └── sources.yaml           # YouTube playlist/video URLs
├── src/
│   ├── __init__.py
│   ├── gurmukhi_utils.py      # Gurmukhi Unicode detection (U+0A00-U+0A7F) & normalization
│   ├── discover.py            # YouTube download (audio + captions) via yt-dlp
│   ├── parse_captions.py      # VTT parsing, deduplication, Gurmukhi filtering
│   ├── segment_audio.py       # Audio slicing per caption timestamp (16kHz, mono, 16-bit WAV)
│   ├── quality_control.py     # Filter by duration/text quality, generate stats
│   ├── build_dataset.py       # Assemble LJSpeech-format output (wavs/ + metadata.csv)
│   └── run_pipeline.py        # CLI orchestrator — runs all stages or individual stages
├── data/                      # Created at runtime, gitignored
│   ├── raw/audio/             # Full-length WAV downloads
│   ├── raw/captions/          # VTT caption files
│   ├── raw/metadata/          # yt-dlp info JSONs
│   ├── processed/segments/    # Intermediate segment manifests
│   └── dataset/               # Final output: wavs/ + metadata.csv + stats.json
├── requirements.txt
└── .gitignore
```

## Dependencies

**Python packages:** `yt-dlp`, `pydub`, `webvtt-py`, `pyyaml`, `tqdm`

**System:** `ffmpeg` (via `brew install ffmpeg`)

## Pipeline Stages

### Stage 1: Download (`discover.py`)

- Use `yt-dlp` Python API to download audio + captions from YouTube playlists
- Audio: extract as WAV via `FFmpegExtractAudio` postprocessor
- Captions: try language codes `pa`, `pa-IN`, `hi` in order; accept auto-generated as fallback for Sehaj Path
- For Kirtan sources: skip videos without manual captions (auto-generated captions are unusable for singing)
- Save yt-dlp info JSON per video for metadata
- Skip videos already downloaded (resume support)
- Log videos with no captions to `skipped_no_captions.log`
- Output: `download_manifest.json`

### Stage 2: Parse Captions (`parse_captions.py`)

- Parse VTT files using `webvtt-py`
- **Deduplicate YouTube's rolling captions** (each cue repeats previous line + adds new one)
- Strip HTML tags and VTT positioning cues
- Normalize Gurmukhi text (Unicode NFC, collapse whitespace)
- Filter out non-Gurmukhi content using `gurmukhi_ratio()` check
- Output: `parsed_manifest.json`

### Stage 3: Segment Audio (`segment_audio.py`)

- Load full audio, convert to 16kHz / mono / 16-bit once
- Slice per caption timestamp with 100ms padding on each side
- Export as `{video_id}_{00001}.wav`
- Output: `segments_manifest.json`

### Stage 4: Quality Control (`quality_control.py`)

- Filter segments: min 0.5s, max 30s duration
- Validate Gurmukhi ratio >= 0.5, min text length >= 2 chars
- Generate rejection stats (counts by reason)
- Output: `qc_passed_manifest.json`, `qc_stats.json`

### Stage 5: Build Dataset (`build_dataset.py`)

- Copy WAVs to `data/dataset/wavs/`
- Write `metadata.csv` — pipe-delimited, no header: `filename|gurmukhi_transcript`
- Write `metadata_extended.csv` with full provenance (video ID, speaker, timestamps, caption language)
- Write `stats.json` with aggregate statistics

## Key Technical Details

**Gurmukhi detection:** Unicode block U+0A00-U+0A7F. `gurmukhi_ratio()` counts fraction of non-whitespace chars in this range.

**YouTube VTT deduplication:** Auto-generated captions use rolling display where each cue contains previous line + new line. Parser compares adjacent cues line-by-line and merges overlapping text spans.

**Config (`pipeline.yaml`):** All thresholds are configurable — sample rate, padding, min/max duration, Gurmukhi ratio threshold, caption language preferences.

**Stage-based execution:** Each stage saves a JSON manifest. If download succeeds but parsing fails, re-run from `--stage parse` without re-downloading. Also supports `--video-id` for single-video debugging.

## Output Format (LJSpeech)

```text
data/dataset/
├── wavs/
│   ├── VIDEO_ID_00001.wav    # 16kHz, mono, 16-bit
│   ├── VIDEO_ID_00002.wav
│   └── ...
└── metadata.csv              # VIDEO_ID_00001|ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ
```

## How to Run

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add YouTube URLs to config/sources.yaml, then:
python src/run_pipeline.py                          # Run all stages
python src/run_pipeline.py --stage parse            # Re-run specific stage
python src/run_pipeline.py --video-id "VIDEO_ID"    # Test with one video
```

## Verification

1. Run pipeline on a single Sehaj Path video (`--video-id`)
2. Check `data/dataset/stats.json` for segment counts and durations
3. Play a few output WAVs and compare with their transcript in `metadata.csv`
4. Verify Gurmukhi text renders correctly (UTF-8 encoding)
5. Check `metadata.csv` line count matches WAV file count in `wavs/`

## Sources Config (`sources.yaml`)

```yaml
sehaj_path:
  playlists:
    - url: "PLAYLIST_URL"
      name: "Speaker Name"
      speaker_id: "speaker_01"
  allow_auto_captions: true       # Auto-generated CC acceptable for spoken recitation

kirtan:                            # Add later
  playlists: []
  allow_auto_captions: false       # ONLY manual captions — auto CC is unusable for singing
```

## Design Decisions

1. **YouTube CC for both Sehaj Path and Kirtan** — simpler single pipeline. Accuracy tradeoff accepted for simplicity.
2. **Kirtan: manual captions only** — auto-generated captions fail on singing + instruments.
3. **LJSpeech output format** — widely supported by ASR training frameworks.
4. **Stage-based pipeline with JSON manifests** — enables resuming from any stage without re-downloading.
5. **16kHz / mono / 16-bit WAV** — standard ASR training format, balances quality and storage.
