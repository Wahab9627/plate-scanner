# Plate Scanner

A web app that detects and reads license plates from uploaded video footage. Upload a video, the pipeline detects each plate, crops it, runs OCR to read the characters, and returns an annotated video alongside a timestamped log of every plate seen — plus a single "best guess" answer aggregated across the whole video.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-web%20app-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

## Features

- **Drag-and-drop video upload** — no CLI needed, runs entirely in the browser
- **Async job processing** — uploads are handled in a background thread with live progress polling, so the browser never blocks on a long-running video (important since detection + OCR can take minutes on CPU)
- **Plate detection** via a hosted [Roboflow](https://roboflow.com) inference API (no local model file to manage or version)
- **OCR** via [EasyOCR](https://github.com/JaidedAI/EasyOCR) to read plate text from each detected crop
- **Result aggregation** — every plate reading across the video is grouped by exact text match; the most frequently and confidently read plate is surfaced as a single "final answer," instead of leaving the user to eyeball a noisy per-frame list
- **Browser-compatible video output** — OpenCV's default `mp4v` codec isn't playable in most browsers, so output is transcoded to H.264 automatically before being served
- **CSV export** of every detection (frame, timestamp, plate text, confidence)

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Flask, threaded background jobs |
| Video processing | OpenCV |
| Plate detection | Roboflow hosted inference API |
| OCR | EasyOCR |
| Video transcoding | ffmpeg (via `imageio-ffmpeg`) |
| Frontend | Vanilla HTML/CSS/JS, no framework |
| Deployment | Gunicorn + `Procfile` (Render-compatible) |

## Architecture

```
Upload → Flask saves file → background thread starts
                                    │
                                    ▼
                    OpenCV reads video frame-by-frame
                                    │
                    every Nth frame → Roboflow API (plate detection)
                                    │
                    detected crop → EasyOCR (text recognition)
                                    │
                    results deduplicated + aggregated
                                    │
                                    ▼
              annotated video (H.264) + CSV log + best-guess summary
                                    │
                                    ▼
                    frontend polls /api/status until done
```

## Getting started

### Prerequisites

- Python 3.11 (newer versions may hit dependency resolution issues with some packages used here)
- A free [Roboflow](https://roboflow.com) account and API key

### Setup

```bash
git clone https://github.com/Wahab9627/plate-scanner.git
cd plate-scanner

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root (never committed — see `.gitignore`):

```
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_MODEL_ID=license-plate-recognition-rxg4e/11
```

### Run locally

```bash
python app.py
```

Visit `http://localhost:5000`, upload a video, and watch it process.

## Project structure

```
aiapp/
├── app.py              # Flask routes, job queue, status API
├── pipeline.py          # Detection + OCR + video processing core
├── requirements.txt
├── Procfile              # gunicorn start command for deployment
├── .gitignore
├── static/
│   ├── style.css
│   └── bg-lambo.jpg
├── templates/
│   ├── index.html        # Upload page
│   └── results.html       # Live progress + results page
├── uploads/               # Temp storage for incoming videos (gitignored)
└── outputs/               # Annotated videos + CSV logs (gitignored)
```

## Known limitations

- Plate readings are grouped by **exact text match** when computing the "final answer" — a single mis-read character creates a separate group rather than merging with the correct reading. Fuzzy matching (e.g. allowing 1-character tolerance) would improve this.
- The in-memory job store (`jobs = {}` in `app.py`) doesn't survive a server restart. Fine for a demo; a real deployment would move this to Redis or a database.
- EasyOCR depends on PyTorch, which makes the app too memory-heavy for most free hosting tiers (512MB RAM caps). A planned follow-up is swapping to Tesseract OCR to cut the memory footprint significantly for cheaper/free hosting.

## Roadmap

- [ ] Swap EasyOCR → Tesseract to reduce memory footprint for free-tier hosting
- [ ] Fuzzy-match plate groupings for the "final answer" summary
- [ ] Move job state to persistent storage (Redis/DB) for multi-worker deployment
- [ ] Deploy a live demo

## License

MIT — free to use, modify, and learn from.

## Author

Built by [Wahab9627](https://github.com/Wahab9627)