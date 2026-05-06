# Calisthenics Tutor v0

AI-powered push-up form analyzer. Record a side-view video and get instant coaching feedback on elbow depth, hip alignment, and head position.

## What it does

- Records video in-browser via webcam
- Runs MediaPipe BlazePose pose estimation at 8 fps
- Detects reps and measures elbow, hip, and head angles per frame
- Returns annotated frame, time-series charts, per-rep depth breakdown, and a personalized coaching summary
- Email gate with one-time verification code (via Resend)
- Inline micro-survey on the results page

## Stack

- **Backend** — Python 3.12, Flask 3.0
- **Pose estimation** — MediaPipe BlazePose (CPU)
- **Email** — Resend
- **Frontend** — Vanilla JS, HTML5 Canvas charts, MediaRecorder API
- **Database** — SQLite

## Setup

### 1. Clone and install

```bash
git clone https://github.com/gerald512-beep/CalisthenicsTutorV0.git
cd CalisthenicsTutorV0
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```
RESEND_API_KEY=your_resend_api_key_here
SECRET_KEY=your_secret_key_here
```

Get a free Resend API key at [resend.com](https://resend.com). Generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Create required folders

```bash
mkdir uploads outputs
```

### 4. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Project structure

```
├── app.py                  # Flask routes and orchestration
├── pose/
│   └── analyzer.py         # BlazePose inference and angle calculations
├── templates/
│   ├── email_gate.html     # Email entry screen
│   ├── verify.html         # Code verification screen
│   ├── index.html          # Record screen
│   └── results.html        # Results page with charts and survey
├── static/
│   └── style.css
├── requirements.txt
└── .env                    # Not committed
```

## How the analysis works

1. Video is recorded at up to 8 fps via MediaRecorder
2. BlazePose runs on each frame (single model context for all frames)
3. Three joint angles are measured per frame:
   - **Elbow** — interior angle at elbow (shoulder → elbow → wrist)
   - **Hip** — directed angle at hip (shoulder → hip → knee), distinguishes sagging vs piking
   - **Head** — interior angle at shoulder (hip → shoulder → nose)
4. A state machine on the elbow angle detects reps (down < 150°, up > 160°)
5. Per-rep hip extremes determine sag/pike counts for the coaching summary

## Notes

- Film from the **side** at hip height, 2–3 meters away
- Landscape mode recommended
- 50 MB file size limit
- Requires camera permission in the browser
