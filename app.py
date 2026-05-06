import os
import uuid
import sqlite3
import logging
import random
from datetime import datetime, timezone
from functools import wraps

import cv2
import resend
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, session, jsonify,
)
from werkzeug.utils import secure_filename

load_dotenv()

from pose.analyzer import (
    extract_keyframes,
    run_inference_batch,
    check_camera_angle,
    analyze_frame,
    build_summary_feedback,
    annotate_frame,
    save_annotated_frame,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "database.db")

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-prod")

resend.api_key = os.environ.get("RESEND_API_KEY")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(BASE_DIR, "app.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            filename      TEXT,
            elbow_angle   REAL,
            hip_angle     REAL,
            feedback      TEXT,
            camera_valid  INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_codes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT,
            code       TEXT,
            created_at TEXT,
            used       INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT UNIQUE,
            first_seen  TEXT,
            last_seen   TEXT,
            visit_count INTEGER DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS surveys (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT,
            analysis_id   INTEGER,
            q1_retention  TEXT,
            q2_instructor TEXT,
            q3_open       TEXT,
            submitted_at  TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


def log_analysis(filename, elbow_angle, hip_angle, feedback, camera_valid):
    """Insert one analysis record and return its row id."""
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO analyses (timestamp, filename, elbow_angle, hip_angle, feedback, camera_valid)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            filename,
            elbow_angle,
            hip_angle,
            feedback,
            int(camera_valid),
        ),
    )
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info("Analysis logged to DB: file=%s elbow=%.1f hip=%.1f id=%d",
                filename, elbow_angle, hip_angle, analysis_id)
    return analysis_id


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "email" not in session:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


def _send_verification_email(email, code):
    """Send a verification code email via Resend. Raises on API failure."""
    resend.Emails.send({
        "from": "Calisthenics Tutor <onboarding@resend.dev>",
        "to": email,
        "subject": "Your access code — Calisthenics Tutor",
        "html": (
            '<div style="font-family:sans-serif;max-width:400px;'
            'margin:0 auto;padding:24px">'
            '<h2 style="margin-bottom:8px">Your access code</h2>'
            '<p style="color:#64748b;margin-bottom:24px">'
            "Enter this code to access Calisthenics Tutor</p>"
            '<div style="font-size:36px;font-weight:700;'
            "letter-spacing:8px;color:#1e293b;"
            "background:#f1f5f9;padding:16px 24px;"
            f'border-radius:8px;text-align:center">{code}</div>'
            '<p style="color:#94a3b8;font-size:12px;margin-top:24px">'
            "Valid for 10 minutes. "
            "If you did not request this, ignore it.</p>"
            "</div>"
        ),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes — auth flow
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "email" in session:
        return redirect(url_for("upload"))
    return render_template("email_gate.html")


@app.route("/gate", methods=["POST"])
def gate():
    email = request.form.get("email", "").strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        return render_template("email_gate.html",
                               error="Please enter a valid email address.")

    code = str(random.randint(100000, 999999))
    now = datetime.utcnow().isoformat()

    conn = get_db()
    conn.execute("DELETE FROM verification_codes WHERE email = ?", (email,))
    conn.execute(
        "INSERT INTO verification_codes (email, code, created_at, used) VALUES (?, ?, ?, 0)",
        (email, code, now),
    )
    conn.commit()
    conn.close()

    try:
        _send_verification_email(email, code)
    except Exception:
        logger.exception("Resend failed for %s", email)
        return render_template("email_gate.html",
                               error="Could not send email. Please try again.")

    session["pending_email"] = email
    return redirect(url_for("verify"))


@app.route("/verify", methods=["GET", "POST"])
def verify():
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("index"))

    if request.method == "GET":
        success = (
            "New code sent. Check your inbox."
            if request.args.get("msg") == "new_code_sent"
            else None
        )
        return render_template("verify.html", email=email, success=success)

    # POST — check the submitted code
    code = request.form.get("code", "").strip()

    conn = get_db()
    row = conn.execute(
        """SELECT * FROM verification_codes
           WHERE email = ? AND code = ? AND used = 0
           ORDER BY created_at DESC LIMIT 1""",
        (email, code),
    ).fetchone()
    conn.close()

    if not row:
        return render_template("verify.html", email=email,
                               error="Incorrect code. Please try again.")

    created_at = datetime.fromisoformat(row["created_at"])
    if (datetime.utcnow() - created_at).total_seconds() > 600:
        return render_template("verify.html", email=email,
                               error="Code expired. Request a new one.")

    conn = get_db()
    conn.execute(
        "UPDATE verification_codes SET used = 1 WHERE email = ? AND code = ?",
        (email, code),
    )
    now = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO users (email, first_seen, last_seen, visit_count)
           VALUES (?, ?, ?, 1)
           ON CONFLICT(email) DO UPDATE SET
             visit_count = visit_count + 1,
             last_seen   = excluded.last_seen""",
        (email, now, now),
    )
    conn.commit()
    conn.close()

    session["email"] = email
    session.pop("pending_email", None)
    logger.info("User verified: %s", email)
    return redirect(url_for("upload"))


@app.route("/resend-code", methods=["POST"])
def resend_code():
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("index"))

    conn = get_db()
    row = conn.execute(
        """SELECT created_at FROM verification_codes
           WHERE email = ? AND used = 0
           ORDER BY created_at DESC LIMIT 1""",
        (email,),
    ).fetchone()
    conn.close()

    if row:
        created_at = datetime.fromisoformat(row["created_at"])
        if (datetime.utcnow() - created_at).total_seconds() < 60:
            return render_template("verify.html", email=email,
                                   error="Please wait before requesting a new code.")

    code = str(random.randint(100000, 999999))
    now = datetime.utcnow().isoformat()

    conn = get_db()
    conn.execute("DELETE FROM verification_codes WHERE email = ?", (email,))
    conn.execute(
        "INSERT INTO verification_codes (email, code, created_at, used) VALUES (?, ?, ?, 0)",
        (email, code, now),
    )
    conn.commit()
    conn.close()

    try:
        _send_verification_email(email, code)
    except Exception:
        logger.exception("Resend failed on resend-code for %s", email)

    return redirect(url_for("verify") + "?msg=new_code_sent")


# ---------------------------------------------------------------------------
# Routes — main app
# ---------------------------------------------------------------------------

@app.route("/upload")
@login_required
def upload():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    upload_path = None

    try:
        # --- File validation ---
        if "video" not in request.files:
            logger.error("No file part in request")
            flash("No file selected.")
            return redirect(url_for("upload"))

        file = request.files["video"]

        if file.filename == "":
            logger.error("Empty filename submitted")
            flash("No file selected.")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            logger.error("Rejected file type: %s", file.filename)
            flash("Unsupported file type. Please upload an .mp4, .mov, or .avi file.")
            return redirect(url_for("upload"))

        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_BYTES:
            logger.error("File too large: %d bytes", file_size)
            flash("File is too large. Maximum size is 50 MB.")
            return redirect(url_for("upload"))

        # --- Save upload ---
        safe_name = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(upload_path)
        logger.info("Upload saved: %s (%d bytes)", upload_path, file_size)

        # --- Video duration ---
        _cap = cv2.VideoCapture(upload_path)
        _vid_fps = _cap.get(cv2.CAP_PROP_FPS) or 25
        _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        duration_seconds = int(_total_frames / _vid_fps) if _vid_fps > 0 else 0

        # --- Extract keyframes ---
        try:
            frames = extract_keyframes(upload_path, fps=8)
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("upload"))

        logger.info("Running batch inference on %d keyframes", len(frames))

        try:
            all_pairs = run_inference_batch(frames)
        except Exception:
            logger.exception("CRASH during run_inference_batch")
            raise
        valid_pairs = [(lm, fr) for lm, fr in all_pairs if lm is not None]
        logger.info("Inference complete: %d/%d frames with person detected",
                    len(valid_pairs), len(frames))

        if not valid_pairs:
            flash("No person detected. Make sure your full body is visible in the frame.")
            log_analysis(safe_name, 0.0, 0.0, "", camera_valid=False)
            return redirect(url_for("upload"))

        first_landmarks, first_frame = valid_pairs[0]
        h, w = first_frame.shape[:2]
        camera_ok = check_camera_angle(first_landmarks, w)

        if not camera_ok:
            logger.warning("Camera angle rejected for %s", safe_name)
            flash("Please film from the side, not the front.")
            log_analysis(safe_name, 0.0, 0.0, "", camera_valid=False)
            return redirect(url_for("upload"))

        # --- Aggregate angles across all valid frames ---
        min_elbow = float("inf")
        min_hip = float("inf")
        max_hip = float("-inf")
        max_head_offset = float("-inf")
        min_knee = float("inf")
        min_head_angle = float("inf")
        display_frame_idx = 0

        elbow_angles = []
        hip_angles   = []
        knee_angles  = []
        head_angles  = []

        frame_results = []
        for i, (landmarks, frame) in enumerate(valid_pairs):
            fh, fw = frame.shape[:2]
            logger.debug("Analyzing frame %d/%d", i + 1, len(valid_pairs))
            try:
                result = analyze_frame(landmarks, frame, fw, fh)
            except Exception:
                logger.exception("CRASH in analyze_frame at index %d", i)
                raise
            frame_results.append(result)

            elbow_angles.append(round(result["elbow_angle"], 1))
            hip_angles.append(round(result["hip_angle"], 1))
            knee_angles.append(round(result["knee_angle"], 1))
            head_angles.append(round(result["head_angle"], 1))

            if result["elbow_angle"] < min_elbow:
                min_elbow = result["elbow_angle"]
                display_frame_idx = i

            if result["hip_angle"] < min_hip:
                min_hip = result["hip_angle"]

            if result["hip_angle"] > max_hip:
                max_hip = result["hip_angle"]

            if result["head_offset"] > max_head_offset:
                max_head_offset = result["head_offset"]

            if result["knee_angle"] < min_knee:
                min_knee = result["knee_angle"]

            if result["head_angle"] < min_head_angle:
                min_head_angle = result["head_angle"]

        frame_count = len(valid_pairs)

        # --- Detect reps via state machine on elbow angle ---
        DOWN_THRESHOLD = 150
        UP_THRESHOLD = 160
        rep_depths    = []
        rep_boundaries = []
        in_rep = False
        current_rep_min = float("inf")
        current_rep_min_idx = 0
        current_rep_min_hip = float("inf")
        current_rep_max_hip = float("-inf")
        n_sag_reps = 0
        n_pike_reps = 0

        for i, (e_angle, h_angle) in enumerate(zip(elbow_angles, hip_angles)):
            if not in_rep and e_angle < DOWN_THRESHOLD:
                in_rep = True
                rep_boundaries.append(i)
                current_rep_min = e_angle
                current_rep_min_idx = i
                current_rep_min_hip = h_angle
                current_rep_max_hip = h_angle
            elif in_rep:
                if e_angle < current_rep_min:
                    current_rep_min = e_angle
                    current_rep_min_idx = i
                if h_angle < current_rep_min_hip:
                    current_rep_min_hip = h_angle
                if h_angle > current_rep_max_hip:
                    current_rep_max_hip = h_angle
                if e_angle > UP_THRESHOLD:
                    rep_depths.append(round(current_rep_min, 1))
                    if current_rep_min_hip < 160:
                        n_sag_reps += 1
                    if current_rep_max_hip > 200:
                        n_pike_reps += 1
                    in_rep = False
                    current_rep_min = float("inf")
                    current_rep_min_hip = float("inf")
                    current_rep_max_hip = float("-inf")

        if in_rep and current_rep_min < DOWN_THRESHOLD:
            rep_depths.append(round(current_rep_min, 1))
            if current_rep_min_hip < 160:
                n_sag_reps += 1
            if current_rep_max_hip > 200:
                n_pike_reps += 1

        rep_count = len(rep_depths)
        avg_depth = round(sum(rep_depths) / rep_count, 1) if rep_depths else None

        n_head_frames = sum(1 for a in head_angles if a < 125 or a > 155)
        avg_elbow_angle = avg_depth if avg_depth is not None else round(min_elbow, 1)

        logger.info(
            "Aggregated %d frames — reps=%d min_elbow=%.1f avg_depth=%s min_hip=%.1f "
            "max_hip=%.1f n_sag=%d n_pike=%d n_head_frames=%d",
            frame_count, rep_count, min_elbow, avg_depth, min_hip, max_hip,
            n_sag_reps, n_pike_reps, n_head_frames,
        )

        summary_feedback = build_summary_feedback(
            avg_elbow_angle, rep_count, n_sag_reps, n_pike_reps, n_head_frames,
        )
        logger.info("Summary feedback: %s", summary_feedback)

        # --- Annotate deepest-elbow frame ---
        display_result = frame_results[display_frame_idx]
        display_elbow = display_result["elbow_angle"]
        display_hip   = display_result["hip_angle"]
        display_head  = display_result["head_angle"]
        display_landmarks, display_frame = valid_pairs[display_frame_idx]

        logger.debug("Annotating frame idx=%d elbow=%.1f hip=%.1f head=%.1f",
                     display_frame_idx, display_elbow, display_hip, display_head)
        try:
            annotated = annotate_frame(display_frame, display_landmarks,
                                       display_elbow, display_hip, display_head)
        except Exception:
            logger.exception("CRASH in annotate_frame")
            raise

        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        save_annotated_frame(annotated, output_path)

        # --- Log to SQLite ---
        analysis_id = log_analysis(safe_name, display_elbow, display_hip,
                                   summary_feedback, camera_valid=True)
        session["last_analysis_id"] = analysis_id

        return render_template(
            "results.html",
            image_filename=output_filename,
            summary_feedback=summary_feedback,
            elbow_angle=display_elbow,
            hip_angle=display_hip,
            frame_count=frame_count,
            elbow_angles=elbow_angles,
            hip_angles=hip_angles,
            knee_angles=knee_angles,
            head_angles=head_angles,
            rep_count=rep_count,
            rep_depths=rep_depths,
            avg_depth=avg_depth,
            avg_elbow_angle=avg_elbow_angle,
            rep_boundaries=rep_boundaries,
            duration_seconds=duration_seconds,
            n_sag_reps=n_sag_reps,
            n_pike_reps=n_pike_reps,
            n_head_frames=n_head_frames,
            min_knee=round(min_knee, 1),
            min_head_angle=round(min_head_angle, 1),
            analysis_id=analysis_id,
            user_email=session.get("email", ""),
        )

    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info("Cleaned up upload: %s", upload_path)


@app.route("/survey", methods=["POST"])
@login_required
def survey():
    data = request.get_json(silent=True) or {}
    email = session.get("email", "")
    conn = get_db()
    conn.execute(
        """INSERT INTO surveys
             (email, analysis_id, q1_retention, q2_instructor, q3_open, submitted_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            email,
            data.get("analysis_id"),
            data.get("q1"),
            data.get("q2"),
            data.get("q3"),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    logger.info("Survey submitted: email=%s analysis_id=%s", email, data.get("analysis_id"))
    return jsonify({"status": "ok"})


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
