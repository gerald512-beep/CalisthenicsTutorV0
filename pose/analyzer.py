import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_keyframes(video_path, fps=1):
    """Extract one frame per second from the video. Returns list of BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25  # fallback for files that don't report FPS

    # How many source frames to skip between each extracted frame
    frame_interval = max(1, int(round(source_fps / fps)))

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frames.append(frame)
        frame_index += 1

    cap.release()
    logger.info("Extracted %d keyframes from %s (source FPS: %.1f)", len(frames), video_path, source_fps)

    if len(frames) < 3:
        raise ValueError(
            f"Video too short: only {len(frames)} frame(s) extracted. "
            "Please upload a video with at least 3 seconds of footage."
        )

    return frames


def run_inference(frame):
    """Run BlazePose on a single BGR frame. Returns pose_landmarks or None."""
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
    return results.pose_landmarks


def run_inference_batch(frames):
    """
    Run BlazePose on a list of BGR frames using a single Pose instance.
    Returns a list of (landmarks_or_None, frame) pairs.
    Much faster than calling run_inference() per frame — avoids 150 cold-starts.
    """
    pairs = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pairs.append((results.pose_landmarks, frame))
    return pairs


def calculate_angle(a, b, c):
    """
    Interior angle at point b using dot product / arccos.
    Always returns a value in [0, 180] — correct for joints like the elbow
    where you just want how much the joint is bent, not which direction.
    """
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / norm
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def calculate_directed_angle(a, b, c):
    """
    Directed angle at point b using arctan2, in [0, 360].
    Needed for the hip so we can distinguish sagging (< 160) from piking (> 200)
    — the direction of the deviation matters and arccos can't tell them apart.
    """
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    angle = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0]))
    return angle % 360


def check_camera_angle(landmarks, frame_width):
    """
    Heuristic side-view check: if both shoulders are visible horizontally,
    the user is facing the camera rather than filming from the side.
    Returns True if side view looks valid, False if front-facing.
    """
    lm = landmarks.landmark
    left_shoulder_x = lm[11].x * frame_width
    right_shoulder_x = lm[12].x * frame_width

    horizontal_spread = abs(left_shoulder_x - right_shoulder_x)
    logger.info("Shoulder horizontal spread: %.1fpx", horizontal_spread)

    # > 60px means both shoulders are visible side-by-side → front-facing camera
    if horizontal_spread > 60:
        logger.warning("Camera angle rejected: spread=%.1fpx (front-facing)", horizontal_spread)
        return False
    return True


def analyze_frame(landmarks, frame, frame_width, frame_height):
    """
    Calculate elbow, hip, knee, and head angles for one frame.
    Returns dict with all angle values and an empty feedback list.
    """
    lm = landmarks.landmark

    def px(idx):
        """Convert normalized landmark to pixel (x, y)."""
        return (lm[idx].x * frame_width, lm[idx].y * frame_height)

    nose       = px(0)
    l_shoulder = px(11)
    l_elbow    = px(13)
    l_wrist    = px(15)
    l_hip      = px(23)
    l_knee     = px(25)
    l_ankle    = px(27)

    # Interior angle at elbow — how much the arm is bent (0-180°)
    elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

    # Directed angle at hip — distinguishes sagging (<160°) from piking (>200°)
    # Uses knee instead of ankle so the annotation and chart measure the same segment
    hip_angle = calculate_directed_angle(l_shoulder, l_hip, l_knee)

    # Interior angle at knee — should be ~180° for a straight plank body line
    knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

    # Interior angle at shoulder — should be ~180° when head is neutral (nose/shoulder/hip collinear)
    head_angle = calculate_angle(l_hip, l_shoulder, nose)

    # Normalized x-offset kept for backward-compat with existing feedback rule
    head_offset = lm[0].x - lm[11].x

    return {
        "elbow_angle": elbow_angle,
        "hip_angle":   hip_angle,
        "knee_angle":  knee_angle,
        "head_angle":  head_angle,
        "head_offset": head_offset,
        "feedback":    [],
    }


def build_summary_feedback(avg_elbow, rep_count, n_sag_reps, n_pike_reps, n_head_frames):
    """
    Return a single personalized coaching sentence that references real measured numbers.
    Covers all combinations of elbow depth, hip alignment, and head position issues.
    """
    elbow_ok = avg_elbow < 90
    hip_ok   = n_sag_reps == 0 and n_pike_reps == 0
    head_ok  = n_head_frames == 0

    if elbow_ok and hip_ok and head_ok:
        return (f"Excellent session. Your elbows averaged {avg_elbow:.1f}° depth across "
                f"{rep_count} reps, well past the 90° target. Hip alignment stayed in the "
                f"safe zone throughout and head position was neutral.")

    if not elbow_ok and hip_ok and head_ok:
        return (f"Your reps averaged {avg_elbow:.1f}° elbow depth — try to go below 90° "
                f"at the bottom of each rep. Hip alignment and head position were solid "
                f"across all {rep_count} reps.")

    if elbow_ok and n_sag_reps > 0 and n_pike_reps == 0 and head_ok:
        return (f"Good depth at {avg_elbow:.1f}° average. Your hips dropped below 160° on "
                f"{n_sag_reps} rep(s) — brace your core harder throughout the movement. "
                f"Head position was neutral.")

    if elbow_ok and n_pike_reps > 0 and n_sag_reps == 0 and head_ok:
        return (f"Good depth at {avg_elbow:.1f}° average. Your hips rose above 200° on "
                f"{n_pike_reps} rep(s) — lower them to maintain a straight body line. "
                f"Head position was neutral.")

    if elbow_ok and hip_ok and not head_ok:
        return (f"Great depth at {avg_elbow:.1f}° and solid hip control across {rep_count} "
                f"reps. Your head jutted forward on {n_head_frames} frames — keep your "
                f"chin slightly tucked.")

    if not elbow_ok and n_sag_reps > 0 and n_pike_reps == 0 and head_ok:
        return (f"Two areas to work on: elbow depth averaged {avg_elbow:.1f}° (target below "
                f"90°), and hips sagged on {n_sag_reps} rep(s). Head position was neutral.")

    if not elbow_ok and n_pike_reps > 0 and n_sag_reps == 0 and head_ok:
        return (f"Two areas to work on: elbow depth averaged {avg_elbow:.1f}° (target below "
                f"90°), and hips rose too high on {n_pike_reps} rep(s). Head position was neutral.")

    if not elbow_ok and hip_ok and not head_ok:
        return (f"Two areas to work on: elbow depth averaged {avg_elbow:.1f}° (target below "
                f"90°), and head was jutting forward on {n_head_frames} frames. "
                f"Hip alignment was solid.")

    if elbow_ok and not hip_ok and not head_ok:
        hip_count = n_sag_reps + n_pike_reps
        return (f"Good depth at {avg_elbow:.1f}° average. Two areas to address: hip alignment "
                f"was off on {hip_count} rep(s), and head was out of position on "
                f"{n_head_frames} frames — brace your core and keep your chin slightly tucked.")

    if elbow_ok and n_sag_reps > 0 and n_pike_reps > 0 and head_ok:
        return (f"Good depth at {avg_elbow:.1f}° and neutral head position. Hip control was "
                f"inconsistent — sagged on {n_sag_reps} rep(s) and piked on {n_pike_reps} "
                f"rep(s). Focus on maintaining a rigid plank throughout each rep.")

    # Fallback: everything is off
    hip_count = n_sag_reps + n_pike_reps
    return (f"Three areas to address: depth averaged {avg_elbow:.1f}° (go below 90°), "
            f"hips were out of range on {hip_count} rep(s), and head jutted forward on "
            f"{n_head_frames} frames.")


def _draw_angle_annotation(img, vertex, p1, p2, angle_deg, label, color, arc_radius=40):
    """
    Draw two lines from vertex toward p1 and p2, an arc showing the angle, and a text label.
    All points are (x, y) pixel tuples. Modifies img in place.
    """
    try:
        v  = np.array(vertex, dtype=float)
        a  = np.array(p1,     dtype=float)
        b  = np.array(p2,     dtype=float)

        # Unit vectors along each arm of the angle
        dir_a = a - v
        norm_a = np.linalg.norm(dir_a)
        dir_b = b - v
        norm_b = np.linalg.norm(dir_b)
        if norm_a < 1e-3 or norm_b < 1e-3:
            return  # landmarks too close — skip annotation
        dir_a /= norm_a
        dir_b /= norm_b

        # cv2 needs plain Python ints/floats — cast everything explicitly
        arm_len = 60
        cx, cy = int(v[0]), int(v[1])
        end_ax, end_ay = int(v[0] + dir_a[0] * arm_len), int(v[1] + dir_a[1] * arm_len)
        end_bx, end_by = int(v[0] + dir_b[0] * arm_len), int(v[1] + dir_b[1] * arm_len)

        cv2.line(img, (cx, cy), (end_ax, end_ay), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (end_bx, end_by), color, 2, cv2.LINE_AA)

        # Arc angles (OpenCV: clockwise degrees from 3 o'clock)
        ang_a = float(np.degrees(np.arctan2(dir_a[1], dir_a[0])))
        ang_b = float(np.degrees(np.arctan2(dir_b[1], dir_b[0])))

        start_ang = float(min(ang_a, ang_b))
        end_ang   = float(max(ang_a, ang_b))
        # Use the shorter arc
        if end_ang - start_ang > 180.0:
            start_ang, end_ang = end_ang, end_ang + (360.0 - (end_ang - start_ang))

        cv2.ellipse(img, (cx, cy), (int(arc_radius), int(arc_radius)),
                    0.0, start_ang, end_ang, color, 2, cv2.LINE_AA)

        # Label at midpoint of arc, pushed outward
        mid_rad = float(np.radians((start_ang + end_ang) / 2.0))
        lx = int(v[0] + (arc_radius + 18) * np.cos(mid_rad))
        ly = int(v[1] + (arc_radius + 18) * np.sin(mid_rad))
        cv2.putText(img, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    except Exception:
        logger.exception("_draw_angle_annotation failed for label=%s", label)


def annotate_frame(frame, landmarks, elbow_angle, hip_angle, head_angle):
    """
    Draw BlazePose skeleton and angle annotations (elbow, hip, head) on a copy of the frame.
    Returns annotated BGR numpy array.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Full pose skeleton
    mp_drawing.draw_landmarks(
        annotated,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

    lm = landmarks.landmark

    def px(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Elbow angle — vertex at elbow (13), arms toward shoulder (11) and wrist (15)
    _draw_angle_annotation(
        annotated, px(13), px(11), px(15),
        elbow_angle, f"{elbow_angle:.0f}",
        color=(0, 255, 255),  # yellow
    )

    # Hip angle — vertex at hip (23), arms toward shoulder (11) and knee (25)
    _draw_angle_annotation(
        annotated, px(23), px(11), px(25),
        hip_angle, f"{hip_angle:.0f}",
        color=(0, 200, 255),  # orange
    )

    # Head angle — vertex at shoulder (11), arms toward hip (23) and nose (0)
    _draw_angle_annotation(
        annotated, px(11), px(23), px(0),
        head_angle, f"{head_angle:.0f}",
        color=(180, 255, 100),  # lime green
        arc_radius=35,
    )

    return annotated


def save_annotated_frame(frame, output_path):
    """Write annotated frame as JPEG to output_path."""
    success = cv2.imwrite(output_path, frame)
    if not success:
        raise IOError(f"Failed to save annotated frame to {output_path}")
    logger.info("Annotated frame saved: %s", output_path)
