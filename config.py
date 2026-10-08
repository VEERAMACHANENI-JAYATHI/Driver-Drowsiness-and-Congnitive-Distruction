from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"
CAPTURE_DIR = BASE_DIR / "captures"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODELS
# ============================================================

FACE_MODEL = MODEL_DIR / "face_landmarker.task"

NTHU_MODEL = MODEL_DIR / "nthu_drowsiness_mobilenetv2.keras"

GAZE_MODEL = MODEL_DIR / "mpiigaze_mobilenetv2.keras"

ACTIVITY_MODEL = MODEL_DIR / "driver_activity_model.keras"

STATEFARM_MODEL = MODEL_DIR / "statefarm_physical_distraction.keras"


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# HEAD POSE
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
#
# These are NOT tiny movement thresholds.
#
# They represent the boundary of the normal driving
# scanning zone.
#
# Small left/right movements for checking mirrors/lanes
# remain SAFE.
# ------------------------------------------------------------

YAW_THRESHOLD = 25.0

# Looking moderately up/down is still allowed.
PITCH_THRESHOLD = 20.0

# Small head tilting is also allowed.
ROLL_THRESHOLD = 20.0


# Number of consecutive abnormal frames required.
HEAD_REQUIRED_FRAMES = 3


# ============================================================
# DOWNWARD ATTENTION
# ============================================================

# Downward detection is based ONLY on pitch.
#
# This is very important:
#
# Turning RIGHT/LEFT changes yaw.
# Looking DOWN changes pitch.
#
# Therefore a right turn can never become
# DOWNWARD DISTRACTION.
# ------------------------------------------------------------

DOWNWARD_PITCH_THRESHOLD = 22.0

# Example:
# 8 out of the latest 20 frames must be downward.
DOWNWARD_REQUIRED_COUNT = 8
DOWNWARD_WINDOW = 20


# ============================================================
# EYE / DROWSINESS
# ============================================================

EAR_CLOSED_THRESHOLD = 0.22

# Eyes must remain closed for this many consecutive frames.
EYE_CLOSED_REQUIRED_FRAMES = 15


# ============================================================
# GAZE
# ============================================================

# Normal gaze tolerance.
#
# Slight eyeball movement is normal while driving.
# ------------------------------------------------------------

GAZE_HORIZONTAL_LEFT = 0.30
GAZE_HORIZONTAL_RIGHT = 0.70

GAZE_VERTICAL_UP = 0.30
GAZE_VERTICAL_DOWN = 0.70


# Gaze has to remain away from center.
GAZE_REQUIRED_FRAMES = 5


# ============================================================
# YAWNING
# ============================================================

MAR_YAWN_THRESHOLD = 0.55

YAWN_REQUIRED_FRAMES = 3


# ============================================================
# ACTIVITY MODEL
# ============================================================

ACTIVITY_CONFIDENCE_THRESHOLD = 0.70

# Activity prediction must remain stable.
ACTIVITY_REQUIRED_FRAMES = 3


# ============================================================
# MODEL EXECUTION FREQUENCY
# ============================================================

# NTHU model is expensive.
NTHU_RUN_EVERY = 8

# Activity model is also expensive.
ACTIVITY_RUN_EVERY = 8


# ============================================================
# CAPTURE
# ============================================================

CAPTURE_COOLDOWN_SECONDS = 3


# ============================================================
# SERVER
# ============================================================

HOST = "127.0.0.1"

PORT = 5000

DEBUG = False