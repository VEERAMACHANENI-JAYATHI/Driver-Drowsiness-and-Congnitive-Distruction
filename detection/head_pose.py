import cv2
import numpy as np


# ============================================================
# 3D GENERIC FACE MODEL
# Coordinates are approximate real-world facial positions.
# ============================================================

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip - landmark 1
    (0.0, -330.0, -65.0),     # Chin - landmark 152
    (-225.0, 170.0, -135.0),  # Left eye - landmark 33
    (225.0, 170.0, -135.0),   # Right eye - landmark 263
    (-150.0, -150.0, -125.0), # Left mouth - landmark 61
    (150.0, -150.0, -125.0)   # Right mouth - landmark 291
], dtype=np.float64)


LANDMARK_IDS = [
    1,      # Nose
    152,    # Chin
    33,     # Left eye
    263,    # Right eye
    61,     # Left mouth
    291     # Right mouth
]


def get_camera_matrix(width, height):
    """
    Approximate camera matrix using webcam image dimensions.
    """

    focal_length = width

    center = (
        width / 2.0,
        height / 2.0
    )

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    return camera_matrix


def get_distortion_coefficients():
    """
    We assume negligible lens distortion.
    """

    return np.zeros((4, 1), dtype=np.float64)


def get_2d_points(landmarks, width, height):
    """
    Convert MediaPipe normalized landmarks
    into pixel coordinates.
    """

    image_points = []

    for landmark_id in LANDMARK_IDS:

        point = landmarks[landmark_id]

        x = point.x * width
        y = point.y * height

        image_points.append((x, y))

    return np.array(image_points, dtype=np.float64)


def rotation_vector_to_angles(rotation_vector):
    """
    Convert rotation vector to Euler angles.

    Returns:
        pitch
        yaw
        roll
    """

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(
        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        pitch = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )

    else:

        pitch = np.arctan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = 0

    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return float(pitch), float(yaw), float(roll)


def estimate_head_pose(landmarks, width, height):
    """
    Main PnP head-pose function.

    MediaPipe 2D landmarks
             ↓
    3D generic face model
             ↓
        solvePnP()
             ↓
       rotation vector
             ↓
     Euler angles
             ↓
    pitch / yaw / roll
    """

    if landmarks is None:
        return 0.0, 0.0, 0.0

    if len(landmarks) <= max(LANDMARK_IDS):
        return 0.0, 0.0, 0.0

    image_points = get_2d_points(
        landmarks,
        width,
        height
    )

    camera_matrix = get_camera_matrix(
        width,
        height
    )

    distortion_coefficients = get_distortion_coefficients()

    try:

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        pitch, yaw, roll = rotation_vector_to_angles(
            rotation_vector
        )

        return pitch, yaw, roll

    except Exception as e:

        print("Head pose error:", e)

        return 0.0, 0.0, 0.0