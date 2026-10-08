import cv2
import mediapipe as mp

from detection.features import (
    calculate_ear,
    calculate_mar
)

from detection.head_pose import (
    estimate_head_pose
)

from detection.eye_gaze import (
    classify_gaze,
    LEFT_EYE,
    RIGHT_EYE
)


class FaceAnalyzer:

    def __init__(self, model_path):

        print("Loading MediaPipe Face Landmarker...")

        BaseOptions = mp.tasks.BaseOptions

        FaceLandmarker = (
            mp.tasks.vision.FaceLandmarker
        )

        FaceLandmarkerOptions = (
            mp.tasks.vision.FaceLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = FaceLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),

            running_mode=RunningMode.VIDEO,

            num_faces=1,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False
        )

        self.landmarker = (
            FaceLandmarker.create_from_options(
                options
            )
        )

        self.timestamp = 0

        print("MediaPipe Face Landmarker loaded.")


    def process(self, frame):

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Must continuously increase
        self.timestamp += 33

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp
        )

        if not result.face_landmarks:

            return None

        landmarks = result.face_landmarks[0]


        # ====================================================
        # HEAD POSE
        # ====================================================

        pitch, yaw, roll = estimate_head_pose(
            landmarks,
            width,
            height
        )


        # ====================================================
        # EYES
        # ====================================================

        left_eye_points = [
            (
                landmarks[i].x * width,
                landmarks[i].y * height
            )
            for i in LEFT_EYE
        ]

        right_eye_points = [
            (
                landmarks[i].x * width,
                landmarks[i].y * height
            )
            for i in RIGHT_EYE
        ]

        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        if (
            left_ear is not None
            and
            right_ear is not None
        ):

            ear = (
                left_ear +
                right_ear
            ) / 2

        elif left_ear is not None:

            ear = left_ear

        else:

            ear = right_ear


        # ====================================================
        # GAZE
        # ====================================================

        gaze = classify_gaze(
            landmarks,
            width,
            height
        )


        # ====================================================
        # MOUTH
        # ====================================================

        upper_lip = (
            landmarks[13].x * width,
            landmarks[13].y * height
        )

        lower_lip = (
            landmarks[14].x * width,
            landmarks[14].y * height
        )

        left_mouth = (
            landmarks[61].x * width,
            landmarks[61].y * height
        )

        right_mouth = (
            landmarks[291].x * width,
            landmarks[291].y * height
        )

        mar = calculate_mar(
            upper_lip,
            lower_lip,
            left_mouth,
            right_mouth
        )


        # ====================================================
        # FACE BOX
        # ====================================================

        xs = [
            p.x * width
            for p in landmarks
        ]

        ys = [
            p.y * height
            for p in landmarks
        ]

        x1 = max(0, int(min(xs)) - 10)
        y1 = max(0, int(min(ys)) - 10)

        x2 = min(width, int(max(xs)) + 10)
        y2 = min(height, int(max(ys)) + 10)


        face_crop = frame[
            y1:y2,
            x1:x2
        ].copy()


        return {

            "landmarks": landmarks,

            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,

            "ear": ear,
            "mar": mar,

            "gaze": gaze,

            "face_crop": face_crop,

            "bbox": (
                x1,
                y1,
                x2,
                y2
            )
        }


    def draw_landmarks(
        self,
        frame,
        landmarks
    ):

        height, width = frame.shape[:2]

        # ====================================================
        # FACE MESH
        # ====================================================

        for i, p in enumerate(landmarks):

            x = int(p.x * width)
            y = int(p.y * height)

            # Draw important landmarks larger
            if i in [
                1,       # nose
                33,
                133,
                263,
                362,
                468,
                473,
                13,
                14
            ]:

                radius = 3

            else:

                radius = 1

            cv2.circle(
                frame,
                (x, y),
                radius,
                (0, 255, 255),
                -1
            )


        # ====================================================
        # EYE LINES
        # ====================================================

        left_eye = [
            (int(landmarks[i].x * width),
             int(landmarks[i].y * height))
            for i in LEFT_EYE
        ]

        right_eye = [
            (int(landmarks[i].x * width),
             int(landmarks[i].y * height))
            for i in RIGHT_EYE
        ]

        cv2.polylines(
            frame,
            [__import__("numpy").array(left_eye)],
            True,
            (255, 255, 0),
            2
        )

        cv2.polylines(
            frame,
            [__import__("numpy").array(right_eye)],
            True,
            (255, 255, 0),
            2
        )


        # ====================================================
        # IRIS
        # ====================================================

        for index in [468, 473]:

            if index < len(landmarks):

                x = int(
                    landmarks[index].x *
                    width
                )

                y = int(
                    landmarks[index].y *
                    height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 255),
                    -1
                )


        return frame