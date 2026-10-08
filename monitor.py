import cv2
import time
from datetime import datetime

from config import (
    CAPTURE_DIR,

    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    ROLL_THRESHOLD,
    HEAD_REQUIRED_FRAMES,

    DOWNWARD_PITCH_THRESHOLD,
    DOWNWARD_REQUIRED_COUNT,
    DOWNWARD_WINDOW,

    EAR_CLOSED_THRESHOLD,
    EYE_CLOSED_REQUIRED_FRAMES,

    GAZE_REQUIRED_FRAMES,

    MAR_YAWN_THRESHOLD,
    YAWN_REQUIRED_FRAMES,

    ACTIVITY_CONFIDENCE_THRESHOLD,
    ACTIVITY_REQUIRED_FRAMES,

    CAPTURE_COOLDOWN_SECONDS,

    NTHU_RUN_EVERY,
    ACTIVITY_RUN_EVERY
)

from detection.face_analyzer import FaceAnalyzer
from detection.drowsiness import DrowsinessDetector
from detection.activity import ActivityDetector
from detection.yawn import YawnDetector

from logic.temporal import StableDirection, WindowCounter

from logic.events import determine_event

from xai.explanation import generate_explanation


class DriverMonitor:

    def __init__(
        self,
        face_model_path,
        nthu_model=None,
        activity_model=None
    ):

        print("========================================")
        print("INITIALIZING DRIVER MONITOR")
        print("========================================")

        # ----------------------------------------------------
        # FACE ANALYZER
        # ----------------------------------------------------

        self.face_analyzer = FaceAnalyzer(face_model_path)

        # ----------------------------------------------------
        # DROWSINESS
        # ----------------------------------------------------

        self.drowsiness_detector = DrowsinessDetector(
            model=nthu_model,
            ear_threshold=EAR_CLOSED_THRESHOLD,
            required_frames=EYE_CLOSED_REQUIRED_FRAMES
        )

        # ----------------------------------------------------
        # ACTIVITY
        # ----------------------------------------------------

        self.activity_detector = ActivityDetector(
            model=activity_model,
            confidence_threshold=ACTIVITY_CONFIDENCE_THRESHOLD
        )

        # ----------------------------------------------------
        # YAWN
        # ----------------------------------------------------

        self.yawn_detector = YawnDetector(
            threshold=MAR_YAWN_THRESHOLD,
            required_frames=YAWN_REQUIRED_FRAMES
        )

        # ----------------------------------------------------
        # GAZE
        # ----------------------------------------------------

        self.gaze_stable = StableDirection(
            required=GAZE_REQUIRED_FRAMES
        )

        # ----------------------------------------------------
        # DOWNWARD
        # ----------------------------------------------------

        self.downward_window = WindowCounter(
            window_size=DOWNWARD_WINDOW,
            required_count=DOWNWARD_REQUIRED_COUNT
        )

        # ----------------------------------------------------
        # HEAD CALIBRATION
        # ----------------------------------------------------

        self.calibration_required = 45

        self.calibration_frames = 0

        self.pitch_values = []
        self.yaw_values = []
        self.roll_values = []

        self.neutral_pitch = 0.0
        self.neutral_yaw = 0.0
        self.neutral_roll = 0.0

        self.calibrated = False

        # ----------------------------------------------------
        # HEAD TEMPORAL STATE
        # ----------------------------------------------------

        self.head_frames = 0

        self.last_head_direction = "CENTER"

        # ----------------------------------------------------
        # ACTIVITY TEMPORAL STATE
        # ----------------------------------------------------

        self.activity_candidate = "SafeDriving"
        self.activity_candidate_frames = 0

        self.stable_activity = {
            "label": "SafeDriving",
            "confidence": 0
        }

        # ----------------------------------------------------
        # GAZE
        # ----------------------------------------------------

        self.gaze_warning_frames = 0

        # ----------------------------------------------------
        # FRAME
        # ----------------------------------------------------

        self.frame_count = 0

        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        self.last_capture_time = 0

        self.capture_history = []

        # ----------------------------------------------------
        # EVENT HISTORY
        # ----------------------------------------------------

        self.event_history = []

        self.last_event = "SAFE"

        print("Driver Monitor initialized successfully.")
        print("========================================")

    # ========================================================
    # CALIBRATION
    # ========================================================

    def calibrate_head(self, pitch, yaw, roll):

        if self.calibrated:
            return

        self.pitch_values.append(pitch)
        self.yaw_values.append(yaw)
        self.roll_values.append(roll)

        self.calibration_frames += 1

        if self.calibration_frames >= self.calibration_required:

            self.neutral_pitch = (
                sum(self.pitch_values)
                / len(self.pitch_values)
            )

            self.neutral_yaw = (
                sum(self.yaw_values)
                / len(self.yaw_values)
            )

            self.neutral_roll = (
                sum(self.roll_values)
                / len(self.roll_values)
            )

            self.calibrated = True

            print()
            print("========================================")
            print("HEAD CALIBRATION COMPLETE")
            print("========================================")

            print(
                f"Neutral Pitch : {self.neutral_pitch:.2f}"
            )

            print(
                f"Neutral Yaw   : {self.neutral_yaw:.2f}"
            )

            print(
                f"Neutral Roll  : {self.neutral_roll:.2f}"
            )

            print("========================================")

    # ========================================================
    # HEAD DIRECTION
    # ========================================================

    def get_head_direction(
        self,
        delta_pitch,
        delta_yaw,
        delta_roll
    ):

        directions = []

        # ----------------------------------------------------
        # PITCH
        # ----------------------------------------------------

        if delta_pitch > PITCH_THRESHOLD:

            directions.append("DOWN")

        elif delta_pitch < -PITCH_THRESHOLD:

            directions.append("UP")

        # ----------------------------------------------------
        # YAW
        # ----------------------------------------------------

        if delta_yaw > YAW_THRESHOLD:

            directions.append("LEFT")

        elif delta_yaw < -YAW_THRESHOLD:

            directions.append("RIGHT")

        # ----------------------------------------------------
        # ROLL
        # ----------------------------------------------------

        if delta_roll > ROLL_THRESHOLD:

            directions.append("TILT_LEFT")

        elif delta_roll < -ROLL_THRESHOLD:

            directions.append("TILT_RIGHT")

        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        if not directions:
            return "CENTER"

        return "_".join(directions)

    # ========================================================
    # HEAD UPDATE
    # ========================================================

    def update_head(
        self,
        pitch,
        yaw,
        roll
    ):

        # ----------------------------------------------------
        # CALIBRATING
        # ----------------------------------------------------

        if not self.calibrated:

            return {
                "warning": False,
                "direction": "CALIBRATING",
                "frames": 0,

                "delta_pitch": 0,
                "delta_yaw": 0,
                "delta_roll": 0,

                "reason": ""
            }

        # ----------------------------------------------------
        # CHANGE FROM NORMAL POSITION
        # ----------------------------------------------------

        delta_pitch = pitch - self.neutral_pitch

        delta_yaw = yaw - self.neutral_yaw

        delta_roll = roll - self.neutral_roll

        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

        direction = self.get_head_direction(
            delta_pitch,
            delta_yaw,
            delta_roll
        )

        abnormal = direction != "CENTER"

        # ----------------------------------------------------
        # TEMPORAL FILTER
        # ----------------------------------------------------

        if abnormal:

            self.head_frames += 1

            self.last_head_direction = direction

        else:

            self.head_frames = 0

            self.last_head_direction = "CENTER"

        warning = (
            self.head_frames
            >= HEAD_REQUIRED_FRAMES
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        if warning:

            reason = (
                "Head remained outside the normal "
                "driving scan zone toward "
                f"{self.last_head_direction.replace('_', ' ').lower()} "
                f"for {self.head_frames} consecutive frames."
            )

        else:

            reason = ""

        return {

            "warning": warning,

            "direction": self.last_head_direction,

            "frames": self.head_frames,

            "delta_pitch": delta_pitch,

            "delta_yaw": delta_yaw,

            "delta_roll": delta_roll,

            "reason": reason
        }

    # ========================================================
    # DOWNWARD ATTENTION
    # ========================================================

    def update_downward(
        self,
        delta_pitch
    ):

        # ====================================================
        # VERY IMPORTANT
        #
        # Downward detection uses ONLY pitch.
        #
        # YAW = left/right
        # PITCH = up/down
        #
        # Therefore:
        #
        # Turning right
        #     -> yaw changes
        #     -> NOT downward
        #
        # Looking down
        #     -> pitch changes
        #     -> downward
        # ====================================================

        downward_condition = (
            delta_pitch
            > DOWNWARD_PITCH_THRESHOLD
        )

        return self.downward_window.update(
            downward_condition
        )

    # ========================================================
    # GAZE
    # ========================================================

    def update_gaze(self, gaze_direction):

        stable_direction = self.gaze_stable.update(
            gaze_direction
        )

        # CENTER
        if stable_direction == "CENTER":

            self.gaze_warning_frames = 0

            return {
                "direction": "CENTER",
                "warning": False,
                "frames": 0
            }

        # Slight / unstable gaze movements are ignored
        if stable_direction == "CALIBRATING":

            self.gaze_warning_frames = 0

            return {
                "direction": "CENTER",
                "warning": False,
                "frames": 0
            }

        self.gaze_warning_frames += 1

        warning = (
            self.gaze_warning_frames
            >= GAZE_REQUIRED_FRAMES
        )

        return {

            "direction": stable_direction,

            "warning": warning,

            "frames": self.gaze_warning_frames
        }

    # ========================================================
    # ACTIVITY
    # ========================================================

    def update_activity(self, prediction):

        label = prediction.get(
            "label",
            "SafeDriving"
        )

        confidence = prediction.get(
            "confidence",
            0
        )

        # ----------------------------------------------------
        # SAME LABEL
        # ----------------------------------------------------

        if label == self.activity_candidate:

            self.activity_candidate_frames += 1

        else:

            self.activity_candidate = label

            self.activity_candidate_frames = 1

        # ----------------------------------------------------
        # STABLE ACTIVITY
        # ----------------------------------------------------

        if (
            self.activity_candidate_frames
            >= ACTIVITY_REQUIRED_FRAMES
        ):

            self.stable_activity = {
                "label": label,
                "confidence": confidence
            }

        return self.stable_activity

    # ========================================================
    # CAPTURE EVIDENCE
    # ========================================================

    def capture_evidence(
        self,
        frame,
        event_type,
        explanation
    ):

        current_time = time.time()

        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if (
            current_time - self.last_capture_time
            < CAPTURE_COOLDOWN_SECONDS
        ):
            return None

        self.last_capture_time = current_time

        evidence = frame.copy()

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        event_name = (
            event_type
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"{event_name}_{timestamp}.jpg"
        )

        filepath = CAPTURE_DIR / filename

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        cv2.rectangle(
            evidence,
            (0, 0),
            (evidence.shape[1], 120),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            evidence,
            f"EVENT: {event_type}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        cv2.putText(
            evidence,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # ANGLES
        # ----------------------------------------------------

        cv2.putText(
            evidence,
            (
                f"Pitch: {self.current_pitch:.1f}  "
                f"Yaw: {self.current_yaw:.1f}  "
                f"Roll: {self.current_roll:.1f}"
            ),
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        short_explanation = explanation[:110]

        cv2.putText(
            evidence,
            short_explanation,
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        success = cv2.imwrite(
            str(filepath),
            evidence
        )

        if not success:

            print(
                "Failed to save:",
                filepath
            )

            return None

        capture = {

            "filename": filename,

            "event": event_type,

            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "explanation": explanation,

            "url": f"/captures/{filename}"
        }

        # ----------------------------------------------------
        # KEEP HISTORY
        # ----------------------------------------------------

        self.capture_history.insert(
            0,
            capture
        )

        self.capture_history = (
            self.capture_history[:50]
        )

        print(
            "Evidence captured:",
            filename
        )

        return capture

    # ========================================================
    # EVENT HISTORY
    # ========================================================

    def add_event_history(
        self,
        event_type,
        explanation
    ):

        if event_type == "SAFE":
            return

        if event_type == self.last_event:
            return

        entry = {

            "event": event_type,

            "time": datetime.now().strftime(
                "%H:%M:%S"
            ),

            "explanation": explanation
        }

        self.event_history.insert(
            0,
            entry
        )

        self.event_history = (
            self.event_history[:15]
        )

        self.last_event = event_type

    # ========================================================
    # OVERLAY
    # ========================================================

    def draw_overlay(
        self,
        frame,
        result
    ):

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # MAIN STATUS PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (470, 320),
            (15, 23, 42),
            -1
        )

        status = result.get(
            "status",
            "WAITING"
        )

        event = result.get(
            "event",
            "SAFE"
        )

        # ----------------------------------------------------
        # STATUS COLOR
        # ----------------------------------------------------

        if status == "ALERT":

            status_color = (
                0,
                0,
                255
            )

        elif status == "CALIBRATING":

            status_color = (
                0,
                255,
                255
            )

        else:

            status_color = (
                0,
                255,
                0
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2
        )

        # ----------------------------------------------------
        # EVENT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"EVENT: {event}",
            (25, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"HEAD: "
                f"{result.get('head_direction', 'CENTER')}"
            ),
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # GAZE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"GAZE: "
                f"{result.get('gaze_direction', 'CENTER')}"
            ),
            (25, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # ANGLES
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"P:{result.get('pitch', 0):.1f}  "
                f"Y:{result.get('yaw', 0):.1f}  "
                f"R:{result.get('roll', 0):.1f}"
            ),
            (25, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # DELTA
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"dP:{result.get('delta_pitch', 0):+.1f}  "
                f"dY:{result.get('delta_yaw', 0):+.1f}  "
                f"dR:{result.get('delta_roll', 0):+.1f}"
            ),
            (25, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 220, 255),
            1
        )

        # ----------------------------------------------------
        # EYES
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"EAR: "
                f"{result.get('ear', 0) or 0:.3f}"
                f"   Closed: "
                f"{result.get('closed_frames', 0)}"
            ),
            (25, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # MOUTH
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"MAR: "
                f"{result.get('mar', 0) or 0:.3f}"
                f"   Yawn: "
                f"{result.get('yawn_frames', 0)}"
            ),
            (25, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # DOWNWARD
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"DOWN: "
                f"{result.get('downward_count', 0)}/"
                f"{result.get('downward_window', 20)}"
            ),
            (25, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # ACTIVITY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"ACTIVITY: "
                f"{result.get('activity', 'SafeDriving')}"
            ),
            (25, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        if not self.calibrated:

            progress = (
                self.calibration_frames
                / self.calibration_required
            )

            bar_width = 350

            filled = int(
                bar_width * progress
            )

            cv2.rectangle(
                frame,
                (500, 30),
                (500 + bar_width, 55),
                (60, 60, 60),
                -1
            )

            cv2.rectangle(
                frame,
                (500, 30),
                (500 + filled, 55),
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "CALIBRATING - LOOK STRAIGHT",
                (500, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        return frame

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self, frame):

        self.frame_count += 1

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        face_data = (
            self.face_analyzer.process(frame)
        )

        if face_data is None:

            return {

                "status": "WAITING",

                "event": "NO FACE",

                "explanation":
                    "No driver's face detected.",

                "confidence": 0,

                "capture": None,

                "captures":
                    self.capture_history,

                "event_history":
                    self.event_history,

                "frame": frame
            }

        # ----------------------------------------------------
        # FACE VALUES
        # ----------------------------------------------------

        pitch = face_data["pitch"]

        yaw = face_data["yaw"]

        roll = face_data["roll"]

        ear = face_data["ear"]

        mar = face_data["mar"]

        gaze = face_data["gaze"]

        # Save current values for evidence capture.
        self.current_pitch = pitch
        self.current_yaw = yaw
        self.current_roll = roll

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        if not self.calibrated:

            self.calibrate_head(
                pitch,
                yaw,
                roll
            )

        # ----------------------------------------------------
        # HEAD
        # ----------------------------------------------------

        head = self.update_head(
            pitch,
            yaw,
            roll
        )

        delta_pitch = head.get(
            "delta_pitch",
            0
        )

        delta_yaw = head.get(
            "delta_yaw",
            0
        )

        delta_roll = head.get(
            "delta_roll",
            0
        )

        # ----------------------------------------------------
        # GAZE
        # ----------------------------------------------------

        gaze_result = self.update_gaze(
            gaze.get(
                "direction",
                "CENTER"
            )
        )

        # ----------------------------------------------------
        # DOWNWARD
        # ----------------------------------------------------

        downward = self.update_downward(
            delta_pitch
        )

        # ----------------------------------------------------
        # DROWSINESS
        # ----------------------------------------------------

        run_nthu = (
            self.frame_count
            % NTHU_RUN_EVERY == 0
        )

        drowsiness = (
            self.drowsiness_detector.process(
                ear,
                face_data["face_crop"],
                run_model=run_nthu
            )
        )

        # ----------------------------------------------------
        # ACTIVITY MODEL
        # ----------------------------------------------------

        run_activity = (
            self.frame_count
            % ACTIVITY_RUN_EVERY == 0
        )

        if run_activity:

            prediction = (
                self.activity_detector.predict(
                    frame
                )
            )

            self.update_activity(
                prediction
            )

        activity = self.stable_activity

        # ----------------------------------------------------
        # YAWN
        # ----------------------------------------------------

        yawn = self.yawn_detector.update(
            mar
        )

        # ----------------------------------------------------
        # DETERMINE EVENT
        # ----------------------------------------------------

        event = determine_event(

            drowsiness=drowsiness,

            activity=activity,

            head=head,

            gaze=gaze_result,

            yawn=yawn,

            downward_triggered=
                downward["triggered"]
        )

        event_type = event["type"]

        # ----------------------------------------------------
        # XAI DATA
        # ----------------------------------------------------

        xai_data = {

            "pitch": pitch,

            "yaw": yaw,

            "roll": roll,

            "delta_pitch":
                delta_pitch,

            "delta_yaw":
                delta_yaw,

            "delta_roll":
                delta_roll,

            "head_direction":
                head.get(
                    "direction",
                    "CENTER"
                ),

            "gaze_direction":
                gaze_result.get(
                    "direction",
                    "CENTER"
                ),

            "ear": ear,

            "closed_frames":
                drowsiness.get(
                    "closed_frames",
                    0
                ),

            "activity_confidence":
                activity.get(
                    "confidence",
                    0
                ),

            "activity":
                activity.get(
                    "label",
                    "SafeDriving"
                ),

            "mar": mar,

            "yawn_frames":
                yawn.get(
                    "frames",
                    0
                ),

            "downward_count":
                downward.get(
                    "count",
                    0
                ),

            "downward_window":
                downward.get(
                    "window",
                    0
                )
        }

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        explanation = (
            generate_explanation(
                event,
                xai_data
            )
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if not self.calibrated:

            status = "CALIBRATING"

        elif event_type == "SAFE":

            status = "SAFE"

        else:

            status = "ALERT"

        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        capture = None

        capture_events = {

            "DROWSY",

            "PHONE DISTRACTION",

            "DRINKING",

            "DANGEROUS DRIVING",

            "DOWNWARD DISTRACTION",

            "HEAD DISTRACTION",

            "YAWNING"
        }

        if (
            event_type
            in capture_events
            and self.calibrated
        ):

            capture = (
                self.capture_evidence(
                    frame,
                    event_type,
                    explanation
                )
            )

        # ----------------------------------------------------
        # EVENT HISTORY
        # ----------------------------------------------------

        self.add_event_history(
            event_type,
            explanation
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "status":
                status,

            "event":
                event_type,

            "explanation":
                explanation,

            "confidence":
                event.get(
                    "confidence",
                    0
                ),

            # ----------------------------------------------
            # HEAD
            # ----------------------------------------------

            "pitch":
                round(pitch, 2),

            "yaw":
                round(yaw, 2),

            "roll":
                round(roll, 2),

            "delta_pitch":
                round(delta_pitch, 2),

            "delta_yaw":
                round(delta_yaw, 2),

            "delta_roll":
                round(delta_roll, 2),

            "head_direction":
                head.get(
                    "direction",
                    "CENTER"
                ),

            "head_frames":
                head.get(
                    "frames",
                    0
                ),

            # ----------------------------------------------
            # GAZE
            # ----------------------------------------------

            "gaze":
                gaze.get(
                    "direction",
                    "CENTER"
                ),

            "gaze_direction":
                gaze_result.get(
                    "direction",
                    "CENTER"
                ),

            "gaze_x":
                gaze.get(
                    "x_ratio",
                    0.5
                ),

            "gaze_y":
                gaze.get(
                    "y_ratio",
                    0.5
                ),

            "gaze_warning":
                gaze_result.get(
                    "warning",
                    False
                ),

            "gaze_frames":
                gaze_result.get(
                    "frames",
                    0
                ),

            # ----------------------------------------------
            # DROWSINESS
            # ----------------------------------------------

            "drowsy":
                drowsiness.get(
                    "drowsy",
                    False
                ),

            "ear":
                drowsiness.get(
                    "ear",
                    ear
                ),

            "closed":
                drowsiness.get(
                    "closed",
                    False
                ),

            "closed_frames":
                drowsiness.get(
                    "closed_frames",
                    0
                ),

            "drowsiness_model":
                drowsiness.get(
                    "model",
                    "NOT_DROWSY"
                ),

            "drowsiness_confidence":
                drowsiness.get(
                    "model_confidence",
                    0
                ),

            # ----------------------------------------------
            # ACTIVITY
            # ----------------------------------------------

            "activity":
                activity.get(
                    "label",
                    "SafeDriving"
                ),

            "activity_confidence":
                activity.get(
                    "confidence",
                    0
                ),

            "phone":
                activity.get(
                    "label"
                ) == "Texting_Phone",

            # ----------------------------------------------
            # YAWN
            # ----------------------------------------------

            "yawn":
                yawn.get(
                    "yawn",
                    False
                ),

            "mar":
                yawn.get(
                    "mar",
                    mar
                ),

            "yawn_frames":
                yawn.get(
                    "frames",
                    0
                ),

            "total_yawns":
                yawn.get(
                    "total_yawns",
                    0
                ),

            # ----------------------------------------------
            # DOWNWARD
            # ----------------------------------------------

            "downward":
                downward.get(
                    "triggered",
                    False
                ),

            "downward_count":
                downward.get(
                    "count",
                    0
                ),

            "downward_window":
                downward.get(
                    "window",
                    0
                ),

            # ----------------------------------------------
            # CALIBRATION
            # ----------------------------------------------

            "calibrated":
                self.calibrated,

            "calibration_frames":
                self.calibration_frames,

            "calibration_required":
                self.calibration_required,

            "neutral_pitch":
                round(
                    self.neutral_pitch,
                    2
                ),

            "neutral_yaw":
                round(
                    self.neutral_yaw,
                    2
                ),

            "neutral_roll":
                round(
                    self.neutral_roll,
                    2
                ),

            # ----------------------------------------------
            # CAPTURES
            # ----------------------------------------------

            "capture":
                capture,

            "captures":
                self.capture_history,

            "event_history":
                self.event_history,

            "frame":
                frame
        }

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        frame = self.draw_overlay(
            frame,
            result
        )

        result["frame"] = frame

        return result

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            if self.face_analyzer.landmarker:

                self.face_analyzer.landmarker.close()

        except Exception:

            pass