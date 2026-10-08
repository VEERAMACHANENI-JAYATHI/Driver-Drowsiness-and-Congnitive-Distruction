import cv2
import numpy as np


class DrowsinessDetector:

    def __init__(
        self,
        model=None,
        ear_threshold=0.22,
        required_frames=15
    ):

        self.model = model

        self.ear_threshold = ear_threshold

        self.required_frames = required_frames

        self.closed_frames = 0

        self.last_model_result = {

            "drowsy": False,

            "confidence": 0,

            "label":
                "NOT_DROWSY"
        }


    def update_ear(
        self,
        ear
    ):

        if ear is None:

            self.closed_frames = 0

            return False


        if ear < self.ear_threshold:

            self.closed_frames += 1

        else:

            self.closed_frames = 0


        return (
            self.closed_frames >=
            self.required_frames
        )


    def predict_model(
        self,
        face_image
    ):

        if (
            self.model is None
            or
            face_image is None
        ):

            return self.last_model_result


        try:

            image = cv2.resize(
                face_image,
                (224, 224)
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = (
                image.astype(
                    np.float32
                ) / 255.0
            )

            image = np.expand_dims(
                image,
                0
            )

            prediction = (
                self.model.predict(
                    image,
                    verbose=0
                )
            )

            values = (
                np.asarray(
                    prediction
                ).flatten()
            )


            if len(values) == 1:

                probability = float(
                    values[0]
                )

                drowsy = (
                    probability >= 0.5
                )

            else:

                index = int(
                    np.argmax(values)
                )

                probability = float(
                    values[index]
                )

                # Change to index 1 only if your
                # NTHU model uses:
                # 0 = NOT_DROWSY
                # 1 = DROWSY

                drowsy = (
                    index == 0
                )


            self.last_model_result = {

                "drowsy": drowsy,

                "confidence":
                    round(
                        probability * 100,
                        2
                    ),

                "label":
                    "DROWSY"
                    if drowsy
                    else
                    "NOT_DROWSY"
            }


        except Exception as e:

            print(
                "NTHU error:",
                e
            )


        return self.last_model_result


    def process(
        self,
        ear,
        face_image=None,
        run_model=False
    ):

        ear_drowsy = self.update_ear(
            ear
        )


        if run_model:

            self.predict_model(
                face_image
            )


        # ====================================================
        # FINAL
        # ====================================================

        final_drowsy = (

            ear_drowsy

            or

            self.last_model_result[
                "drowsy"
            ]
        )


        if ear_drowsy:

            reason = (
                "Eyes remained closed for "
                f"{self.closed_frames} "
                "consecutive frames."
            )

        elif self.last_model_result["drowsy"]:

            reason = (
                "NTHU drowsiness model "
                "detected possible drowsiness."
            )

        else:

            reason = ""


        return {

            "drowsy":
                final_drowsy,

            "ear":
                round(
                    float(ear),
                    3
                )
                if ear is not None
                else None,

            "closed":
                self.closed_frames > 0,

            "closed_frames":
                self.closed_frames,

            "model":
                self.last_model_result[
                    "label"
                ],

            "model_confidence":
                self.last_model_result[
                    "confidence"
                ],

            "reason":
                reason
        }