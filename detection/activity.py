import cv2
import numpy as np


class ActivityDetector:

    def __init__(
        self,
        model=None,
        confidence_threshold=0.60
    ):

        self.model = model

        self.threshold = (
            confidence_threshold
        )


        # Your activity dataset classes

        self.class_names = [

            "DangerousDriving",

            "Distracted",

            "Drinking",

            "SafeDriving",

            "Texting_Phone",

            "Turning",

            "Yawn",

            "close",

            "open"
        ]


    def predict(
        self,
        frame
    ):

        if self.model is None:

            return {

                "label":
                    "SafeDriving",

                "confidence": 0
            }


        try:

            image = cv2.resize(
                frame,
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
                axis=0
            )


            prediction = (
                self.model.predict(
                    image,
                    verbose=0
                )
            )


            values = np.asarray(
                prediction
            ).flatten()


            index = int(
                np.argmax(values)
            )


            confidence = float(
                values[index]
            )


            if index >= len(
                self.class_names
            ):

                label = "UNKNOWN"

            else:

                label = (
                    self.class_names[index]
                )


            if (
                confidence <
                self.threshold
            ):

                label = "SafeDriving"


            return {

                "label":
                    label,

                "confidence":
                    round(
                        confidence * 100,
                        2
                    )
            }


        except Exception as e:

            print(
                "Activity model error:",
                e
            )


            return {

                "label":
                    "UNKNOWN",

                "confidence":
                    0
            }