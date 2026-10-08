class YawnDetector:

    def __init__(
        self,
        threshold=0.55,
        required_frames=3
    ):

        self.threshold = threshold

        self.required_frames = required_frames

        self.frames = 0

        self.total_yawns = 0

        self.was_yawning = False


    def update(self, mar):

        if mar is None:

            self.frames = 0

            self.was_yawning = False

            return {
                "yawn": False,
                "mar": None,
                "frames": 0,
                "total_yawns": self.total_yawns,
                "reason": ""
            }


        # ----------------------------------------------------
        # MOUTH OPEN
        # ----------------------------------------------------

        if mar >= self.threshold:

            self.frames += 1

        else:

            self.frames = 0

            self.was_yawning = False


        # ----------------------------------------------------
        # CONFIRM YAWN
        # ----------------------------------------------------

        yawn = (
            self.frames >=
            self.required_frames
        )


        # Count once per yawn

        if (
            yawn
            and not self.was_yawning
        ):

            self.total_yawns += 1

            self.was_yawning = True


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        if yawn:

            reason = (
                "Mouth remained open for "
                f"{self.frames} consecutive frames "
                f"(MAR={mar:.3f}), indicating a possible yawn."
            )

        else:

            reason = ""


        return {

            "yawn": yawn,

            "mar": round(
                float(mar),
                3
            ),

            "frames": self.frames,

            "total_yawns":
                self.total_yawns,

            "reason": reason
        }