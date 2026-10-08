def determine_event(
    drowsiness,
    activity,
    head,
    gaze,
    yawn,
    downward_triggered
):

    # ========================================================
    # 1. DROWSINESS
    # ========================================================

    if drowsiness.get(
        "drowsy",
        False
    ):

        return {

            "type":
                "DROWSY",

            "reason":
                drowsiness.get(
                    "reason",
                    "Possible drowsiness detected."
                ),

            "confidence":
                drowsiness.get(
                    "model_confidence",
                    0
                )
        }

    # ========================================================
    # 2. YAWNING
    #
    # Yawning gets priority over distraction.
    # Therefore an open mouth won't become
    # DOWNWARD DISTRACTION.
    # ========================================================

    if yawn.get(
        "yawn",
        False
    ):

        return {

            "type":
                "YAWNING",

            "reason":
                yawn.get(
                    "reason",
                    "Possible yawning detected."
                ),

            "confidence":
                0
        }

    # ========================================================
    # 3. PHONE
    # ========================================================

    if (
        activity.get("label")
        == "Texting_Phone"
        and activity.get(
            "confidence",
            0
        )
        >= 70
    ):

        return {

            "type":
                "PHONE DISTRACTION",

            "reason":
                "Possible phone usage detected.",

            "confidence":
                activity.get(
                    "confidence",
                    0
                )
        }

    # ========================================================
    # 4. DRINKING
    # ========================================================

    if (
        activity.get("label")
        == "Drinking"
        and activity.get(
            "confidence",
            0
        )
        >= 70
    ):

        return {

            "type":
                "DRINKING",

            "reason":
                "Drinking activity detected.",

            "confidence":
                activity.get(
                    "confidence",
                    0
                )
        }

    # ========================================================
    # 5. DANGEROUS DRIVING
    # ========================================================

    if (
        activity.get("label")
        == "DangerousDriving"
        and activity.get(
            "confidence",
            0
        )
        >= 70
    ):

        return {

            "type":
                "DANGEROUS DRIVING",

            "reason":
                "Dangerous-driving activity pattern detected.",

            "confidence":
                activity.get(
                    "confidence",
                    0
                )
        }

    # ========================================================
    # 6. DOWNWARD
    #
    # This is already calculated using ONLY pitch.
    #
    # So:
    #
    # RIGHT = yaw
    # LEFT  = yaw
    # DOWN  = pitch
    #
    # No mixing.
    # ========================================================

    if downward_triggered:

        return {

            "type":
                "DOWNWARD DISTRACTION",

            "reason":
                "Driver's head remained below the normal downward attention limit for multiple recent frames.",

            "confidence":
                0
        }

    # ========================================================
    # 7. LARGE HEAD MOVEMENT
    # ========================================================

    if head.get(
        "warning",
        False
    ):

        return {

            "type":
                "HEAD DISTRACTION",

            "reason":
                head.get(
                    "reason",
                    "Head remained outside the normal driving scan zone."
                ),

            "confidence":
                0
        }

    # ========================================================
    # 8. PERSISTENT GAZE
    # ========================================================

    if gaze.get(
        "warning",
        False
    ):

        return {

            "type":
                "GAZE DISTRACTION",

            "reason":
                "Eye gaze remained away from the normal forward viewing zone.",

            "confidence":
                0
        }

    # ========================================================
    # 9. SAFE
    # ========================================================

    return {

        "type":
            "SAFE",

        "reason":
            "Driver appears attentive. Head movement is within the normal driving scan zone and no significant distraction was detected.",

        "confidence":
            0
    }