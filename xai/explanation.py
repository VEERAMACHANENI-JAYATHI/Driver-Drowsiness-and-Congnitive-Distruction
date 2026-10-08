def generate_explanation(
    event,
    data
):

    event_type = event.get(
        "type",
        "SAFE"
    )

    # ========================================================
    # SAFE
    # ========================================================

    if event_type == "SAFE":

        return (
            "Driver appears attentive. "
            "Small head movements used for normal "
            "road scanning are within the safe driving zone."
        )

    # ========================================================
    # CALIBRATING
    # ========================================================

    if event_type == "CALIBRATING":

        return (
            "System is learning the driver's normal "
            "head position. Please look straight at "
            "the camera during calibration."
        )

    # ========================================================
    # DROWSY
    # ========================================================

    if event_type == "DROWSY":

        closed = data.get(
            "closed_frames",
            0
        )

        ear = data.get(
            "ear"
        )

        if closed > 0:

            if ear is not None:

                return (
                    "Possible drowsiness detected because "
                    f"the eyes remained closed for "
                    f"{closed} consecutive frames. "
                    f"Current EAR is {ear:.3f}."
                )

            return (
                "Possible drowsiness detected because "
                f"the eyes remained closed for "
                f"{closed} consecutive frames."
            )

        return (
            "The drowsiness model detected a "
            "possible reduced-alertness pattern."
        )

    # ========================================================
    # HEAD DISTRACTION
    # ========================================================

    if event_type == "HEAD DISTRACTION":

        direction = data.get(
            "head_direction",
            "UNKNOWN"
        )

        pitch = data.get(
            "pitch",
            0
        )

        yaw = data.get(
            "yaw",
            0
        )

        roll = data.get(
            "roll",
            0
        )

        dp = data.get(
            "delta_pitch",
            0
        )

        dy = data.get(
            "delta_yaw",
            0
        )

        dr = data.get(
            "delta_roll",
            0
        )

        return (
            "Head movement exceeded the normal "
            "driving scan zone. "
            f"Detected direction: "
            f"{direction.replace('_', ' ').lower()}. "
            f"Current angles: "
            f"Pitch={pitch:.1f}°, "
            f"Yaw={yaw:.1f}°, "
            f"Roll={roll:.1f}°. "
            f"Change from neutral: "
            f"Pitch={dp:+.1f}°, "
            f"Yaw={dy:+.1f}°, "
            f"Roll={dr:+.1f}°."
        )

    # ========================================================
    # DOWNWARD
    # ========================================================

    if event_type == "DOWNWARD DISTRACTION":

        count = data.get(
            "downward_count",
            0
        )

        window = data.get(
            "downward_window",
            0
        )

        pitch = data.get(
            "delta_pitch",
            0
        )

        return (
            "Driver's head remained below the "
            "normal downward attention limit. "
            f"Downward frames: {count} of "
            f"the latest {window} frames. "
            f"Pitch change from neutral: "
            f"{pitch:+.1f}°."
        )

    # ========================================================
    # GAZE
    # ========================================================

    if event_type == "GAZE DISTRACTION":

        direction = data.get(
            "gaze_direction",
            "UNKNOWN"
        )

        return (
            "Eye gaze remained directed "
            f"{direction.replace('_', ' ').lower()} "
            "for multiple consecutive frames. "
            "This represents visual attention away "
            "from the forward driving view and is "
            "not classified as drowsiness."
        )

    # ========================================================
    # PHONE
    # ========================================================

    if event_type == "PHONE DISTRACTION":

        confidence = data.get(
            "activity_confidence",
            0
        )

        return (
            "The driver activity model detected "
            "a possible phone-use pattern. "
            f"Model confidence: {confidence:.1f}%."
        )

    # ========================================================
    # DRINKING
    # ========================================================

    if event_type == "DRINKING":

        confidence = data.get(
            "activity_confidence",
            0
        )

        return (
            "The driver activity model detected "
            "a possible drinking activity pattern. "
            f"Model confidence: {confidence:.1f}%."
        )

    # ========================================================
    # DANGEROUS DRIVING
    # ========================================================

    if event_type == "DANGEROUS DRIVING":

        confidence = data.get(
            "activity_confidence",
            0
        )

        return (
            "The driver activity model detected "
            "a dangerous-driving activity pattern. "
            f"Model confidence: {confidence:.1f}%."
        )

    # ========================================================
    # YAWNING
    # ========================================================

    if event_type == "YAWNING":

        mar = data.get(
            "mar"
        )

        frames = data.get(
            "yawn_frames",
            0
        )

        if mar is not None:

            return (
                "Mouth opening remained above the "
                f"yawning threshold for "
                f"{frames} consecutive frames. "
                f"Current MAR={mar:.3f}. "
                "This indicates a possible yawn."
            )

        return (
            "Mouth remained open for multiple "
            "consecutive frames, indicating "
            "a possible yawn."
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    return event.get(
        "reason",
        "Driver monitoring event detected."
    )