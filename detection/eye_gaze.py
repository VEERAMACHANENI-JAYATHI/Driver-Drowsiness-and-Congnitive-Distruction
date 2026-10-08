import math


LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472
]

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477
]


def get_xy(
    landmarks,
    index,
    width,
    height
):

    p = landmarks[index]

    return (
        p.x * width,
        p.y * height
    )


def get_iris_center(
    landmarks,
    indices,
    width,
    height
):

    points = []

    for i in indices:

        if i >= len(landmarks):
            return None

        points.append(
            get_xy(
                landmarks,
                i,
                width,
                height
            )
        )

    x = sum(
        p[0] for p in points
    ) / len(points)

    y = sum(
        p[1] for p in points
    ) / len(points)

    return x, y


def get_eye_box(
    landmarks,
    indices,
    width,
    height
):

    points = [
        get_xy(
            landmarks,
            i,
            width,
            height
        )
        for i in indices
    ]

    xs = [
        p[0] for p in points
    ]

    ys = [
        p[1] for p in points
    ]

    return (
        min(xs),
        max(xs),
        min(ys),
        max(ys)
    )


def calculate_gaze(
    landmarks,
    width,
    height
):

    left_iris = get_iris_center(
        landmarks,
        LEFT_IRIS,
        width,
        height
    )

    right_iris = get_iris_center(
        landmarks,
        RIGHT_IRIS,
        width,
        height
    )

    if (
        left_iris is None
        or
        right_iris is None
    ):

        return {
            "direction": "CENTER",
            "x_ratio": 0.5,
            "y_ratio": 0.5
        }


    left_box = get_eye_box(
        landmarks,
        LEFT_EYE,
        width,
        height
    )

    right_box = get_eye_box(
        landmarks,
        RIGHT_EYE,
        width,
        height
    )


    def ratio(
        iris,
        box
    ):

        left, right, top, bottom = box

        w = right - left
        h = bottom - top

        if w <= 0 or h <= 0:

            return 0.5, 0.5

        x = (
            iris[0] - left
        ) / w

        y = (
            iris[1] - top
        ) / h

        return (
            max(0, min(1, x)),
            max(0, min(1, y))
        )


    lx, ly = ratio(
        left_iris,
        left_box
    )

    rx, ry = ratio(
        right_iris,
        right_box
    )


    x = (
        lx + rx
    ) / 2

    y = (
        ly + ry
    ) / 2


    # ========================================================
    # HORIZONTAL
    # ========================================================

    if x < 0.35:

        horizontal = "LEFT"

    elif x > 0.65:

        horizontal = "RIGHT"

    else:

        horizontal = "CENTER"


    # ========================================================
    # VERTICAL
    # ========================================================

    if y < 0.35:

        vertical = "UP"

    elif y > 0.65:

        vertical = "DOWN"

    else:

        vertical = "CENTER"


    # ========================================================
    # COMBINE
    # ========================================================

    if (
        horizontal != "CENTER"
        and
        vertical != "CENTER"
    ):

        direction = (
            vertical +
            "_" +
            horizontal
        )

    elif horizontal != "CENTER":

        direction = horizontal

    elif vertical != "CENTER":

        direction = vertical

    else:

        direction = "CENTER"


    return {

        "direction": direction,

        "x_ratio":
            round(x, 3),

        "y_ratio":
            round(y, 3)
    }


def classify_gaze(
    landmarks,
    width,
    height
):

    return calculate_gaze(
        landmarks,
        width,
        height
    )