import math


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ear(points):

    """
    points must contain six eye points:

        0 = left corner
        1 = upper-left
        2 = upper-right
        3 = right corner
        4 = lower-right
        5 = lower-left
    """

    if points is None or len(points) < 6:

        return None

    horizontal = distance(
        points[0],
        points[3]
    )

    if horizontal == 0:

        return None

    vertical_1 = distance(
        points[1],
        points[5]
    )

    vertical_2 = distance(
        points[2],
        points[4]
    )

    return (
        vertical_1 +
        vertical_2
    ) / (
        2.0 * horizontal
    )


def calculate_mar(
    upper_lip,
    lower_lip,
    left_corner,
    right_corner
):

    horizontal = distance(
        left_corner,
        right_corner
    )

    if horizontal == 0:

        return 0.0

    vertical = distance(
        upper_lip,
        lower_lip
    )

    return vertical / horizontal