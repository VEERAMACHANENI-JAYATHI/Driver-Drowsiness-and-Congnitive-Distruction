from collections import deque


class ConsecutiveCounter:

    def __init__(self, required):

        self.required = required
        self.count = 0


    def update(self, condition):

        if condition:

            self.count += 1

        else:

            self.count = 0

        return (
            self.count >=
            self.required
        )


    def reset(self):

        self.count = 0


class StableDirection:

    def __init__(
        self,
        required=3
    ):

        self.required = required

        self.history = deque(
            maxlen=required
        )


    def update(
        self,
        direction
    ):

        self.history.append(
            direction
        )

        if len(self.history) < self.required:

            return "CENTER"


        if all(
            d == direction
            for d in self.history
        ):

            return direction


        return "CENTER"


    def reset(self):

        self.history.clear()


class WindowCounter:

    def __init__(
        self,
        window_size,
        required_count
    ):

        self.window_size = window_size

        self.required_count = required_count

        self.values = deque(
            maxlen=window_size
        )


    def update(self, condition):

        self.values.append(
            bool(condition)
        )

        count = sum(
            self.values
        )

        return {

            "count": count,

            "window":
                len(self.values),

            "triggered":
                count >=
                self.required_count
        }


    def reset(self):

        self.values.clear()