import cv2


class Camera:

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720
    ):

        print(f"Opening camera {camera_index}...")

        self.camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            raise RuntimeError(
                f"Could not open camera {camera_index}"
            )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        print("Camera initialized successfully.")


    def read(self):

        if not self.camera.isOpened():

            return False, None

        return self.camera.read()


    def release(self):

        if self.camera is not None:

            self.camera.release()

        cv2.destroyAllWindows()