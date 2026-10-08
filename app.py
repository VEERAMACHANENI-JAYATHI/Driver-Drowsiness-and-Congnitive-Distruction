from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    send_from_directory
)

import threading
import time
import os
import atexit

import tensorflow as tf

from camera import Camera

from monitor import DriverMonitor

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,

    HOST,
    PORT,
    DEBUG,

    TEMPLATE_DIR,
    CAPTURE_DIR,

    FACE_MODEL,
    NTHU_MODEL,
    ACTIVITY_MODEL
)


# ============================================================
# APP
# ============================================================

app = Flask(
    __name__,
    template_folder=str(
        TEMPLATE_DIR
    )
)


# ============================================================
# GLOBALS
# ============================================================

camera = None

monitor = None

latest_frame = None

latest_result = {

    "status":
        "STARTING",

    "event":
        "STARTING",

    "explanation":
        "Starting system...",

    "capture":
        None,

    "captures":
        []
}


lock = threading.Lock()

running = True


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    model_name
):

    print()

    print(
        f"Loading {model_name}"
    )

    try:

        if not model_path.exists():

            print(
                f"{model_name} not found:"
            )

            print(
                model_path
            )

            return None


        model = tf.keras.models.load_model(
            str(model_path),
            compile=False
        )


        print(
            f"{model_name} loaded."
        )

        print(
            "Input:",
            model.input_shape
        )

        print(
            "Output:",
            model.output_shape
        )


        return model


    except Exception as e:

        print(
            f"Error loading {model_name}:",
            e
        )

        return None


# ============================================================
# CAMERA
# ============================================================

def initialize_camera():

    global camera

    try:

        camera = Camera(

            camera_index=
                CAMERA_INDEX,

            width=
                FRAME_WIDTH,

            height=
                FRAME_HEIGHT
        )


        return True


    except Exception as e:

        print(
            "Camera error:",
            e
        )

        return False


# ============================================================
# MONITOR
# ============================================================

def initialize_monitor():

    global monitor


    if not FACE_MODEL.exists():

        print(
            "face_landmarker.task not found."
        )

        return False


    nthu_model = load_model(

        NTHU_MODEL,

        "NTHU Drowsiness Model"
    )


    activity_model = load_model(

        ACTIVITY_MODEL,

        "Driver Activity Model"
    )


    try:

        monitor = DriverMonitor(

            face_model_path=
                FACE_MODEL,

            nthu_model=
                nthu_model,

            activity_model=
                activity_model
        )


        return True


    except Exception as e:

        print(
            "Monitor initialization error:",
            e
        )

        import traceback

        traceback.print_exc()

        return False


# ============================================================
# PROCESSING LOOP
# ============================================================

def processing_loop():

    global latest_frame

    global latest_result


    while running:

        if camera is None:

            time.sleep(
                0.1
            )

            continue


        success, frame = (
            camera.read()
        )


        if (
            not success
            or frame is None
        ):

            time.sleep(
                0.05
            )

            continue


        try:

            result = (
                monitor.process(
                    frame
                )
            )


        except Exception as e:

            print(
                "Monitor error:",
                e
            )

            import traceback

            traceback.print_exc()


            result = {

                "status":
                    "ERROR",

                "event":
                    "PROCESSING ERROR",

                "explanation":
                    str(e),

                "capture":
                    None,

                "captures":
                    [],

                "frame":
                    frame
            }


        with lock:

            latest_frame = (
                result["frame"].copy()
            )

            latest_result = (
                result.copy()
            )


        time.sleep(
            0.001
        )


# ============================================================
# VIDEO
# ============================================================

def generate_frames():

    while running:

        with lock:

            if latest_frame is None:

                frame = None

            else:

                frame = (
                    latest_frame.copy()
                )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        import cv2


        success, buffer = (
            cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]
            )
        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n"

            b"Content-Length: "
            +
            str(
                len(frame_bytes)
            ).encode()
            +
            b"\r\n\r\n"

            +
            frame_bytes

            +
            b"\r\n"
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=
            "multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    with lock:

        result = (
            latest_result.copy()
        )


    result.pop(
        "frame",
        None
    )


    return jsonify(
        result
    )


# ============================================================
# CAPTURES
# ============================================================

@app.route("/captures_list")
def captures_list():

    captures = []


    if CAPTURE_DIR.exists():

        filenames = [

            filename

            for filename
            in os.listdir(
                CAPTURE_DIR
            )

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]


        # Newest first

        filenames.sort(
            reverse=True
        )


        for filename in filenames:

            filepath = (
                CAPTURE_DIR /
                filename
            )


            try:

                timestamp = (
                    filepath.stat().st_mtime
                )

                capture_time = (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(
                            timestamp
                        )
                    )
                )

            except Exception:

                capture_time = ""


            # Event name

            event_name = (
                os.path.splitext(
                    filename
                )[0]
            )


            # Remove timestamp

            parts = (
                event_name.split("_")
            )


            if len(parts) >= 3:

                event_name = "_".join(
                    parts[:-2]
                )


            event_name = (
                event_name.replace(
                    "_",
                    " "
                )
            )


            captures.append({

                "filename":
                    filename,

                "url":
                    f"/captures/{filename}",

                "event":
                    event_name,

                "time":
                    capture_time
            })


    return jsonify({

        "captures":
            captures
    })


# ============================================================
# CAPTURE IMAGE
# ============================================================

@app.route(
    "/captures/<path:filename>"
)
def captured_image(
    filename
):

    return send_from_directory(

        str(
            CAPTURE_DIR
        ),

        filename
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "application":
            "Driver Monitoring System",

        "camera":
            camera is not None,

        "monitor":
            monitor is not None,

        "running":
            running
    })


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown():

    global running


    if not running:

        return


    running = False


    try:

        if camera:

            camera.release()

    except Exception:

        pass


    try:

        if monitor:

            monitor.close()

    except Exception:

        pass


atexit.register(
    shutdown
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        " REAL-TIME DRIVER MONITORING SYSTEM"
    )

    print(
        "=========================================="
    )


    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not initialize_camera():

        shutdown()

        raise SystemExit(1)


    if not initialize_monitor():

        shutdown()

        raise SystemExit(1)


    thread = threading.Thread(

        target=
            processing_loop,

        daemon=True
    )


    thread.start()


    print()

    print(
        f"Open http://{HOST}:{PORT}"
    )

    print()


    try:

        app.run(

            host=
                HOST,

            port=
                PORT,

            debug=
                DEBUG,

            threaded=True,

            use_reloader=False
        )

    except KeyboardInterrupt:

        pass

    finally:

        shutdown()