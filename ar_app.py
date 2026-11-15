import cv2 as cv

import config

WINDOW_NAME = "Marker AR"


def application():
    capture = cv.VideoCapture(config.CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Can't open camera {config.CAMERA_INDEX}")

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break

        cv.imshow(WINDOW_NAME, frame)

        if cv.waitKey(5) == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    application()
