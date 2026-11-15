import cv2 as cv

import config
from estimators import HomographyNotFoundError
from pipeline import (
    apply_detector,
    apply_estimator,
    apply_filter,
    apply_matcher,
    build_detector,
    build_estimator,
    build_filter,
    build_matcher,
)
from utils import (
    center_crop_to_size,
    load_image,
    overlay_image,
    shrink_to_max_side,
    warp_image,
)

WINDOW_NAME = "Marker AR"


def application():
    """
    Run the live app: find the marker in every camera frame and show the
    frame with the marker replaced by the substitute, until q is pressed.
    Raises RuntimeError if the camera can't be opened.
    """
    capture = cv.VideoCapture(config.CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Can't open camera {config.CAMERA_INDEX}")

    marker = load_image(config.MARKER_PATH)
    marker = shrink_to_max_side(marker, config.MARKER_MAX_SIDE)

    substitute = load_image(config.SUBSTITUTE_PATH)
    substitute = center_crop_to_size(substitute, marker.shape[1], marker.shape[0])

    detector = build_detector(config.DETECTOR_NAME)
    matcher = build_matcher(config.MATCHER_NAME, config.DETECTOR_NAME)
    match_filter = build_filter(config.FILTER_NAME)
    estimator = build_estimator(config.ESTIMATOR_NAME)

    marker_keypoints, marker_descriptors = apply_detector(detector, marker)

    while True:
        frame_was_read, frame = capture.read()
        if not frame_was_read:
            break

        frame_keypoints, frame_descriptors = apply_detector(detector, frame)
        knn_matches = apply_matcher(matcher, marker_descriptors, frame_descriptors)
        good_matches = apply_filter(match_filter, knn_matches)

        try:
            homography, inlier_mask = apply_estimator(
                estimator, marker_keypoints, frame_keypoints, good_matches
            )
            frame_height, frame_width = frame.shape[:2]
            warped_substitute, substitute_mask = warp_image(
                substitute, homography, frame_width, frame_height
            )
            shown_frame = overlay_image(
                frame, warped_substitute, substitute_mask, config.FRAME_ALPHA
            )
        except HomographyNotFoundError:
            # No marker found: show the plain frame
            shown_frame = frame

        cv.imshow(WINDOW_NAME, shown_frame)

        if cv.waitKey(5) == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    application()
