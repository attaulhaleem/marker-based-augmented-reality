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


def create_setting_trackbar(window_name, setting_name, control, value, on_change):
    """
    Add a trackbar for the setting `setting_name` to the window
    `window_name`. `control` is the setting's entry in a registry
    "controls" (label, minimum, maximum, step) and `value` is where the
    slider starts. A trackbar only holds whole numbers, so its position
    counts steps. When the slider moves, `on_change` is called with
    `setting_name` and the new setting value.
    """
    step = control["step"]

    def on_position_change(position):
        """
        Trackbar callback: turn the slider `position` into the setting
        value and pass it on.
        """
        on_change(setting_name, position * step)

    cv.createTrackbar(
        control["label"],
        window_name,
        round(value / step),
        round(control["maximum"] / step),
        on_position_change,
    )
    cv.setTrackbarMin(control["label"], window_name, round(control["minimum"] / step))


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

    filter_entry = config.FILTERS[config.FILTER_NAME]
    filter_settings = dict(filter_entry["settings"])

    def on_filter_setting_change(setting_name, value):
        """
        Store the new `value` of the filter setting `setting_name` and
        rebuild the match filter with it.
        """
        nonlocal match_filter
        filter_settings[setting_name] = value
        match_filter = build_filter(config.FILTER_NAME, **filter_settings)

    cv.namedWindow(WINDOW_NAME)
    for setting_name, control in filter_entry.get("controls", {}).items():
        create_setting_trackbar(
            WINDOW_NAME,
            setting_name,
            control,
            filter_settings[setting_name],
            on_filter_setting_change,
        )

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
