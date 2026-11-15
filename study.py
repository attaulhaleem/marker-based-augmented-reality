import cv2 as cv
import numpy as np

from ar_app import StageTimer
from estimators import HomographyNotFoundError
from pipeline import (apply_detector, apply_estimator, apply_filter,
                      apply_matcher, build_detector, build_estimator,
                      build_filter, build_matcher)
from utils import (center_crop_to_size, count_best_matches, draw_outline,
                   load_image, overlay_image, project_corners,
                   shrink_to_max_side, warp_image)

# Text shown in a table where a frame has no value (marker not found)
NO_VALUE_TEXT = "-"

# Times the pipeline runs on a frame to time it
TIMING_RUN_COUNT = 3


class PipelineMeasurement:
    """
    What the pipeline measured on one frame: how many keypoints and
    matches every stage left, and how long every stage took.
    """

    def __init__(
        self,
        marker_keypoint_count,
        frame_keypoint_count,
        unfiltered_match_count,
        filtered_match_count,
        inlier_count,
        homography,
        stage_seconds,
    ):
        """
        Store the keypoint counts of the marker and the frame, the match
        counts before and after filtering, the `inlier_count` (filtered
        matches that agree with the homography), the `homography` from
        marker to frame (None if the marker was not found, and then the
        inlier count is 0) and `stage_seconds`, the time each stage took
        on the frame, by stage name.
        """
        self.marker_keypoint_count = marker_keypoint_count
        self.frame_keypoint_count = frame_keypoint_count
        self.unfiltered_match_count = unfiltered_match_count
        self.filtered_match_count = filtered_match_count
        self.inlier_count = inlier_count
        self.homography = homography
        self.stage_seconds = stage_seconds


def load_study_images(study_frames, marker_max_side):
    """
    Read the images of `study_frames` (config.STUDY_FRAMES: for every
    frame name, the paths of its "frame" and its "marker"). Markers are
    shrunk to `marker_max_side`. Returns {frame_name: (marker, frame)}.
    Raises FileNotFoundError if an image can't be read.
    """
    study_images = {}
    for frame_name, entry in study_frames.items():
        marker = load_image(entry["marker"])
        marker = shrink_to_max_side(marker, marker_max_side)
        frame = load_image(entry["frame"])
        study_images[frame_name] = (marker, frame)
    return study_images


def count_close_matches(
    homography, query_keypoints, train_keypoints, matches, max_error
):
    """
    Count the `matches` (a flat list of DMatch) that agree with
    `homography`: those whose query keypoint it maps to within
    `max_error` pixels of the matched train keypoint. Returns the count.
    """
    if not matches:
        return 0

    query_points = []
    train_points = []
    for match in matches:
        query_points.append(query_keypoints[match.queryIdx].pt)
        train_points.append(train_keypoints[match.trainIdx].pt)

    query_points = np.float32(query_points).reshape(-1, 1, 2)
    train_points = np.float32(train_points).reshape(-1, 1, 2)
    projected_points = cv.perspectiveTransform(query_points, homography)

    errors = np.linalg.norm(projected_points - train_points, axis=2)
    return int(np.sum(errors <= max_error))


def measure_pipeline(marker, frame, technique_names, max_error, overrides=None):
    """
    Run the pipeline on `frame` to find `marker`, and measure it.
    `technique_names` gives the registered name of the "detector",
    "matcher", "filter" and "estimator" to use. `overrides` can replace
    their default settings: {stage_name: {setting_name: value}}.
    A filtered match counts as an inlier if the homography maps its
    marker keypoint to within `max_error` pixels of its frame keypoint.
    That is the same test for every estimator, also for one that marks
    no outliers itself.
    Returns a PipelineMeasurement. The stage times cover the work on the
    frame only: the marker features are computed before, as in the app.
    Each is the shortest of TIMING_RUN_COUNT runs.
    """
    if overrides is None:
        overrides = {}

    detector_name = technique_names["detector"]
    detector = build_detector(detector_name, **overrides.get("detector", {}))
    matcher = build_matcher(
        technique_names["matcher"], detector_name, **overrides.get("matcher", {})
    )
    match_filter = build_filter(
        technique_names["filter"], **overrides.get("filter", {})
    )
    estimator = build_estimator(
        technique_names["estimator"], **overrides.get("estimator", {})
    )

    marker_keypoints, marker_descriptors = apply_detector(detector, marker)

    # The time of one run varies, so the frame is processed several
    # times and every stage keeps its shortest time
    stage_seconds = {}
    for _ in range(TIMING_RUN_COUNT):
        timer = StageTimer()
        timer.start_frame()

        frame_keypoints, frame_descriptors = apply_detector(detector, frame)
        timer.end_stage("detector")

        knn_matches = apply_matcher(matcher, marker_descriptors, frame_descriptors)
        timer.end_stage("matcher")

        filtered_matches = apply_filter(match_filter, knn_matches)
        timer.end_stage("filter")

        try:
            homography, _ = apply_estimator(
                estimator, marker_keypoints, frame_keypoints, filtered_matches
            )
        except HomographyNotFoundError:
            homography = None
        timer.end_stage("estimator")

        for stage_name, seconds in timer.stage_seconds.items():
            shortest_seconds = stage_seconds.get(stage_name, seconds)
            stage_seconds[stage_name] = min(seconds, shortest_seconds)

    if homography is None:
        inlier_count = 0
    else:
        inlier_count = count_close_matches(
            homography, marker_keypoints, frame_keypoints, filtered_matches, max_error
        )

    return PipelineMeasurement(
        len(marker_keypoints),
        len(frame_keypoints),
        count_best_matches(knn_matches),
        len(filtered_matches),
        inlier_count,
        homography,
        stage_seconds,
    )


def measure_frames(study_images, technique_names, max_error, overrides=None):
    """
    Run measure_pipeline on every frame of `study_images`
    ({frame_name: (marker, frame)}) with the same `technique_names`,
    `max_error` and `overrides`. Returns {frame_name: measurement}.
    """
    measurements = {}
    for frame_name, (marker, frame) in study_images.items():
        measurements[frame_name] = measure_pipeline(
            marker, frame, technique_names, max_error, overrides
        )
    return measurements


def measure_techniques(
    study_images, technique_names, stage_name, stage_technique_names, max_error
):
    """
    Compare the techniques of one stage: run measure_frames once per
    name in `stage_technique_names`, used as the technique of
    `stage_name`, with the other stages as in `technique_names`.
    Returns {technique_name: {frame_name: measurement}}.
    """
    measurements_by_name = {}
    for stage_technique_name in stage_technique_names:
        compared_names = {**technique_names, stage_name: stage_technique_name}
        measurements_by_name[stage_technique_name] = measure_frames(
            study_images, compared_names, max_error
        )
    return measurements_by_name


def measure_sweep(study_images, technique_names, sweep, max_error):
    """
    Vary one setting: run measure_frames once per value of `sweep` (an
    entry of config.STUDY_SWEEPS: the "stage" and "technique" the
    "setting" belongs to and the "values" it takes), with the other
    stages as in `technique_names`.
    Returns {value_text: {frame_name: measurement}}.
    """
    stage_name = sweep["stage"]
    swept_names = {**technique_names, stage_name: sweep["technique"]}

    measurements_by_value = {}
    for value in sweep["values"]:
        overrides = {stage_name: {sweep["setting"]: value}}
        measurements_by_value[f"{value:g}"] = measure_frames(
            study_images, swept_names, max_error, overrides
        )
    return measurements_by_value


def build_summary_rows(heading, measurements_by_name):
    """
    Summarise the measurements of several techniques over all frames.
    `measurements_by_name` is {technique_name: {frame_name:
    measurement}}. Returns one table row (a dict of column title to
    text) per technique, titled `heading` in the first column: in how
    many frames the marker was found, the names of the frames where it
    was not, the mean inlier count per frame, the share of all filtered
    matches that are inliers, and the mean time of the pipeline per
    frame in milliseconds.
    """
    rows = []
    for technique_name, measurements in measurements_by_name.items():
        frame_count = len(measurements)
        missed_frame_names = []
        filtered_match_count = 0
        inlier_count = 0
        total_seconds = 0

        for frame_name, measurement in measurements.items():
            if measurement.homography is None:
                missed_frame_names.append(frame_name)
            filtered_match_count += measurement.filtered_match_count
            inlier_count += measurement.inlier_count
            total_seconds += sum(measurement.stage_seconds.values())

        if missed_frame_names:
            missed_text = ", ".join(missed_frame_names)
        else:
            missed_text = NO_VALUE_TEXT

        if filtered_match_count == 0:
            inlier_share_text = NO_VALUE_TEXT
        else:
            inlier_share_text = f"{100 * inlier_count / filtered_match_count:.0f}%"

        found_count = frame_count - len(missed_frame_names)
        rows.append(
            {
                heading: technique_name,
                "Found": f"{found_count} / {frame_count}",
                "Not Found": missed_text,
                "Inliers": f"{inlier_count / frame_count:.0f}",
                "Inlier Share": inlier_share_text,
                "Time (ms)": f"{1000 * total_seconds / frame_count:.0f}",
            }
        )
    return rows


def build_frame_rows(study_frames, measurements):
    """
    List what one configuration measured on every frame. `study_frames`
    is config.STUDY_FRAMES and `measurements` is {frame_name:
    measurement}. Returns one table row (a dict of column title to
    text) per frame: its condition, the keypoint counts of the marker
    and the frame, the filtered match count, the inlier count and
    whether the marker was found.
    """
    rows = []
    for frame_name, measurement in measurements.items():
        if measurement.homography is None:
            found_text = "No"
        else:
            found_text = "Yes"
        rows.append(
            {
                "Frame": frame_name,
                "Condition": study_frames[frame_name]["condition"],
                "Marker Keypoints": str(measurement.marker_keypoint_count),
                "Frame Keypoints": str(measurement.frame_keypoint_count),
                "Filtered": str(measurement.filtered_match_count),
                "Inliers": str(measurement.inlier_count),
                "Found": found_text,
            }
        )
    return rows


def build_time_rows(measurements):
    """
    List the mean time of each stage over `measurements`
    ({frame_name: measurement}). Returns one table row (a dict of
    column title to text) per stage, in milliseconds, and a last row
    for the total.
    """
    frame_count = len(measurements)
    stage_seconds = {}
    for measurement in measurements.values():
        for stage_name, seconds in measurement.stage_seconds.items():
            stage_seconds[stage_name] = stage_seconds.get(stage_name, 0) + seconds

    rows = []
    for stage_name, seconds in stage_seconds.items():
        rows.append(
            {
                "Stage": stage_name.capitalize(),
                "Time (ms)": f"{1000 * seconds / frame_count:.1f}",
            }
        )
    total_seconds = sum(stage_seconds.values())
    rows.append(
        {"Stage": "Total", "Time (ms)": f"{1000 * total_seconds / frame_count:.1f}"}
    )
    return rows


def draw_found_marker(marker, frame, measurement):
    """
    Draw where `measurement` (a PipelineMeasurement) found `marker` in
    `frame`. Returns a copy of the frame with the outline of the marker,
    or the frame itself if the marker was not found.
    """
    if measurement.homography is None:
        return frame
    frame_corners = project_corners(measurement.homography, marker)
    return draw_outline(frame, frame_corners, thickness=8)


def draw_augmented_frame(marker, substitute, frame, measurement, frame_alpha):
    """
    Draw the augmentation that `measurement` (a PipelineMeasurement)
    gives: `substitute` (an image of any size, cropped to the shape of
    `marker`) blended into `frame` where the marker was found.
    `frame_alpha` is how much of the frame shows through the substitute.
    Returns the augmented frame, or the frame itself if the marker was
    not found.
    """
    if measurement.homography is None:
        return frame

    marker_height, marker_width = marker.shape[:2]
    fitted_substitute = center_crop_to_size(substitute, marker_width, marker_height)

    frame_height, frame_width = frame.shape[:2]
    warped_substitute, substitute_mask = warp_image(
        fitted_substitute, measurement.homography, frame_width, frame_height
    )
    return overlay_image(frame, warped_substitute, substitute_mask, frame_alpha)
