import os
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Room, in inches, that a figure leaves above a row of images for their titles
TITLE_HEIGHT = 0.5


def load_image(image_path):
    """
    Read the image at `image_path` as a BGR array.
    Raises FileNotFoundError if it can't be read.
    """
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't read {image_path}")
    return image


def open_camera(camera_index, fallback_index=None):
    """
    Open camera `camera_index`. If it can't be opened and
    `fallback_index` is given, open that camera instead and print which
    one is used. Returns the opened cv.VideoCapture.
    Raises RuntimeError if no camera can be opened.
    """
    capture = cv.VideoCapture(camera_index)
    if capture.isOpened():
        return capture
    capture.release()

    if fallback_index is None or fallback_index == camera_index:
        raise RuntimeError(f"Couldn't open camera {camera_index}")

    capture = cv.VideoCapture(fallback_index)
    if capture.isOpened():
        print(f"Couldn't open camera {camera_index}, using camera {fallback_index}")
        return capture
    capture.release()

    raise RuntimeError(f"Couldn't open camera {camera_index} or {fallback_index}")


def grab_frame(camera_index, warmup_frames=30, fallback_index=None):
    """
    Open camera `camera_index` (or camera `fallback_index`, if given
    and the first can't be opened), let it warm up, and return one
    frame.
    Raises RuntimeError if no camera can be opened or it gives no frame.
    """
    capture = open_camera(camera_index, fallback_index)

    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            _, frame = capture.read()
    finally:
        capture.release()

    if frame is None:
        raise RuntimeError("The opened camera gave no frame")

    return frame


def save_numbered_image(image, folder, prefix="frame"):
    """
    Save `image` in `folder` (created if needed) as a PNG named `prefix`
    and the first number that is not used yet, such as frame_01.png.
    Returns the path of the saved file.
    Raises OSError if the image can't be written.
    """
    os.makedirs(folder, exist_ok=True)

    image_number = 1
    image_path = os.path.join(folder, f"{prefix}_{image_number:02d}.png")
    while os.path.exists(image_path):
        image_number += 1
        image_path = os.path.join(folder, f"{prefix}_{image_number:02d}.png")

    if not cv.imwrite(image_path, image):
        raise OSError(f"Couldn't write {image_path}")
    return image_path


def shrink_to_max_side(image, max_side):
    """
    Shrink `image` so its longest side is at most `max_side` pixels,
    keeping the aspect ratio. Returns the shrunk image, or `image`
    itself if it is already small enough (it is never enlarged).
    """
    image_height, image_width = image.shape[:2]
    scale = min(1.0, max_side / max(image_height, image_width))

    if scale == 1.0:
        return image

    # INTER_AREA averages pixels: the best choice when shrinking
    return cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def center_crop_to_size(image, target_width, target_height):
    """
    Cut the largest centred region from `image` that has the aspect
    ratio of the target, and resize it to `target_width` x
    `target_height` pixels. The image is not stretched: the parts that
    don't fit are trimmed equally on both sides. Returns the result as
    a new image.
    """
    image_height, image_width = image.shape[:2]

    # Largest scale at which a target-shaped rectangle fits in the image
    scale = min(image_width / target_width, image_height / target_height)
    crop_width = int(target_width * scale)
    crop_height = int(target_height * scale)

    left = (image_width - crop_width) // 2
    top = (image_height - crop_height) // 2

    cropped = image[top : top + crop_height, left : left + crop_width]
    return cv.resize(cropped, (target_width, target_height))


def project_corners(homography, image):
    """
    Map the four corners of `image` with `homography`. Returns their
    (x, y) positions as a float array of shape (4, 1, 2), in the order
    top-left, top-right, bottom-right, bottom-left.
    """
    image_height, image_width = image.shape[:2]
    right = image_width - 1
    bottom = image_height - 1

    corners = np.float32([[0, 0], [right, 0], [right, bottom], [0, bottom]])
    corners = corners.reshape(4, 1, 2)

    return cv.perspectiveTransform(corners, homography)


def warp_image(image, homography, output_width, output_height):
    """
    Warp `image` with `homography` into an output of `output_width` x
    `output_height` pixels. Returns (warped_image, mask), where mask is
    a grayscale image of the output size: 255 where the output is fully
    covered by `image`, 0 where it is not, and in between on the edge.
    """
    output_size = (output_width, output_height)
    warped_image = cv.warpPerspective(image, homography, output_size)

    # Warping a white image marks the output pixels that `image` covers
    white = np.full(image.shape[:2], 255, np.uint8)
    mask = cv.warpPerspective(white, homography, output_size)

    return warped_image, mask


def overlay_image(background, overlay, mask, background_alpha):
    """
    Blend `overlay` into `background` where `mask` is 255 (fully
    covered); the two images and the mask have the same size. A blended
    pixel is `background_alpha` times the background plus the rest from
    the overlay, so 0 replaces the background and 1 keeps it. Returns
    the result as a new image.
    """
    blended = cv.addWeighted(
        overlay, 1 - background_alpha, background, background_alpha, 0
    )

    is_covered = mask == 255
    result = background.copy()
    result[is_covered] = blended[is_covered]
    return result


def select_best_matches(knn_matches):
    """
    Pick the best candidate of each query descriptor from `knn_matches`
    (one list of DMatch candidates per query descriptor, best first).
    Returns them as a flat list of DMatch, closest first; query
    descriptors with no candidates are dropped.
    """
    best_matches = []
    for candidates in knn_matches:
        if candidates:
            best_matches.append(candidates[0])

    best_matches.sort(key=lambda match: match.distance)
    return best_matches


def count_best_matches(knn_matches):
    """
    Count the query descriptors in `knn_matches` (one list of DMatch
    candidates per query descriptor) that have at least one candidate.
    Returns the count, which is the length of select_best_matches
    without the sorting.
    """
    best_match_count = 0
    for candidates in knn_matches:
        if candidates:
            best_match_count += 1
    return best_match_count


def select_inlier_matches(matches, inlier_mask):
    """
    Pick the matches that `inlier_mask` marks as inliers. `matches` is
    a flat list of DMatch and `inlier_mask` has one row per match (1
    for an inlier), as returned by apply_estimator. Returns the inliers
    as a flat list of DMatch.
    """
    inlier_matches = []
    for match, is_inlier in zip(matches, inlier_mask.ravel()):
        if is_inlier:
            inlier_matches.append(match)
    return inlier_matches


def draw_keypoints(image, keypoints):
    """
    Return a copy of `image` with `keypoints` drawn on it as circles
    that indicate each keypoint's size and orientation.
    """
    return cv.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches(query_image, query_keypoints, train_image, train_keypoints, matches):
    """
    Return an image of both images side by side with a line joining
    each matched pair of keypoints. `matches` is a flat list of DMatch
    whose queryIdx and trainIdx index into `query_keypoints` and
    `train_keypoints`.
    """
    return cv.drawMatches(
        query_image,
        query_keypoints,
        train_image,
        train_keypoints,
        matches,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )


def draw_outline(image, corners, color=(0, 255, 0), thickness=3):
    """
    Draw the closed outline that joins `corners` (an array of (x, y)
    positions, in order) on a copy of `image`, in the BGR `color` and
    with lines `thickness` pixels wide. Returns the copy.
    """
    image_with_outline = image.copy()
    # cv.polylines takes a list of shapes with whole pixel positions
    outline = np.int32(np.round(corners))
    cv.polylines(image_with_outline, [outline], True, color, thickness)
    return image_with_outline


def draw_text_rows(
    rows,
    width,
    text_scale=1.0,
    row_height=30,
    heading_width=130,
    margin=14,
    font_scale=0.6,
    text_color=(235, 235, 235),
    background_color=(40, 40, 40),
):
    """
    Draw `rows`, a list of (heading, text) pairs, as a table on a new
    image `width` pixels wide: one row per pair, `row_height` pixels
    high, with the heading in a left column `heading_width` pixels wide
    and the text right of it. `margin` is the space around the table,
    and the colours are BGR. `text_scale` enlarges the text together
    with the row height, the heading column and the margin (2 doubles
    them). Returns the image.
    """
    row_height = round(row_height * text_scale)
    heading_width = round(heading_width * text_scale)
    margin = round(margin * text_scale)
    font_scale = font_scale * text_scale

    height = 2 * margin + row_height * len(rows)
    image = np.full((height, width, 3), background_color, np.uint8)

    for row_index, (heading, text) in enumerate(rows):
        baseline = margin + row_height * row_index + round(row_height * 0.7)
        for column_text, left in ((heading, margin), (text, margin + heading_width)):
            cv.putText(
                image,
                column_text,
                (left, baseline),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv.LINE_AA,
            )
    return image


class StageTimer:
    """
    Stopwatch of the pipeline: how long every stage of the last frame
    took, and how long the whole frame took.
    """

    def __init__(self):
        """
        Create the timer, with no times yet.
        """
        self.stage_seconds = {}
        self.frame_seconds = None
        self.frame_start_time = None
        self.stage_start_time = None

    def start_frame(self):
        """
        Note that a new frame starts now. From the second frame on,
        `frame_seconds` is the time since the previous frame started.
        """
        start_time = time.perf_counter()
        if self.frame_start_time is not None:
            self.frame_seconds = start_time - self.frame_start_time
        self.frame_start_time = start_time
        self.stage_start_time = start_time

    def end_stage(self, stage_name):
        """
        Note that the stage `stage_name` ends now. Its entry in
        `stage_seconds` is the time since the previous stage ended, or
        since the frame started for the first stage.
        """
        end_time = time.perf_counter()
        self.stage_seconds[stage_name] = end_time - self.stage_start_time
        self.stage_start_time = end_time


def format_table(rows):
    """
    Format `rows` (a list of dicts that all map the same column titles
    to texts) as a Markdown table: one line of titles, then one line
    per row. Returns the table as a string.
    """
    titles = list(rows[0])
    lines = [
        "| " + " | ".join(titles) + " |",
        "| " + " | ".join("---" for _ in titles) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[title]) for title in titles) + " |")
    return "\n".join(lines)


def plot_image(image, title="", size=8):
    """
    Show `image` (BGR or grayscale) in a `size` x `size` inch figure,
    with `title` above it and no axes.
    """
    plt.figure(figsize=(size, size))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_image_row(images, titles, size=20):
    """
    Show `images` (BGR or grayscale) side by side in one row, each with
    its entry of `titles` above it and no axes. The figure is `size`
    inches wide; its height follows the shape of the first image.
    """
    image_count = len(images)
    image_height, image_width = images[0].shape[:2]
    column_width = size / image_count
    row_height = column_width * image_height / image_width

    figure, axes_row = plt.subplots(
        1,
        image_count,
        figsize=(size, row_height + TITLE_HEIGHT),
        squeeze=False,
    )
    for axes, image, title in zip(axes_row[0], images, titles):
        if image.ndim == 2:
            axes.imshow(image, cmap="gray")
        else:
            axes.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axes.set_title(title)
        axes.axis("off")
    figure.tight_layout()
    plt.show()


def plot_image_grid(images, titles, images_per_row, size=20):
    """
    Show `images` (BGR or grayscale) in rows of `images_per_row`, each
    with its entry of `titles` above it and no axes. Every row is a
    figure `size` inches wide; the images should have the same shape.
    """
    for row_start in range(0, len(images), images_per_row):
        row_end = row_start + images_per_row
        plot_image_row(images[row_start:row_end], titles[row_start:row_end], size)


def plot_keypoints(image, keypoints, title="", size=8):
    """
    Show `image` with `keypoints` drawn on it as circles that indicate
    each keypoint's size and orientation.
    """
    plot_image(draw_keypoints(image, keypoints), title, size)


def plot_matches(
    query_image,
    query_keypoints,
    train_image,
    train_keypoints,
    matches,
    title="",
    size=16,
):
    """
    Show both images side by side with a line joining each matched pair
    of keypoints. `matches` is a flat list of DMatch whose queryIdx and
    trainIdx index into `query_keypoints` and `train_keypoints`.
    """
    image_with_matches = draw_matches(
        query_image, query_keypoints, train_image, train_keypoints, matches
    )
    plot_image(image_with_matches, title, size)


def plot_outline(image, corners, title="", size=8):
    """
    Show `image` with the closed outline that joins `corners` (an array
    of (x, y) positions, in order) drawn on it.
    """
    plot_image(draw_outline(image, corners), title, size)
