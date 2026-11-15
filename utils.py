import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    """
    Read the image at `image_path` as a BGR array.
    Raises FileNotFoundError if it can't be read.
    """
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't read {image_path}")
    return image


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


def plot_keypoints(image, keypoints, title="", size=8):
    """
    Show `image` with `keypoints` drawn on it as circles that indicate
    each keypoint's size and orientation.
    """
    image_with_keypoints = cv.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    plot_image(image_with_keypoints, title, size)


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
    image_with_matches = cv.drawMatches(
        query_image,
        query_keypoints,
        train_image,
        train_keypoints,
        matches,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )
    plot_image(image_with_matches, title, size)


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


def plot_outline(image, corners, title="", size=8):
    """
    Show `image` with the closed outline that joins `corners` (an array
    of (x, y) positions, in order) drawn on it.
    """
    plot_image(draw_outline(image, corners), title, size)


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


def shrink_to_max_side(image, max_side):
    """
    Shrink `image` so its longest side is at most `max_side` pixels,
    keeping the aspect ratio. Images that are already small enough
    are returned unchanged (never enlarged).
    """
    image_height, image_width = image.shape[:2]
    scale = min(1.0, max_side / max(image_height, image_width))

    if scale == 1.0:
        return image

    # INTER_AREA averages pixels: the best choice when shrinking
    return cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)


def center_crop_to_size(image, target_width, target_height):
    """
    Cut the largest centred region from `image` that has the same
    aspect ratio as the target, then resize it to (target_width, target_height).
    No stretching: the parts that don't fit are trimmed equally on both sides.
    """
    image_height, image_width = image.shape[:2]

    # Largest scale at which a target-shaped rectangle still fits inside the image
    scale = min(image_width / target_width, image_height / target_height)
    crop_width = int(target_width * scale)
    crop_height = int(target_height * scale)

    # Top-left corner of the centred crop
    left = (image_width - crop_width) // 2
    top = (image_height - crop_height) // 2

    # numpy indexing is [rows, columns] = [y, x]
    cropped = image[top : top + crop_height, left : left + crop_width]

    # cv.resize takes (width, height)
    return cv.resize(cropped, (target_width, target_height))


def grab_frame(camera_index, warmup_frames=30):
    """
    Open camera `camera_index`, let it warm up, and return one frame.
    Raises RuntimeError if the camera can't be opened or gives no frame.
    """
    capture = cv.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Couldn't open camera {camera_index}")

    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            _, frame = capture.read()
    finally:
        capture.release()

    if frame is None:
        raise RuntimeError(f"No frame from camera {camera_index}")

    return frame
