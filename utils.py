import cv2 as cv
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't read {image_path}")
    return image


def plot_image(image, title="", size=8):
    plt.figure(figsize=(size, size))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_image_with_keypoints(image, keypoints, title="", size=8):
    image_with_keypoints = cv.drawKeypoints(
        image, keypoints, None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    plot_image(image_with_keypoints, title, size)


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
    crop_width  = int(target_width  * scale)
    crop_height = int(target_height * scale)

    # Top-left corner of the centred crop
    left = (image_width  - crop_width)  // 2
    top  = (image_height - crop_height) // 2

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
