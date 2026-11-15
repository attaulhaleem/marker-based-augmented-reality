from utils import (draw_keypoints, draw_matches, overlay_image,
                   select_best_matches, select_inlier_matches, warp_image)


class FrameResult:
    """
    What the pipeline found in one frame, for a view to draw.
    """

    def __init__(
        self,
        marker,
        marker_keypoints,
        substitute,
        frame,
        frame_keypoints,
        knn_matches,
        filtered_matches,
        homography,
        inlier_mask,
    ):
        """
        Store the images `marker`, `substitute` (same size as the
        marker) and `frame`, the keypoints of the marker and the frame,
        the unfiltered `knn_matches` between them (one list of DMatch
        candidates per marker descriptor), the `filtered_matches` (a
        flat list of DMatch), the `homography` from marker to frame and
        its `inlier_mask` (one row per filtered match, 1 for an
        inlier). The homography and the mask are None if the marker was
        not found.
        """
        self.marker = marker
        self.marker_keypoints = marker_keypoints
        self.substitute = substitute
        self.frame = frame
        self.frame_keypoints = frame_keypoints
        self.knn_matches = knn_matches
        self.filtered_matches = filtered_matches
        self.homography = homography
        self.inlier_mask = inlier_mask


class AugmentedView:
    """
    Augmented view: the frame with the marker replaced by the
    substitute.
    """

    def __init__(self, frame_alpha):
        """
        Create the view. `frame_alpha` is how much of the frame shows
        through the substitute: 0 replaces the marker completely, 1
        leaves the frame unchanged.
        """
        self.frame_alpha = frame_alpha

    def draw(self, result):
        """
        Draw `result` (a FrameResult). Returns the frame with the
        substitute blended in at the marker's position, or the plain
        frame if the marker was not found.
        """
        if result.homography is None:
            return result.frame

        frame_height, frame_width = result.frame.shape[:2]
        warped_substitute, substitute_mask = warp_image(
            result.substitute, result.homography, frame_width, frame_height
        )
        return overlay_image(
            result.frame, warped_substitute, substitute_mask, self.frame_alpha
        )


class KeypointsView:
    """
    Keypoints view: the frame with its detected keypoints.
    """

    def draw(self, result):
        """
        Draw `result` (a FrameResult). Returns the frame with its
        keypoints as circles that indicate each keypoint's size and
        orientation.
        """
        return draw_keypoints(result.frame, result.frame_keypoints)


class UnfilteredMatchesView:
    """
    Unfiltered matches view: the marker and the frame side by side,
    with the closest matches before filtering.
    """

    def __init__(self, shown_match_count):
        """
        Create the view. It shows at most `shown_match_count` matches,
        since a line for every marker keypoint would hide the images.
        """
        self.shown_match_count = shown_match_count

    def draw(self, result):
        """
        Draw `result` (a FrameResult). Returns the marker next to the
        frame, with a line for the best match of each marker keypoint,
        closest matches first, up to `shown_match_count`.
        """
        best_matches = select_best_matches(result.knn_matches)
        return draw_matches(
            result.marker,
            result.marker_keypoints,
            result.frame,
            result.frame_keypoints,
            best_matches[: self.shown_match_count],
        )


class FilteredMatchesView:
    """
    Filtered matches view: the marker and the frame side by side, with
    the matches the filter kept.
    """

    def draw(self, result):
        """
        Draw `result` (a FrameResult). Returns the marker next to the
        frame, with a line for each filtered match.
        """
        return draw_matches(
            result.marker,
            result.marker_keypoints,
            result.frame,
            result.frame_keypoints,
            result.filtered_matches,
        )


class InlierMatchesView:
    """
    Inlier matches view: the marker and the frame side by side, with
    the filtered matches that agree with the homography.
    """

    def draw(self, result):
        """
        Draw `result` (a FrameResult). Returns the marker next to the
        frame, with a line for each inlier match, or with no lines if
        the marker was not found.
        """
        if result.homography is None:
            inlier_matches = []
        else:
            inlier_matches = select_inlier_matches(
                result.filtered_matches, result.inlier_mask
            )

        return draw_matches(
            result.marker,
            result.marker_keypoints,
            result.frame,
            result.frame_keypoints,
            inlier_matches,
        )
