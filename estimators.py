import cv2 as cv

# A homography has 8 unknowns and each point pair gives 2 equations
MIN_POINT_PAIR_COUNT = 4


class HomographyNotFoundError(RuntimeError):
    """
    Raised when no homography can be estimated, for example because the
    marker is not visible in the frame.
    """


class HomographyEstimator:
    """
    Homography estimation: finds the 3 x 3 matrix that maps query
    points onto train points.
    """

    def __init__(self, method, reprojection_threshold):
        """
        Create the estimator. `method` is the OpenCV estimation method
        (such as cv.RANSAC). A point pair is an inlier if the homography
        maps its query point to within `reprojection_threshold` pixels
        of its train point.
        """
        self.method = method
        self.reprojection_threshold = reprojection_threshold

    def estimate(self, query_points, train_points):
        """
        Estimate the homography from `query_points` to `train_points`
        (float arrays of matching (x, y) positions). Returns
        (homography, inlier_mask), where inlier_mask has one row per
        point pair (1 for an inlier).
        Raises HomographyNotFoundError if there are fewer than four
        point pairs or no homography fits them.
        """
        point_pair_count = len(query_points)
        if point_pair_count < MIN_POINT_PAIR_COUNT:
            raise HomographyNotFoundError(
                f"A homography needs {MIN_POINT_PAIR_COUNT} point pairs, "
                f"got {point_pair_count}"
            )

        homography, inlier_mask = cv.findHomography(
            query_points,
            train_points,
            self.method,
            self.reprojection_threshold,
        )
        if homography is None:
            raise HomographyNotFoundError(
                f"No homography fits the {point_pair_count} point pairs"
            )
        return homography, inlier_mask
