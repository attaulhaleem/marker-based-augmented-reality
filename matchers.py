import cv2 as cv

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

BINARY_NORMS = (cv.NORM_HAMMING, cv.NORM_HAMMING2)

# Candidates returned per query descriptor: the best match, plus the
# second best that the ratio test compares it with
CANDIDATE_COUNT = 2

# Cross-check keeps a match only if it is mutual, so there is no second
# best candidate
CROSS_CHECK_CANDIDATE_COUNT = 1


class BruteForceMatcher:
    """
    Brute-force matching: compares every query descriptor with every
    train descriptor.
    """

    def __init__(self, norm, cross_check):
        """
        Create the matcher. Descriptors are compared with `norm`. With
        `cross_check`, a match is kept only if both descriptors are each
        other's nearest neighbour.
        """
        self.matcher = cv.BFMatcher(norm, crossCheck=cross_check)
        if cross_check:
            self.candidate_count = CROSS_CHECK_CANDIDATE_COUNT
        else:
            self.candidate_count = CANDIDATE_COUNT

    def knn_match(self, query_descriptors, train_descriptors):
        """
        Find the nearest train descriptors of each query descriptor.
        Returns one list of DMatch candidates per query descriptor, best
        first: up to two, or at most one with cross-check.
        """
        return self.matcher.knnMatch(
            query_descriptors,
            train_descriptors,
            k=self.candidate_count,
        )


class FlannMatcher:
    """
    Approximate (FLANN) matching: searches an index of the train
    descriptors, which is faster than brute force on large sets but can
    miss the true nearest neighbour.
    """

    def __init__(
        self,
        norm,
        trees,
        checks,
        table_number,
        key_size,
        multi_probe_level,
    ):
        """
        Create the matcher with the index suited to `norm`: LSH for
        binary descriptors (Hamming norms), k-d trees for float
        descriptors (any other norm). `checks` applies to both; `trees`
        only to k-d trees; `table_number`, `key_size` and
        `multi_probe_level` only to LSH.
        """
        if norm in BINARY_NORMS:
            index_params = {
                "algorithm": FLANN_INDEX_LSH,
                "table_number": table_number,
                "key_size": key_size,
                "multi_probe_level": multi_probe_level,
            }
        else:
            index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": trees}
        self.matcher = cv.FlannBasedMatcher(index_params, {"checks": checks})

    def knn_match(self, query_descriptors, train_descriptors):
        """
        Find the nearest train descriptors of each query descriptor.
        Returns one list of up to two DMatch candidates per query
        descriptor, best first.
        """
        return self.matcher.knnMatch(
            query_descriptors,
            train_descriptors,
            k=CANDIDATE_COUNT,
        )
