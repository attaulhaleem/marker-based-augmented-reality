class RatioFilter:
    """
    Ratio test: keeps the best match of a query descriptor only if it
    is clearly closer than the second best.
    """

    def __init__(self, ratio):
        """
        Create the filter. A best match is kept if its distance is
        below `ratio` times the second best distance.
        """
        self.ratio = ratio

    def match_filter(self, knn_matches):
        """
        Apply the ratio test to `knn_matches` (one list of DMatch
        candidates per query descriptor). Returns the kept matches as a
        flat list of DMatch; query descriptors with fewer than two
        candidates are dropped.
        """
        good_matches = []
        for candidates in knn_matches:
            if len(candidates) < 2:
                continue
            best, second_best = candidates[:2]
            if best.distance < self.ratio * second_best.distance:
                good_matches.append(best)
        return good_matches
