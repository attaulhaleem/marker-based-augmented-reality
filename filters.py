class LocalRatioFilter:
    """
    Local ratio test: judges each match against its own second best
    candidate. Keeps the best match of a query descriptor only if it is
    clearly closer than the second best.
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


class GlobalThresholdFilter:
    """
    Global threshold: judges every match against one limit for the
    whole set. Keeps the best match of a query descriptor only if its
    distance is a small fraction of the largest best-match distance.
    """

    def __init__(self, fraction):
        """
        Create the filter. A best match is kept if its distance is at
        most `fraction` times the largest best-match distance.
        """
        self.fraction = fraction

    def match_filter(self, knn_matches):
        """
        Apply the global threshold to `knn_matches` (one list of DMatch
        candidates per query descriptor). Returns the kept matches as a
        flat list of DMatch; query descriptors with no candidates are
        dropped.
        """
        best_matches = []
        for candidates in knn_matches:
            if candidates:
                best_matches.append(candidates[0])

        if not best_matches:
            return []

        max_distance = max(match.distance for match in best_matches)

        good_matches = []
        for match in best_matches:
            if match.distance <= self.fraction * max_distance:
                good_matches.append(match)
        return good_matches
