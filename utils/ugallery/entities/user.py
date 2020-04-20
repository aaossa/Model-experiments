"""Entity to contain and manage a single user transactions.

This module contains a class (Inventory) to process, simulate and
iterate through the inventory states in time.
"""


class User:
    """Represents a single user transactions through time.

    Contains information from purchases and liked artsits and visual
    clusters (through the items they bought). Also, includes
    validations.

    Attributes:
        user_id: User identifier in the purchases history.
        gt_baskets: Dict (mapping) of all purchases, mapping from
            timestamp to purchased items.
        evaluation_basket: Last purchase items, used for evaluation.
            Only if user has more than one purchase.
        evaluation_timestamp: Time of last purchase, used for
            validation. Only if user has more than one purchase.
        baskets: Same as gt_baskets, but without evaluation_basket.
        liked_artists: Artists of purchased items.
        liked_clusters: Visual clusters of purchased items.
        profile: Items id from all baskets (except validation).
    """

    def __init__(self, user_id):
        """Inits a User to contain its purchases.

        Args:
            user_id: Identifier in the purchases history.
        """
        self.user_id = user_id
        self.gt_baskets = dict()
        self.baskets = dict()
        self.evaluation_basket = None
        self.evaluation_timestamp = None
        self.liked_artists = None
        self.liked_clusters = None
        self.profile = None

    def add_purchase(self, purchased_items, timestamp):
        """Register a new purchase in the user baskets.

        Adds a new purchase (one or multiple items) into the baskets
        container.

        Args:
            puchased_items: Iterable with items id.
            timestamp: Time of the purchase.

        Raises:
            AssertionError: Timestamp already has a purchase.
        """
        assert timestamp not in self.baskets
        timestamp = int(timestamp)
        # Baskets are assumed to contain each item once
        self.baskets[timestamp] = set(purchased_items)
        self.gt_baskets[timestamp] = set(purchased_items)

    def assert_baskets(self):
        """Validates user attributes.

        Must be applied after creating evaluation_basket.

        Raises:
            AssertionError: One of the validations failed.
        """
        assert isinstance(self.baskets, dict)
        assert isinstance(self.evaluation_basket, set)
        assert isinstance(self.gt_baskets, dict)
        if len(self.evaluation_basket):
            misses = 0
            for gt_timestamp, gt_basket in self.gt_baskets.items():
                if gt_timestamp in self.baskets:
                    assert gt_basket == self.baskets[gt_timestamp]
                else:
                    misses += 1
                    assert gt_basket == self.evaluation_basket
            assert misses == 1
        else:
            assert self.baskets == self.gt_baskets

    def create_profile(self):
        """Creates the user profile from available baskets.

        Must be applied after creating baskets.

        Raises:
            AssertionError: One of the validations failed.
        """
        assert self.profile is None
        self.profile = list(set(
            item
            for basket in self.baskets.values()
            for item in basket
        ))

    def create_likes(self, id2cluster, id2artist):
        """Extract liked clusters and artists from profile items.

        Args:
            id2cluster: Dict (mapping) from item id to visual cluster.
            id2artist: Dict (mapping) from item id to artist.
        """
        self.liked_clusters = set(id2cluster[item]
                                  for item in self.profile)
        self.liked_artists = set(id2artist[item]
                                 for item in self.profile)

    def save_basket_for_evaluation(self):
        """Pick last purchase to create evaluation_basket.

        Splits baskets to retrieve last purchased items and the
        timestamp of that purchase.

        Raises:
            AssertionError: One of the validations failed.
        """
        assert self.evaluation_basket is None
        assert self.evaluation_timestamp is None
        # If user has a single basket, use it for training only
        if len(self.baskets) == 1:
            self.evaluation_basket = set()
            return
        # All baskets are still available in self.gt_baskets
        last_basket_timestamp = max(self.baskets.keys())
        assert last_basket_timestamp in self.baskets
        last_basket = self.baskets.pop(last_basket_timestamp)
        assert last_basket is not None
        self.evaluation_basket = last_basket
        self.evaluation_timestamp = last_basket_timestamp

    def strategy_1_valid_baskets(self, min_size=0):
        """Strategy #1 helper method.

        Returns with more than min_size items.

        Args:
            min_size: Optional. Minimum size of baskets considered.

        Returns:
            List of tuples, where each element is a tuple of the
            form (timestamp, basket).
        """
        return [(timestamp, basket)
                for timestamp, basket in self.baskets.items()
                if len(basket) >= min_size]

    def strategy_2_valid_partitions(self):
        """Strategy #2 helper method.

        Returns baskets partitions.

        Returns:
            List of tuples, where each element is a tuple of the
            form (timestamp, profile, basket).
        """
        sorted_timestamps = sorted(self.baskets)
        valid_partitions = []
        for i in range(1, len(sorted_timestamps)):
            profile = {
                item
                for timestamp in sorted_timestamps[:i]
                for item in self.baskets[timestamp]
            }
            timestamp = sorted_timestamps[i]
            basket = self.baskets[timestamp]
            valid_partitions.append((timestamp, profile, basket))
        return valid_partitions

    def strategy_3_valid_liked(self, artwork_cluster2id, artwork_artist2id):
        """Strategy #3 helper method.

        Returns possible artworks based on user likes.

        Returns:
            List of tuples, where each element is a tuple of the
            form (timestamp, profile, basket).
        """
        liked_clusters_artworks = {
            artwork
            for cluster in self.liked_clusters
            for artwork in artwork_cluster2id[cluster]
        }
        liked_artists_artworks = {
            artwork
            for artist in self.liked_artists
            for artwork in artwork_artist2id[artist]
        }
        positive_candidates = liked_clusters_artworks & liked_artists_artworks
        valid_liked = positive_candidates - set(self.profile)
        return valid_liked
