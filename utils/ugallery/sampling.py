"""Utilities to sample triples from the dataset.

This module contains a class (HashesContainer) that calculate hashes
and notify collisions.
"""
import random
from math import ceil

from tqdm.autonotebook import tqdm


def pre_hash(triple):
    """Prepare tuple to be hashed.

    This means that each element of the tuple will be converted to
    string. The first item (profile) should be iterable, whilst only
    the second and third items (positive and negative) will be
    considered as well, leaving the rest out of the tuple.

    Args:
        triple: Tuple with the profile items (iterable), positive
            item and its negative counterpart.

    Returns:
        Same tuple but converted to string. Example:

            ([1, 2], 3, 4, 5)

        Becomes:

            (['1', '2'], '3', '4')
    """
    _sorted_t0 = tuple(sorted([str(_id) for _id in triple[0]]))
    return (_sorted_t0, str(triple[1]), str(triple[2]))


def strategy_1(n_samples, inventory, hashes_container, id2cluster, id2artist):
    """Generates samples following strategy #1 from the paper.

    Strategy #1, "Predicting missing item in purchase basket", is
    described in the paper: "Given a user 'u' who purchased items
    'P_u,k' in his k-th purchase basket with '|P_u,k | ≥ 2', if we
    hide an item 'i ∈ P_u,k' and use the rest as profile, then 'i'
    should be ranked above any item 'j < I+u' as long as 'j' does
    not belong to a visual cluster or artist that user 'u'' likes".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Each tuple includes the
        basket timestamp and user id for validating purpouses.
        Example:

            [
                ({1, 2}, 3, 4, 5, 'id')
            ]
    """
    print("Strategy 1) Predicting missing item in purchase basket")
    # Count valid users
    valid_users = 0
    for user in inventory.users.values():
        if user.strategy_1_valid_baskets(min_size=2):
            valid_users += 1

    samples_per_user = ceil(n_samples / valid_users)
    print(
        f"Valid users: {valid_users} | Samples/user: {samples_per_user}\n"
        f"Target: {n_samples} | Total samples: {valid_users * samples_per_user}"
    )
    initial_collisions = hashes_container.collisions

    samples = []
    for user in tqdm(inventory.users.values(), desc="Valid users"):
        # Pick items from baskets with more than one item
        valid_baskets = user.strategy_1_valid_baskets(min_size=2)
        if not valid_baskets:
            continue
        # Pick visual clusters and artists liked by the user
        liked_clusters = set(id2cluster[item] for item in user.profile)
        liked_artists = set(id2artist[item] for item in user.profile)

        n = samples_per_user
        while n > 0:
            ni = random.choice(inventory.items)
            if id2cluster[ni] in liked_clusters:
                continue
            if id2artist[ni] in liked_artists:
                continue
            timestamp, basket = random.choice(valid_baskets)
            pi = random.choice(tuple(basket))
            profile = {item for item in basket if item != pi}
            triple = (profile, pi, ni, timestamp, user.user_id)
            if not hashes_container.enroll(pre_hash(triple[:3])):
                continue
            samples.append(triple)
            n -= 1
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_1(samples, inventory.users, id2cluster, id2artist)
    return samples


def sanity_checks_strategy_1(samples, users, id2cluster, id2artist):
    """Performs sanity checks on strategy #1 samples.

    Validates if given samples comply with strategy #1 requirements.

    Args:
        sample: Samples generated following strategy #1.
        users: User instances from the Inventory object.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni, timestamp, uid) in tqdm(samples, desc="Check S1"):
        user = users[uid]
        gt_basket = user.baskets[timestamp]
        assert len(profile) + 1 == len(gt_basket)
        # Positive item
        assert pi in gt_basket
        # (Might not be true)
        assert pi not in profile
        # Negative item
        assert ni not in user.profile
        assert id2cluster[ni] not in user.liked_clusters
        assert id2artist[ni] not in user.liked_artists
        # Profile
        assert profile.issubset(gt_basket)
        assert profile.issubset(user.profile)


def strategy_2(n_samples, inventory, hashes_container, id2cluster, id2artist):
    """Generates samples following strategy #2 from the paper.

    Strategy #2, "Predicting next purchase basket", is described in the paper:
    "Given a user 'u' who has purchased the items 'I+u,k' up to his k-th
    purchase basket, an item 'i' in 'u'’s next purchase basket 'P_u,k+1'
    should be ranked above any item 'j < I+u' as long as 'j' does not belong
    to a visual cluster or artist that user 'u' likes.".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Each tuple includes the
        basket timestamp and user id for validating purpouses.
        Example:

            [
                ({1, 2}, 3, 4, 5, 'id')
            ]
    """
    print("Strategy 2) Predicting next purchase basket")
    # Count valid users
    valid_users = 0
    for user in inventory.users.values():
        if user.strategy_2_valid_partitions():
            valid_users += 1

    samples_per_user = ceil(n_samples / valid_users)
    print(
        f"Valid users: {valid_users} | Samples/user: {samples_per_user}\n"
        f"Target: {n_samples} | Total samples: {valid_users * samples_per_user}"
    )
    initial_collisions = hashes_container.collisions

    samples = []
    for user in tqdm(inventory.users.values(), desc="Valid users"):
        valid_partitions = user.strategy_2_valid_partitions()
        if not valid_partitions:
            continue
        # Pick visual clusters and artists liked by the user
        liked_clusters = user.liked_clusters
        liked_artists = user.liked_artists

        n = samples_per_user
        while n > 0:
            ni = random.choice(inventory.items)
            if id2cluster[ni] in liked_clusters:
                continue
            if id2artist[ni] in liked_artists:
                continue
            timestamp, profile, basket = random.choice(valid_partitions)
            pi = random.choice(tuple(basket))
            # TODO(Antonio): Not sure if possible (item purchased twice)
            if pi in profile:
                continue
            triple = (profile, pi, ni, timestamp, user.user_id)
            if not hashes_container.enroll(pre_hash(triple[:3])):
                continue
            samples.append(triple)
            n -= 1
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_2(samples, inventory.users, id2cluster, id2artist)
    return samples


def sanity_checks_strategy_2(samples, users, id2cluster, id2artist):
    """Performs sanity checks on strategy #2 samples.

    Validates if given samples comply with strategy #2 requirements.

    Args:
        sample: Samples generated following strategy #2.
        users: User instances from the Inventory object.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print("Strategy 2) Predicting next purchase basket")
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni, timestamp, uid) in tqdm(samples, desc="Check S2"):
        user = users[uid]
        gt_basket = user.baskets[timestamp]
        previous_baskets = set(item
                               for b_timestamp, basket in user.baskets.items()
                               if b_timestamp < timestamp for item in basket)
        # Positive item
        assert pi in gt_basket
        assert profile == previous_baskets
        # (Might not be true)
        assert pi not in profile
        # Negative item
        assert ni not in user.profile
        assert id2cluster[ni] not in user.liked_clusters
        assert id2artist[ni] not in user.liked_artists
        # Profile
        assert not gt_basket.issubset(profile)
        assert profile.issubset(user.profile)


def strategy_3(n_samples, inventory, hashes_container, id2cluster, id2artist, cluster2id, artist2id):
    """Generates samples following strategy #3 from the paper.

    Strategy #3, "Recommending visually similar artworks from favorite
    artists", is described in the paper: "Given a user 'u' who has purchased
    the items 'I+u', a nonpurchased item 'i < I+u' that shares artist 'ai'
    and visual cluster 'vci' with items in 'I+u' should be ranked above any
    item 'j < I+u' as long as 'j' does not belong to a visual cluster or
    artist that user 'u' likes".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.
        cluster2id: Mapping from visual cluster to item id.
        artist2id: Mapping from artist to item id.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Each tuple includes the
        user id for validating purpouses. Example:

            [
                ({1, 2}, 3, 4, 'id')
            ]
    """
    print(("Strategy 3) Recommending visually similar "
           "artworks from favorite artists"))

    # Count valid users
    valid_users = 0
    for user in inventory.users.values():
        if user.strategy_3_valid_liked(cluster2id, artist2id):
            valid_users += 1

    samples_per_user = ceil(n_samples / valid_users)
    print(
        f"Valid users: {valid_users} | Samples/user: {samples_per_user}\n"
        f"Target: {n_samples} | Total samples: {valid_users * samples_per_user}"
    )
    initial_collisions = hashes_container.collisions

    samples = []
    for user in tqdm(inventory.users.values(), desc="Valid users"):
        valid_liked = tuple(user.strategy_3_valid_liked(cluster2id, artist2id))
        if not valid_liked:
            continue
        # Pick visual clusters and artists liked by the user
        liked_clusters = user.liked_clusters
        liked_artists = user.liked_artists

        n = samples_per_user
        while n > 0:
            ni = random.choice(inventory.items)
            if id2cluster[ni] in liked_clusters:
                continue
            if id2artist[ni] in liked_artists:
                continue
            pi = random.choice(valid_liked)
            triple = (user.profile, pi, ni, user.user_id)
            if not hashes_container.enroll(pre_hash(triple[:3])):
                continue
            samples.append(triple)
            n -= 1
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_3(samples, inventory.users, id2cluster, id2artist)
    return samples


def sanity_checks_strategy_3(samples, users, id2cluster, id2artist):
    """Performs sanity checks on strategy #3 samples.

    Validates if given samples comply with strategy #3 requirements.

    Args:
        sample: Samples generated following strategy #3.
        users: User instances from the Inventory object.
        id2cluster: Mapping from item id to visual cluster.
        id2artist: Mapping from item id to artist.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print(("Strategy 3) Recommending visually similar "
           "artworks from favorite artists"))
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni, uid) in tqdm(samples, desc="Check S3"):
        user = users[uid]

        # Positive item
        assert pi not in user.profile
        assert id2cluster[pi] in user.liked_clusters
        assert id2artist[pi] in user.liked_artists
        # Negative item
        assert ni not in user.profile
        assert id2cluster[ni] not in user.liked_clusters
        assert id2artist[ni] not in user.liked_artists
        # Profile
        assert set(profile) == set(user.profile)


def strategy_4(n_samples, inventory, hashes_container):
    """Generates samples following strategy #4 from the paper.

    Strategy #4, "Recommending profile items from the same user profile", is
    described in the paper: "Given a user 'u' who has purchased the items
    'I+u', each item 'i ∈ I+u' should be ranked above any item 'j < I+u'
    (outside 'u'’s full history)".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Each tuple includes the
        user id for validating purpouses. Example:

            [
                ({1, 2}, 3, 4, 'id')
            ]
    """
    print("Strategy 4) Recommending profile items from the same user profile")

    samples_per_user = ceil(n_samples / len(inventory.users))
    print(
        f"Valid users: {len(inventory.users)} | Samples/user: {samples_per_user}\n"
        f"Target: {n_samples} | Total samples: {len(inventory.users) * samples_per_user}"
    )
    initial_collisions = hashes_container.collisions

    samples = []
    for user in tqdm(inventory.users.values(), desc="Valid users"):
        profile = user.profile

        n = samples_per_user
        while n > 0:
            ni = random.choice(inventory.items)
            if ni in profile:
                continue
            pi = random.choice(profile)
            triple = (profile, pi, ni, user.user_id)
            if not hashes_container.enroll(pre_hash(triple[:3])):
                continue
            samples.append(triple)
            n -= 1
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_4(samples, inventory.users)
    return samples


def sanity_checks_strategy_4(samples, users):
    """Performs sanity checks on strategy #4 samples.

    Validates if given samples comply with strategy #4 requirements.

    Args:
        sample: Samples generated following strategy #4.
        users: User instances from the Inventory object.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print("Strategy 4) Recommending profile items from the same user profile")
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni, uid) in tqdm(samples, desc="Check S4"):
        user = users[uid]

        # Positive item
        assert pi in user.profile
        # Negative item
        assert ni not in user.profile
        # Profile
        assert set(profile) == set(user.profile)


def strategy_5(n_samples, inventory, hashes_container):
    """Generates samples following strategy #5 from the paper.

    Strategy #5, "Recommending profile items given an artificially created
    user profile", is described in the paper: "Given an artificial user
    profile of a single item 'i', that same item 'i' should be ranked above
    any item 'j'".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Each tuple includes the
        basket timestamp and user id for validating purpouses.
        Example:

            [
                ({1, 2}, 1, 4)
            ]
    """
    print(("Strategy 5) Recommending profile items given "
           "an artificially created user profile"))

    print(f"Target: {n_samples} | Total samples: {n_samples}")
    initial_collisions = hashes_container.collisions

    samples = []
    n = n_samples
    progress_bar = tqdm(total=n_samples, desc="Valid artificial profiles")
    while n > 0:
        pi = random.choice(inventory.items)
        ni = random.choice(inventory.items)
        if ni == pi:
            continue
        triple = ([pi], pi, ni)
        if not hashes_container.enroll(pre_hash(triple)):
            continue
        progress_bar.update(1)
        samples.append(triple)
        n -= 1
    progress_bar.close()
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_5(samples)
    return samples


def sanity_checks_strategy_5(samples):
    """Performs sanity checks on strategy #5 samples.

    Validates if given samples comply with strategy #5 requirements.

    Args:
        sample: Samples generated following strategy #5.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print(("Strategy 5) Recommending profile items given "
           "an artificially created user profile"))
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni) in tqdm(samples, desc="Check S5"):
        # Positive item
        assert pi == profile[0]
        # Negative item
        assert ni != pi
        # Profile
        assert len(profile) == 1


def strategy_6(n_samples, inventory, hashes_container, id2artist, artist2id):
    """Generates samples following strategy #6 from the paper.

    Strategy #6, "Artificial profile with a single item: recommend
    visually similar items from the same artist", is described in
    the paper: "Given an artificial profile of a single item 'i′',
    an item 'i' sharing artist 'a′i'  should be ranked above any
    item 'j' not sharing artist 'a′i'".

    Args:
        n_samples: Total samples expected from this strategy.
        inventory: Inventory object with users and items.
        hashes_container: Hash manager to detect duplicates early.
        id2artist: Mapping from item id to artist.
        artist2id: Mapping from artist to item id.

    Returns:
        List of n_samples samples as tuples, following the detailed
        strategy. Won't include duplicates. Example:

            [
                ({1, 2}, 3, 4)
            ]
    """
    print(("Strategy 6) Artificial profile with a single item: "
           "recommend visually similar items from the same artist"))

    print(f"Target: {n_samples} | Total samples: {n_samples}")
    initial_collisions = hashes_container.collisions

    samples = []
    n = n_samples
    progress_bar = tqdm(total=n_samples, desc="Valid artificial profiles")
    while n > 0:
        profile_item = random.choice(inventory.items)
        profile_item_artist = id2artist[profile_item]
        pi = random.choice(artist2id[profile_item_artist])
        if pi == profile_item:
            continue
        ni = random.choice(inventory.items)
        if id2artist[ni] == profile_item_artist:
            continue
        triple = ([profile_item], pi, ni)
        if not hashes_container.enroll(pre_hash(triple)):
            continue
        progress_bar.update(1)
        samples.append(triple)
        n -= 1
    progress_bar.close()
    final_collisions = hashes_container.collisions
    print(f"Hash collisions: {final_collisions - initial_collisions}")
    sanity_checks_strategy_6(samples, id2artist)
    return samples


def sanity_checks_strategy_6(samples, id2artist):
    """Performs sanity checks on strategy #5 samples.

    Validates if given samples comply with strategy #5 requirements.

    Args:
        sample: Samples generated following strategy #5.
        id2artist: Mapping from item id to artist.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    print(("Strategy 6) Artificial profile with a single item: "
           "recommend visually similar items from the same artist"))
    print(f"Samples: {len(samples)}")
    for (profile, pi, ni) in tqdm(samples, desc="Check S6"):
        # Positive item
        assert pi != profile[0]
        assert id2artist[profile[0]] == id2artist[pi]
        # Negative item
        assert ni != pi
        assert id2artist[profile[0]] != id2artist[ni]
        # Profile
        assert len(profile) == 1
