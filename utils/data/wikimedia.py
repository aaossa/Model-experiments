import pandas as pd


def get_interactions_dataframe(interactions_path, display_stats=False):
    # Load interactions from CSV
    interactions_df = pd.read_csv(
        interactions_path,
        dtype={"item_id": str},
    )
    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values("timestamp")
    # Reset index according to new order
    interactions_df = interactions_df.reset_index(drop=True)

    if display_stats:
        for col in interactions_df.columns:
            print(f"Interactions - {col}: {interactions_df[col].nunique()}")

    return interactions_df


def mark_evaluation_rows(interactions_df, threshold=1):
    def _mark_evaluation_rows(n_interactions_series):
        # Only the last item is used for evaluation, unless
        # its the only one (then is used for training)
        evaluation_series = pd.Series(False, index=n_interactions_series.index)
        if len(n_interactions_series) > 1:
            evaluation_series.iloc[-1] = True
        return evaluation_series

    # Mark evaluation rows
    interactions_df["evaluation"] = interactions_df.groupby(
        "user_id")["score"].apply(_mark_evaluation_rows)
    # Sort transactions by timestamp
    interactions_df = interactions_df.sort_values("timestamp")
    # Reset index according to new order
    interactions_df = interactions_df.reset_index(drop=True)
    return interactions_df


def get_holdout(interactions_df):
    # Create evaluation dataframe
    holdout = []
    for user_id, group in interactions_df.groupby("user_id"):
        # Check if there's a profile for training
        size = len(group)
        profile = group.head(size - 1)["item_id"].values
        profile = profile.flatten().tolist()
        if not profile:
            continue
        # Keep last purchase for evaluation
        timestamp = group.tail(1)["timestamp"].values[0]
        predict = group.tail(1)["item_id"].values[0]
        holdout.append([timestamp, profile, predict, user_id])
    # Store holdout in a pandas dataframe
    holdout = pd.DataFrame(
        holdout,
        columns=["timestamp", "profile", "predict", "user_id"],
    )
    holdout = holdout.sort_values(by=["timestamp"])
    holdout = holdout.reset_index(drop=True)
    holdout

    # Pick purchases not used for evaluation
    new_dataset = interactions_df[~interactions_df["evaluation"]]
    # Sort transactions by timestamp
    new_dataset = new_dataset.sort_values("timestamp")
    # Reset index according to new order
    new_dataset = new_dataset.reset_index(drop=True)

    return holdout, new_dataset


def get_evaluation_dataframe(evaluation_path):
    # Load evaluation DataFrame from CSV
    evaluation_df = pd.read_csv(evaluation_path)
    string_to_list = lambda s: list(map(int, s.strip("[]").split(", ")))
    # Transform lists from str to int
    evaluation_df["profile"] = evaluation_df["profile"].apply(
        lambda s: string_to_list(s) if isinstance(s, str) else s,
    )
    evaluation_df["predict"] = evaluation_df["predict"].apply(
        lambda s: string_to_list(s) if isinstance(s, str) else s,
    )
    return evaluation_df
