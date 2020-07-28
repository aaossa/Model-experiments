import pandas as pd


def get_interactions_dataframe(interactions_path, display_stats=False):
    # Load interactions from CSV
    interactions_df = pd.read_csv(
        interactions_path,
    )
    # Rename columns
    interactions_df = interactions_df.rename(columns={
        "board_id": "user_id",
        "im_name": "item_id",
    })

    if display_stats:
        for col in interactions_df.columns:
            print(f"Interactions - {col}: {interactions_df[col].nunique()}")

    return interactions_df


def mark_evaluation_rows(interactions_df, threshold=1):
    def _mark_evaluation_rows(n_interactions_series):
        # Only the last item is used for evaluation, unless
        # its the only one (then is used for training)
        evaluation_series = pd.Series(False, index=n_interactions_series.index)
        if len(n_interactions_series) > threshold:
            evaluation_series.iloc[-threshold:] = True
        return evaluation_series

    # Mark evaluation rows
    interactions_df["evaluation"] = interactions_df.groupby("user_id")["item_id"].apply(_mark_evaluation_rows)
    return interactions_df


def get_holdout(interactions_df):
    # Create evaluation dataframe
    holdout = []
    for user_id, group in interactions_df.groupby("user_id"):
        profile_rows = group[~group["evaluation"]]
        predict_rows = group[group["evaluation"]]
        if predict_rows.empty:
            continue
        # Extract items
        profile = profile_rows["item_id"].values.tolist()
        predict = predict_rows["item_id"].values.tolist()
        index = predict_rows.tail(1)["index"].values[0]
        # Keep last interactions for evaluation
        holdout.append([index, profile, predict, user_id])
    # Store holdout in a pandas dataframe
    holdout = pd.DataFrame(
        holdout,
        columns=["index", "profile", "predict", "user_id"],
    )
    holdout = holdout.sort_values(by=["index"])
    holdout = holdout.reset_index(drop=True)

    # Pick interactions not used for evaluation
    new_dataset = interactions_df[~interactions_df["evaluation"]]
    # Sort transactions by timestamp
    new_dataset = new_dataset.sort_values("index")
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
