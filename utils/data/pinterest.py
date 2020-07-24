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
    holdout = interactions_df[interactions_df["evaluation"]]
    # Pick interactions not used for evaluation
    new_dataset = interactions_df[~interactions_df["evaluation"]]

    return holdout, new_dataset
