from datetime import datetime

import pandas as pd


def get_transactions_dataframes(inventory_path, purchases_path, display_stats=False):
    date_to_timestamp = lambda t: int(datetime.fromisoformat(t[:19]).timestamp())
    # Load additions DataFrame from CSV
    inventory_df = pd.read_csv(
        inventory_path,
        dtype={"id": str},  # Force artwork_id to be read as string
    )
    # Rename columns to a common format
    inventory_df = inventory_df.rename(columns={
        "id": "artwork_id",
        "upload_date": "timestamp",
    })
    # Drop unused columns
    inventory_df = inventory_df.drop(["original", "medium_id"], axis=1)
    # Transform transaction date into timestamp
    inventory_df["timestamp"] = inventory_df["timestamp"].apply(date_to_timestamp)
    # Sort trasactions by timestamp
    inventory_df = inventory_df.sort_values("timestamp")
    # Reset index according to new order
    inventory_df = inventory_df.reset_index(drop=True)
    
    # Load removals DataFrame from CSV
    purchases_df = pd.read_csv(
        purchases_path,
        dtype={"artwork_id": str},  # Force artwork_id to be read as string
    )
    # Rename columns to a common format
    purchases_df = purchases_df.rename(columns={
        "order_date": "timestamp",
    })
    # Transform transaction date into timestamp
    purchases_df["timestamp"] = purchases_df["timestamp"].apply(date_to_timestamp)
    # Form purchases baskets and transform into list
    purchases_df = purchases_df.groupby(["timestamp", "customer_id"])["artwork_id"].apply(list)
    # Move groupby indexes to columns (by reindexing)
    purchases_df = purchases_df.reset_index()
    # Sort transactions by timestamp
    purchases_df = purchases_df.sort_values("timestamp")
    # Reset index according to new order
    purchases_df = purchases_df.reset_index(drop=True)
    
    if display_stats:    
        for col in inventory_df.columns:
            print(f"Inventory - {col}: {inventory_df[col].nunique()}")

        for col in purchases_df.columns:
            if col != "artwork_id":
                print(f"Purchases - {col}: {purchases_df[col].nunique()}")
            else:
                print(f"Purchases - {col}: {purchases_df[col].map(len).mean()}")
    
    return inventory_df, purchases_df

def add_aggregation_columns(purchases_df):
    # Add column with number of baskets per customer_id
    purchases_df["n_baskets"] = purchases_df.groupby("customer_id")["timestamp"].transform("size")
    # Add column with size of purchase basket for each purchase
    purchases_df["n_items"] = purchases_df["artwork_id"].apply(len)
    # Sort transactions by timestamp
    purchases_df = purchases_df.sort_values("timestamp")
    # Reset index according to new order
    purchases_df = purchases_df.reset_index(drop=True)
    return purchases_df

def mark_evaluation_rows(purchases_df):
    def _mark_evaluation_basket(n_baskets_series):
        # Only the last purchase is used for evaluation, unless
        # its the only one (then is used for training)
        evaluation_series = pd.Series(False, index=n_baskets_series.index)
        if int(n_baskets_series.head(1)) > 1:
            evaluation_series.iloc[-1] = True
        return evaluation_series

    # Mark evaluation baskets
    purchases_df["evaluation"] = purchases_df.groupby(["customer_id"])["n_baskets"].apply(_mark_evaluation_basket)
    # Sort transactions by timestamp
    purchases_df = purchases_df.sort_values("timestamp")
    # Reset index according to new order
    purchases_df = purchases_df.reset_index(drop=True)
    return purchases_df

def get_holdout(purchases_df):
    # Create evaluation dataframe
    holdout = []
    for customer_id, group in purchases_df.groupby("customer_id"):
        # Check if there's a profile for training
        size = len(group)
        profile = group.head(size - 1)["artwork_id"].values
        profile = [item for p in profile for item in p]
        if not profile:
            continue
        # Keep last purchase for evaluation
        timestamp = group.tail(1)["timestamp"].values[0]
        predict = group.tail(1)["artwork_id"].values[0]
        holdout.append([timestamp, profile, predict, customer_id])
    # Store holdout in a pandas dataframe
    holdout = pd.DataFrame(
        holdout,
        columns=["timestamp", "profile", "predict", "user_id"],
    )
    holdout = holdout.sort_values(by=["timestamp"])
    holdout = holdout.reset_index(drop=True)
    holdout

    # Pick purchases not used for evaluation
    new_dataset = purchases_df[~purchases_df["evaluation"]]
    # Sort transactions by timestamp
    new_dataset = new_dataset.sort_values("timestamp")
    # Reset index according to new order
    new_dataset = new_dataset.reset_index(drop=True)
    
    return holdout, new_dataset

def map_ids_to_indexes(dataframe, id2index):
    # Apply mapping
    if isinstance(dataframe["artwork_id"].values[0], list):
        dataframe["artwork_id"] = dataframe["artwork_id"].apply(
            lambda artwork_ids: [id2index[_id] for _id in artwork_ids],
        )
    elif isinstance(dataframe["artwork_id"].values[0], str):
        dataframe["artwork_id"] = dataframe["artwork_id"].apply(
            lambda _id: id2index[_id],
        )
    return dataframe

def get_evaluation_dataframe(evaluation_path):
    # Load evaluation DataFrame from CSV
    evaluation_df = pd.read_csv(evaluation_path)
    string_to_list = lambda s: list(map(int, s.strip("[]").split(", ")))
    # Transform lists from str to int
    evaluation_df["shopping_cart"] = evaluation_df["shopping_cart"].apply(
        lambda s: string_to_list(s) if isinstance(s, str) else s,
    )
    evaluation_df["profile"] = evaluation_df["profile"].apply(
        lambda s: string_to_list(s) if isinstance(s, str) else s,
    )
    evaluation_df["predict"] = evaluation_df["predict"].apply(
        lambda s: string_to_list(s) if isinstance(s, str) else s,
    )
    return evaluation_df
