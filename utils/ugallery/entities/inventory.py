"""Entity to simulate and manage inventory through time

This module contains a class (Inventory) to process, simulate and
iterate through the inventory states in time.
"""
from datetime import datetime

import pandas as pd

from .user import User


class Inventory:
    """Represents the inventory through time and simulate its states.

    Uses information from the given inventory data to simulate
    additions and removals (purchases) of items. Also, creates and
    contains the system users.

    Attributes:
        inventory: Pandas DataFrame of additions to inventory.
        purchases: Pandas DataFrame of purchases from inventory.
        users: List of User objects, created from purchases.
        items: Tuple of items ids.
        non_unique_items: Set of items ids that can be consumed
            multiple times.
    """

    def __init__(self, inventory_path, purchases_path, legacy=False):
        """Inits an Inventory with data from the given csv files.

        Args:
            inventory_path: Path (string) to the uploads file.
            purchases_path: Path (string) to the purchases file.
            legacy: Optional. Boolean to use legacy information.
        """
        self.users = None
        # Build dataframes to manage data
        self.inventory = pd.read_csv(inventory_path)
        self.purchases = pd.read_csv(purchases_path)
        # Process dataframes
        if not legacy:
            self.__prepare_dataframes()
        else:
            self.__prepare_dataframes_legacy()
        # Sort by timestamp and reset index
        self.inventory.sort_values(by="timestamp", inplace=True)
        self.inventory.reset_index(drop=True, inplace=True)
        self.purchases.sort_values(by="timestamp", inplace=True)
        self.purchases.reset_index(drop=True, inplace=True)

        # Check if artwork_id_hash has duplicates (inventory)
        assert not self.inventory["artwork_id"].duplicated().any()
        # Check for missing values in data (inventory)
        assert not self.inventory.isnull().values.any()
        # Check for missing values in data (purchases)
        assert not self.purchases.isnull().values.any()
        # Check if all purchases contain elements
        assert all(p for p in self.purchases["basket"])

        # Find non-unique items
        purchased_items = self.purchases["basket"].sum()
        self.non_unique_items, seen = set(), set()
        for item in purchased_items:
            if item not in seen:
                seen.add(item)
            else:
                self.non_unique_items.add(item)
        # Create tuple with items ids
        self.items = tuple(self.inventory["artwork_id"].unique())

    def __prepare_dataframes(self):
        """Process dataframes into a common format.

        Mostly renaming columns and applying changes to columns.
        Same as __prepare_dataframes_legacy.

        Returns:
            Hash value as an integer.
        """
        # Change column names
        self.inventory.rename(
            columns={
                "artist_id_hash": "artist_id",
                "artwork_id_hash": "artwork_id",
                "upload_timestamp": "timestamp",
            },
            inplace=True,
        )
        self.purchases.rename(
            columns={
                "purchase_timestamp": "timestamp",
                "purchased_artwork_ids_hash": "basket",
                "user_id_hash": "user_id",
            },
            inplace=True,
        )
        # Process purchases column (from string to list)
        self.purchases["basket"] = self.purchases["basket"].map(
            lambda p: p[1:-1].replace("'", "").split(", ")
        )

    def __prepare_dataframes_legacy(self):
        """Process dataframes into a common format.

        Mostly renaming columns and applying changes to columns.
        Same as __prepare_dataframes.

        Returns:
            Hash value as an integer.
        """
        def to_tstp(t):
            return int(datetime.fromisoformat(t[:19]).timestamp())
        # Change column names
        self.inventory.rename(
            columns={
                "artist_id_hash": "artist_id",
                "id": "artwork_id",
                "upload_date": "timestamp",
            },
            inplace=True,
        )
        self.purchases.rename(
            columns={
                "order_date": "timestamp",
            },
            inplace=True,
        )
        # Drop irrelevant columns
        self.inventory.drop(["original", "medium_id"], axis=1, inplace=True)
        # Process timestamps column (to comparable format)
        self.inventory["timestamp"] = self.inventory["timestamp"].map(to_tstp)
        self.purchases["timestamp"] = self.purchases["timestamp"].map(to_tstp)
        # Generate purchases column (from string to list)
        by_purchase = self.purchases.groupby(["timestamp", "customer_id"])
        baskets = []
        for _, frame in by_purchase:
            baskets.append(list(frame["artwork_id"]))
        self.purchases = pd.DataFrame(
            [[*b[0], baskets[i]] for i, b in enumerate(by_purchase)],
            columns=["timestamp", "user_id", "basket"]
        )

    def __forward_time(self, up_to_timestamp=None):
        """Generates transactions in the order they happened.

        Progress through the inventory and purchases DataFrame
        simultaneously, after them being sorted by time of
        transaction, to yield tuples with the relevant information.

        Args:
            up_to_timestamp: Optional. Ending timestamp. If not
                provided, both DataFrames will be fully traversed.

        Yields:
            A tuple with the action, time and DataFrame row.
            Hash value as an integer.
        """
        # Sort data by timestamp
        df_inventory = self.inventory.sort_values(
            by=["timestamp"]).reset_index()
        df_purchases = self.purchases.sort_values(
            by=["timestamp"]).reset_index()
        # Limits of iteration
        i_inventory, max_inventory = 0, len(df_inventory.index)
        i_purchases, max_purchases = 0, len(df_purchases.index)
        # First row of dataframes
        row_inventory = df_inventory.loc[i_inventory, :]
        row_purchases = df_purchases.loc[i_purchases, :]
        assert row_inventory["timestamp"] == min(
            self.inventory["timestamp"]), "Wrong first inventory action"
        assert row_purchases["timestamp"] == min(
            self.purchases["timestamp"]), "Wrong first purchase action"

        while row_inventory is not None or row_purchases is not None:
            # If next timestamp is an upload
            time_inventory = getattr(row_inventory, "timestamp", float("inf"))
            time_purchases = getattr(row_purchases, "timestamp", float("inf"))
            if time_inventory <= time_purchases:
                yield ("Add item", time_inventory, row_inventory)
                i_inventory += 1
                if i_inventory >= max_inventory:
                    row_inventory = None
                else:
                    row_inventory = df_inventory.loc[i_inventory, :]
            # If next timestamp is a purchase
            elif time_purchases < time_inventory:
                yield ("Sell items", time_purchases, row_purchases)
                i_purchases += 1
                if i_purchases >= max_purchases:
                    row_purchases = None
                else:
                    row_purchases = df_purchases.loc[i_purchases, :]
            # If limit was given
            if up_to_timestamp is not None:
                if min(time_inventory, time_purchases) > up_to_timestamp:
                    break

    def available_at_t(self, up_to_timestamp=None):
        """Returns the inventory state at a given time.

        Progress in time from an empty inventory up to a point in
        time and returns all the available items at that moment.

        Args:
            up_to_timestamp: Optional. Ending timestamp. If not
                provided, last inventory will be returned.

        Returns:
            A set with the item ids available at t.
        """
        inventory = set()
        # Forward time by timestamp
        for step, timestamp, row in self.__forward_time(up_to_timestamp):
            # Add item to inventory
            if step == "Add item":
                item = row["artwork_id"]
                if item in inventory:
                    # Item already present
                    pass
                inventory.add(item)
            # Remove item if purchased item is not unique
            elif step == "Sell items":
                for item in row["basket"]:
                    if item not in inventory:
                        # Item already sold or not present
                        if item not in self.non_unique_items:
                            # Item already sold or nor present
                            pass
                    if item not in self.non_unique_items:
                        inventory.discard(item)
        return inventory

    def build_users(self, id2cluster, id2artist):
        """Create User instances and does some validations.

        Actually, users are created in __build_users.

        Args:
            id2cluster: Mapping from item ids to visual cluster.
            id2artist: Mapping from item ids to artists.
        """
        self.__build_users(id2cluster, id2artist)
        assert isinstance(self.users, dict)
        # Check if all users were built
        users_in_df = set(self.purchases["user_id"].unique())
        users_in_dict = set(self.users.keys())
        assert users_in_df == users_in_dict
        # Check if all users are present in dict
        assert all(self.purchases["user_id"].isin(self.users.keys()))
        # Check if user profiles were created
        assert all(user.profile for user in self.users.values())
        # Check if all users have evaluation basket or a single purchase
        assert all(user.evaluation_basket is not None
                   for user in self.users.values())
        assert all(user.assert_baskets for user in self.users.values())

    def __build_users(self, id2cluster, id2artist):
        """Creates user instances.

        Progress in time through purchases to identify users and
        their transactions. Also, adds their purchases and
        timestamps.

        Args:
            id2cluster: Mapping from item ids to visual cluster.
            id2artist: Mapping from item ids to artists.
        """
        self.users = dict()
        purchases = 0
        inventory = 0
        for step, timestamp, row in self.__forward_time():
            if step != "Sell items":
                inventory += 1
                continue
            purchases += 1
            if row["user_id"] not in self.users:
                user_id_hash = row["user_id"]
                self.users[user_id_hash] = User(user_id_hash)
            user = self.users[user_id_hash]
            user.add_purchase(
                row["basket"],
                row["timestamp"],
            )
        assert purchases == len(self.purchases)
        assert inventory == len(self.inventory)
        for _, user in self.users.items():
            user.save_basket_for_evaluation()
            user.create_profile()
            user.create_likes(id2cluster, id2artist)
