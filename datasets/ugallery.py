"""UGallery Dataset (PyTorch) object

This module contains a that contains the information about the
UGallery dataset, to be accessible from PyTorch.
"""
import errno
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# TODO(Antonio): Options for training, validation and testing. Training
# and validation are stored as csv files, but testing is a json file.
# Based on torchvision.Dataset maybe


class UGalleryDataset(Dataset):
    """Represents the UGallery Dataset as a PyTorch Dataset.

    Attributes:
        profile_sizes: Size of each user profile.
        unique_profiles: Actual profile data to save space.
        profile, pi, ni: Dataset triples (in different arrays).
        transform: Transforms for each sample.
    """

    def __init__(self, csv_file, transform=None, id2index=None):
        """Inits a UGallery Dataset.

        Args:
            csv_file: Path (string) to the triplets file.
            transform: Optional. Torchvision like transforms.
            id2index: Optional. Transformation to apply on items.
        """
        # Data sources
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_file,
            )
        self.__source_file = csv_file
        # Load triples from dataframe
        triples = pd.read_csv(self.__source_file)
        # Process profile elements
        if id2index:
            # Note: Assumes id is str and index is int
            def map_id2index(element):
                if type(element) is list:
                    return [id2index[e] for e in element]
                else:
                    return id2index[str(element)]
            triples["profile"] = triples["profile"].map(lambda p: p.split())
            triples = triples.applymap(map_id2index)
            triples["profile"] = triples["profiles"].map(lambda p: " ".join(p))
        self.profile_sizes = np.fromiter(
            map(lambda p: p.count(" ") + 1, triples["profile"]),
            dtype=int, count=len(triples),
        )
        # Mapping to unique profiles
        self.unique_profiles = triples["profile"].unique()
        profile2index = {k: v for v, k in enumerate(self.unique_profiles)}
        triples["profile"] = triples["profile"].map(lambda p: profile2index[p])
        self.unique_profiles = self.unique_profiles.astype(np.string_)
        # Using numpy arrays for faster lookup
        self.profile = triples["profile"].to_numpy(copy=True)
        self.pi = triples["pi"].to_numpy(copy=True)
        self.ni = triples["ni"].to_numpy(copy=True)
        # Common setup
        self.transform = transform

    def __len__(self):
        return len(self.pi)

    def __getitem__(self, idx):
        prof = self.profile[idx]
        if isinstance(idx, int) or isinstance(idx, np.number):
            profile = np.fromstring(
                self.unique_profiles[prof], dtype=int, sep=" ",
            )
        else:
            profile = self.unique_profiles[prof]
            profile = b" ".join(profile)
            profile = np.fromstring(profile, dtype=int, sep=" ")
            profile = profile.reshape((len(idx), -1))

        return (
            profile,
            self.pi[idx],
            self.ni[idx],
        )

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
