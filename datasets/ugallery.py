"""UGallery Dataset (PyTorch) object

This module contains a that contains the information about the
UGallery dataset, to be accessible from PyTorch.
"""
import pandas as pd
from torch.utils.data import Dataset

# TODO(Antonio): Options for training, validation and testing. Training
# and validation are stored as csv files, but testing is a json file.
# Based on torchvision.Dataset maybe


class UGalleryDataset(Dataset):
    """Represents the UGallery Dataset as a PyTorch Dataset.

    Attributes:
        triples: Dataset triples in the form (profile, pi, ni).
        profile_sizes: Size of each profile, for faster calculation.
        embedding: Precomputed features for each item.
        transform: Transforms for each item.
    """

    def __init__(self, csv_file, embedding, transform=None):
        """Inits a UGallery Dataset.

        Args:
            csv_file: Path (string) to the triplets file.
            embedding: Numpy array of items features.
            transform: Optional. Torchvision like transforms.
        """
        # Dataframe
        self.triples = pd.read_csv(csv_file)
        self.triples["profile"] = self.triples["profile"].map(
            lambda p: p[1:-1].replace("'", "").split(", ")
        )
        # Caching profile sizes
        self.profile_sizes = tuple(self.triples["profile"].map(len))
        self.embedding = embedding
        self.transform = transform
        self.__ready = False

    def prepare(self, id2index=None):
        """Prepare dataset for training/evaluation procedure.

        Map ids into indexes (in the embedding) if necessary and
        transforms triples from a Pandas DataFrem into a Numpy
        object for faster processing.

        Args:
            id2index: Optional. Mapping from item ids to indexes in
                the features container (self.embedding).
        """
        if self.__ready:
            raise Exception("Dataset was already prepared")
        if id2index:
            self.__apply_mapping(id2index)
        self.triples = self.triples.to_numpy()
        self.__ready = True
        print("Dataset is ready")

    def __apply_mapping(self, id2index):
        """Applies mapping from item ids to indexes.

        Args:
            id2index: Mapping from item ids to indexes in the
                features container (self.embedding).
        """
        def map_id2index(element):
            if type(element) is list:
                return [id2index[e] for e in element]
            else:
                return id2index[element]
        self.triples = self.triples.applymap(map_id2index)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        profile = self.embedding[self.triples[idx, 0], :]
        pi = self.embedding[self.triples[idx, 1]]
        ni = self.embedding[self.triples[idx, 2]]

        sample = {
            "profile": profile,
            "pi": pi,
            "ni": ni,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
