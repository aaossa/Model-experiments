"""Utilities to manage UGallery dataset.

This module contains multiple functions to manage UGallery dataset
files, such as embedding and ids.
"""
from dataclasses import dataclass

import numpy as np

# TODO(Antonio): Transform 'embedding' into 'features' where necessary


@dataclass
class Embedding:
    """An instance of an embeding.

        A container for important information to use as a pretrained
        layer, with data stored in its attributes.

        Attributes:
            features: np.array with items as rows.
            id2index: mapping from each id to its index in features.
            index2id: mapping from each item index in features to its id.
    """
    features: np.array
    id2index: dict
    index2id: dict


def load_embedding(file, shape=(13297, 2048)):
    """Loads embedding from a file.

    Uses numpy.load to retrieve an embedding from a path. Also,
    creates mappings from id to index and viceversa.

    Args:
        file: Path (string) to the embedding file.
        shape: Optional. Output numpy.array shape. Default is the
            original UGallery embedding shape, (13297, 2048).

    Returns:
        A tuple containing the embedding as a numpy.array, a dict
        mapping from each id to its index, and the opposite
        mapping, in that order:

        (embedding, id2index, index2id)

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    data = np.load(file, allow_pickle=True)
    # Generate indexes and contiguous embedding
    embedding = np.zeros(shape=shape)
    id2index = dict()
    index2id = dict()
    # Fill indexes and embedding
    for i, (_id, vector_embedding) in enumerate(data):
        assert _id not in id2index, "Duplicated id"
        id2index[_id] = i
        assert i not in index2id, "Duplicated index"
        index2id[i] = _id
        assert not np.any(embedding[i]), "Embedding position not empty"
        embedding[i] = vector_embedding
    assert not np.all(embedding == 0), "Output embedding is still empty"
    assert id2index, "Output id to index mapping is still empty"
    assert index2id, "Output index to id mapping is still empty"
    return Embedding(embedding, id2index, index2id)


def load_embedding_legacy(embedding_file, ids_file):
    """Loads embedding from a file (legacy version).

    Uses numpy.load to retrieve an embedding from a given path. Also,
    loads mappings from another path.

    Args:
        embedding_file: Path (string) to the embedding file.
        ids_file: Path (string) to the ids file.

    Returns:
        An instance of the Embedding namedtuple, with the
        processed data from the files.
        A tuple containing the embedding as a numpy.array, a dict
        mapping from each id to its index, and the opposite
        mapping, in that order:

        (embedding, id2index, index2id)

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    embedding = np.load(embedding_file, allow_pickle=True)
    index2id = None
    if ids_file.endswith(".npy"):
        index2id = np.load(ids_file)
    elif ids_file.endswith("ids"):
        with open(ids_file, "r") as f:
            index2id = [int(l) for l in f.readlines()]
    assert index2id is not None, "Output index to id mapping is still None"
    index2id = {i: str(_id) for i, _id in enumerate(index2id)}
    id2index = {str(_id): i for i, _id in enumerate(index2id)}
    return Embedding(embedding, id2index, index2id)


def concatenate_embedding(embeddings, use_intersection=True):
    """Concatenate embeddings by item.

    Merges multiple embedding into a single one, creating a new
    embedding with larger vectors.

    Args:
        embeddings: Dict of embeddings (each as a Embedding
            namedtuple instance).
        use_intersection: Optional. If True, common ids will be
            used. Otherwise, every id will be available, but using
            zero as the default value for missing data.

    Returns:
        A new instance of the Embedding namedtuple, with the
        combination of the input data.

    Raises:
        AssertionError: One of the validations failed (check output).
    """
    assert all(isinstance(emb, Embedding) for emb in embeddings.values())
    assert embeddings, "Input should have at least one embedding"
    if len(embeddings) <= 1:
        emb = list(embeddings.values())[0]
        return emb
    # Use intersection of ids
    if use_intersection:
        ids = set.intersection(*(set(emb.id2index.keys())
                                 for emb in embeddings.values()))
    else:
        ids = set.union(*(set(emb.id2index.keys())
                          for emb in embeddings.values()))
    # "Freeze" set using a list instead
    ids = list(ids)
    # Prepare output
    output_shape = (
        len(ids), sum(emb.features.shape[1] for emb in embeddings.values())
    )
    output = np.zeros(shape=output_shape)
    column_start = 0
    for emb in embeddings.values():
        column_end = column_start + emb.features.shape[1]
        for i, _id in enumerate(ids):
            if _id in emb.id2index:
                index = emb.id2index[_id]
                output[i, column_start:column_end] = emb.features[index]
        column_start += emb.features.shape[1]
    assert not np.all(output == 0), "Output embedding is still empty"
    id2index = {_id: i for i, _id in enumerate(ids)}
    index2id = {i: _id for i, _id in enumerate(ids)}
    return Embedding(output, id2index, index2id)
