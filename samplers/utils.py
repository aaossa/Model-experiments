"""Helper functions to be used with samplers."""
import torch


def merge_samples(data):
    """Merge samples to create a batch.

    Function defined to be used in the collate_fn argument from a
    DataLoader object, to be applied over the items yielded by a 
    BatchSampler. Concats elements from each dict to create a single
    output dict. 

    Args:
        data: List of dicts, each from CustomDataset.__getitem__,
            that contains each triple information.

    Returns:
        A tuple containing a single dict with the concatenation of
        each item in data, and the corresponding target vector, a 
        Tensor filled with ones. Each element is a Tensor. Example:

            [{"key": 2}, {"key": 3}]

        Becomes:

            ({"key": [2, 3]}, [1, 1])
    """
    elem = data[0]
    batch = dict()
    for key, value in elem.items():
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = value.numel() * len(data)
            storage = value.storage()._new_shared(numel)
            out = value.new(storage)
        else:
            out = torch.zeros(len(data), *value.size())
        batch[key] = torch.cat([b[key] for b in data],
                               out=out).view(-1, *value.size())
    target = torch.ones(len(data), 1, 1)
    return batch, target
