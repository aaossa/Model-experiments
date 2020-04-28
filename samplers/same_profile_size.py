"""Custom batch sampler"""
import random
from collections import Counter, defaultdict

from torch.utils.data.sampler import BatchSampler


class SameProfileSizeBatchSampler(BatchSampler):
    """Custom batch sampler that yields batches of triples with the
    same profile size (CuratorNet).

    Retrieves items from the sampler and yields a batch of size
    batch_size with items of the same size.

    Attributes:
        sampler: PyTorch sampler object to retrieve triples.
        batch_size: Size of each batch.
        drop_last: Decides what to do with items that do not fill a
            batch.
        bump_rate: Probability of yielding a batch before reaching
            batch_size.
    """

    def __init__(self, sampler, batch_size, bump_rate=0.05, drop_last=False):
        """Inits a SameProfileSizeBatchSampler instance.

        Args:
            sampler: PyTorch sampler object to retrieve triples.
            batch_size: Size of each batch to be yielded.
            drop_last: Optional. Decides what to do with items that did
                not fill a batch. Defaults to False.
            bump_rate: Optional. Probability of yielding a batch before
                reaching batch_size. Defaults to 0.05 (5%).
        """
        self.sampler = sampler
        assert hasattr(self.sampler.data_source, "profile_sizes")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bump_rate = bump_rate

    def __iter__(self):
        batch_queue = defaultdict(list)
        profile_sizes = self.sampler.data_source.profile_sizes
        for idx in self.sampler:
            p_size = profile_sizes[idx]
            batch_queue[p_size].append(idx)
            if len(batch_queue[p_size]) == self.batch_size:
                batch, batch_queue[p_size] = batch_queue[p_size][:], []
                yield batch
                if random.random() < self.bump_rate and not self.drop_last:
                    possible_keys = [k for k, v in batch_queue.items() if v]
                    if possible_keys:
                        bumped_key = random.choice(possible_keys)
                        batch, batch_queue[bumped_key] = batch_queue[bumped_key][:], []
                        yield batch
        if not self.drop_last:
            for k in random.sample(list(batch_queue.keys()), len(batch_queue)):
                if batch_queue[k]:
                    yield batch_queue[k]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            counter = Counter(self.sampler.data_source.profile_sizes)
            n_samples = 0
            for k, v in counter.items():
                n_samples += (v + self.batch_size - 1) // self.batch_size
            return n_samples
