"""Custom batch sampler"""
import torch


class DictToTensor:
    """Custom torchvision transform from dict to Tensor.

    Each value is expected to be a Numpy objects and is transformed
    into a PyTorch Tensor.
    """

    def __call__(self, sample):
        """Transforms each sample dict in a dict of Tensors.

        Args:
                sample: Dict with data.

        Returns:
                A dict with Tensors. Each value in the sample dict is
                converted into an appropiate Tensor object. Example:

                    {"key": [1, 2, 3]}

                Becomes:

                    {"key": Tensor([1, 2, 3]}
        """
        return {
            k: torch.from_numpy(v).float()
            for k, v in sample.items()
        }
