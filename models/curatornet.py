"""CuratorNet implementation in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CuratorNet(nn.Module):
    """CuratorNet model architecture from 'CuratorNet: A Neural
    Network for Visually-aware Recommendation of Art Images'.
    """

    def __init__(self, embedding, input_size=2048):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)

        # Common section
        self.selu_common1 = nn.Linear(input_size, 200)
        self.selu_common2 = nn.Linear(200, 200)

        # Profile section
        self.maxpool = nn.AdaptiveMaxPool2d((1, 200))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 200))
        self.selu_pu1 = nn.Linear(200 + 200, 300)
        self.selu_pu2 = nn.Linear(300, 300)
        self.selu_pu3 = nn.Linear(300, 200)

        # Random weight initialization
        self.reset_parameters()

    def forward(self, profile, pi, ni):
        """Forward pass of the model.

        Feed forward a given input (batch). Each object is expected
        to be a Tensor.

        Args:
            profile: User profile items embeddings, as a Tensor.
            pi: Positive item embedding, as a Tensor.
            ni: Negative item embedding, as a Tensor.

        Returns:
            Network output (scalar) for each input.
        """
        # Load embedding data
        profile = self.embedding(profile)
        pi = self.embedding(pi)
        ni = self.embedding(ni)

        # Positive item
        pi = F.selu(self.selu_common1(pi))
        pi = F.selu(self.selu_common2(pi))

        # Negative item
        ni = F.selu(self.selu_common1(ni))
        ni = F.selu(self.selu_common2(ni))

        # User profile
        profile = F.selu(self.selu_common1(profile))
        profile = F.selu(self.selu_common2(profile))
        profile = torch.cat(
            (self.maxpool(profile), self.avgpool(profile)), dim=-1
        )
        profile = F.selu(self.selu_pu1(profile))
        profile = F.selu(self.selu_pu2(profile))
        profile = F.selu(self.selu_pu3(profile))

        # x_ui > x_uj
        x_ui = torch.bmm(profile, pi.unsqueeze(-1))
        x_uj = torch.bmm(profile, ni.unsqueeze(-1))

        return x_ui - x_uj

    def recommend(self, profile, items, grad_enabled=False):
        """Generate recommendatiosn for a given profile.

        Feed forward a given profile and predict scores for each
        item.

        Args:
            profile: User profile items embeddings, as a Tensor.
            items: Items available for recommendation, as a Tensor.
            grad_enabled: Optional. If True, gradient will be enable.
                Defaults to False.

        Returns:
            Scores for each item for the given profile.
        """
        print("WARNING: Not updated according to latest versions of the model")
        with torch.set_grad_enabled(grad_enabled):
            # User profile
            profile = F.selu(self.selu_common1(profile))
            profile = F.selu(self.selu_common2(profile))
            profile = torch.cat(
                (self.maxpool(profile), self.avgpool(profile)), dim=-1
            )
            profile = F.selu(self.selu_pu1(profile))
            profile = F.selu(self.selu_pu2(profile))
            profile = F.selu(self.selu_pu3(profile))

            # Items
            items = F.selu(self.selu_common1(items))
            items = F.selu(self.selu_common2(items))

            # x_ui
            x_ui = torch.bmm(profile, items.transpose(-1, -2)).squeeze()

            return x_ui

    def reset_parameters(self):
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Common section
        nn.init.xavier_uniform_(self.selu_common1.weight)
        nn.init.xavier_uniform_(self.selu_common2.weight)
        # Profile section
        nn.init.xavier_uniform_(self.selu_pu1.weight)
        nn.init.xavier_uniform_(self.selu_pu2.weight)
        nn.init.xavier_uniform_(self.selu_pu3.weight)
