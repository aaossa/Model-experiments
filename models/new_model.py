"""NewModel implementation in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewModel(nn.Module):

    def __init__(self, implicit_embedding, explicit_embedding):
        """
        Implicit embedding: DNN extracted features
        Explicit embedding: Human-understandable concepts
        """
        super().__init__()

        # Embeddings
        self.implicit_embedding = nn.Embedding.from_pretrained(implicit_embedding, freeze=True)
        self.explicit_embedding = nn.Embedding.from_pretrained(explicit_embedding, freeze=True)

        # Implicit section (per item)
        self.selu_implicit1 = nn.Linear(implicit_embedding.shape[1], 200)
        self.selu_implicit2 = nn.Linear(200, 200)

        # Explicit section (per item)
        self.selu_explicit1 = nn.Linear(explicit_embedding.shape[1], 200)
        self.selu_explicit2 = nn.Linear(200, 200)
        # Explicit section (pooling/merge)
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
        profile = self.explicit_embedding(profile)
        pi = self.implicit_embedding(pi)
        ni = self.implicit_embedding(ni)

        # Positive item
        pi = F.selu(self.selu_implicit1(pi))
        pi = F.selu(self.selu_implicit2(pi))

        # Negative item
        ni = F.selu(self.selu_implicit1(ni))
        ni = F.selu(self.selu_implicit2(ni))

        # User profile
        profile = F.selu(self.selu_explicit1(profile))
        profile = F.selu(self.selu_explicit2(profile))
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

    def recommend_all(self, profile, cache=None, grad_enabled=False):
        with torch.set_grad_enabled(grad_enabled):
            # Load embedding data
            profile = self.explicit_embedding(profile)

            # Items
            if cache is not None:
                items = cache[0]
            else:
                items = self.implicit_embedding.weight.unsqueeze(0)
                items = F.selu(self.selu_implicit1(items))
                items = F.selu(self.selu_implicit2(items))
                items = items.transpose(-1, -2)

            # User profile
            profile = F.selu(self.selu_explicit1(profile))
            profile = F.selu(self.selu_explicit2(profile))
            profile = torch.cat(
                (self.maxpool(profile), self.avgpool(profile)), dim=-1
            )
            profile = F.selu(self.selu_pu1(profile))
            profile = F.selu(self.selu_pu2(profile))
            profile = F.selu(self.selu_pu3(profile))

            # x_ui
            x_ui = torch.bmm(profile, items).squeeze()

            return x_ui

    def recommend(self, profile, items=None, grad_enabled=False):
        with torch.set_grad_enabled(grad_enabled):
            # Load embedding data
            profile = self.explicit_embedding(profile)

            # Items
            items = self.implicit_embedding(items)
            items = F.selu(self.selu_implicit1(items))
            items = F.selu(self.selu_implicit2(items))
            items = items.transpose(-1, -2)

            # User profile
            profile = F.selu(self.selu_explicit1(profile))
            profile = F.selu(self.selu_explicit2(profile))
            profile = torch.cat(
                (self.maxpool(profile), self.avgpool(profile)), dim=-1
            )
            profile = F.selu(self.selu_pu1(profile))
            profile = F.selu(self.selu_pu2(profile))
            profile = F.selu(self.selu_pu3(profile))

            # x_ui
            x_ui = torch.bmm(profile, items).squeeze()

            return x_ui

    def reset_parameters(self):
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Implicit section (per item)
        nn.init.xavier_uniform_(self.selu_implicit1.weight)
        nn.init.xavier_uniform_(self.selu_implicit2.weight)

        # Explicit section (per item)
        nn.init.xavier_uniform_(self.selu_explicit1.weight)
        nn.init.xavier_uniform_(self.selu_explicit2.weight)
        # Explicit section (pooling/merge)
        nn.init.xavier_uniform_(self.selu_pu1.weight)
        nn.init.xavier_uniform_(self.selu_pu2.weight)
        nn.init.xavier_uniform_(self.selu_pu3.weight)

    def generate_cache(self, grad_enabled=False):
        with torch.set_grad_enabled(grad_enabled):
            # Items
            items = self.implicit_embedding.weight.unsqueeze(0)
            items = F.selu(self.selu_implicit1(items))
            items = F.selu(self.selu_implicit2(items))
            items = items.transpose(-1, -2)
        return (items,)
