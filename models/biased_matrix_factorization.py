"""Biased Matrix Factorization implementation in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasedMF(nn.Module):

    def __init__(self, n_users, n_items, dim_gamma):
        super().__init__()

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, dim_gamma)
        self.gamma_items = nn.Embedding(n_items, dim_gamma)

        # Biases (beta)
        # self.beta_users = nn.Embedding(n_users, 1)
        self.beta_items = nn.Embedding(n_items, 1)

        # Random weight initialization
        self.reset_parameters()

    def forward(self, ui, pi, ni):
        """Forward pass of the model.

        Feed forward a given input (batch). Each object is expected
        to be a Tensor.

        Args:
            ui: User index, as a Tensor.
            pi: Positive item index, as a Tensor.
            ni: Negative item index, as a Tensor.

        Returns:
            Network output (scalar) for each input.
        """
        # User
        ui_latent_factors = self.gamma_users(ui)  # Latent factors of user u
        # ui_bias = self.beta_users(ui)  # User bias
        # Items
        pi_bias = self.beta_items(pi)  # Pos. item bias
        ni_bias = self.beta_items(ni)  # Neg. item bias
        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors

        # Precompute differences
        diff_latent_factors = pi_latent_factors - ni_latent_factors

        # x_uij
        x_uij = (
            (ui_latent_factors * diff_latent_factors).sum(dim=1).unsqueeze(-1)
            + pi_bias - ni_bias
        )

        return x_uij.unsqueeze(-1)

    def recommend_all(self, user, grad_enabled=False, **kwargs):
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_latent_factors = self.gamma_users(user)  # Latent factors of user u

            # Items
            i_bias = self.beta_items.weight  # Items bias
            i_latent_factors = self.gamma_items.weight  # Items latent factors

            # x_ui
            x_ui = (
                (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(-1)
                + i_bias
            )

            return x_ui

    def recommend(self, user, items=None, grad_enabled=False):
        if items is None:
            return self.recommend_all(user, grad_enabled=grad_enabled)
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_latent_factors = self.gamma_users(user)  # Latent factors of user u

            # Items
            i_bias = self.beta_items(items)  # Items bias
            i_latent_factors = self.gamma_items(items)  # Items visual factors

            # x_ui
            x_ui = (
                (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(-1)
                + i_bias
            )

            return x_ui

    def reset_parameters(self):
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Latent factors (gamma)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)

        # Biases (beta)
        # nn.init.xavier_uniform_(self.beta_users.weight)
        nn.init.xavier_uniform_(self.beta_items.weight)

    def generate_cache(self, **kwargs):
        pass
