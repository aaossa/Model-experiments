"""VBPR implementation in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VBPR(nn.Module):
    """VBPR model architecture from 'VBPR: Visual Bayesian 
    Personalized Ranking from Implicit Feedback'.
    """

    def __init__(self, n_users, n_items, features, dim_gamma, dim_theta):
        super().__init__()

        # Image features
        self.features = nn.Embedding.from_pretrained(features, freeze=True)

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, dim_gamma)
        self.gamma_items = nn.Embedding(n_items, dim_gamma)

        # Visual factors (theta)
        self.theta_users = nn.Embedding(n_users, dim_theta)
        self.embedding = nn.Embedding(features.size(1), dim_theta)

        # Biases (beta)
        # self.beta_users = nn.Embedding(n_users, 1)
        self.beta_items = nn.Embedding(n_items, 1)
        self.visual_bias = nn.Embedding(features.size(1), 1)

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
        ui_visual_factors = self.theta_users(ui)  # Visual factors of user u
        # Items
        pi_bias = self.beta_items(pi)  # Pos. item bias
        ni_bias = self.beta_items(ni)  # Neg. item bias
        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors
        pi_features = self.features(pi)  # Pos. item visual features
        ni_features = self.features(ni)  # Neg. item visual features

        # Precompute differences
        diff_features = pi_features - ni_features
        diff_latent_factors = pi_latent_factors - ni_latent_factors

        # x_uij
        x_uij = (
            pi_bias - ni_bias
            + (ui_latent_factors * diff_latent_factors).sum(dim=1).unsqueeze(-1)
            + (ui_visual_factors * diff_features.mm(self.embedding.weight)).sum(dim=1).unsqueeze(-1)
            + diff_features.mm(self.visual_bias.weight)
        )

        return x_uij.unsqueeze(-1)

    def recommend(self, user, items=None, grad_enabled=False):
        """Generate recommendatiosn for a given usr.

        Feed forward a user id and predict scores for each given
        item.

        Args:
            user: User index in embedding, as a Tensor.
            items: Optional. Items available for recommendation
                indexes, as a Tensor. Default to score all items.
            grad_enabled: Optional. If True, gradient will be enable.
                Defaults to False.

        Returns:
            Scores for each item for the given user.
        """
        with torch.set_grad_enabled(grad_enabled):
            # User
            u_latent_factors = self.gamma_users(user)  # Latent factors of user u
            u_visual_factors = self.theta_users(user)  # Visual factors of user u
            # Items
            i_bias = self.beta_items(items)  # Items bias
            i_latent_factors = self.gamma_items(items)  # Items visual factors
            i_features = self.features(items)  # Items visual features

            # x_ui
            x_ui = (
                i_bias
                + (u_latent_factors * i_latent_factors).sum(dim=1).unsqueeze(-1)
                + (u_visual_factors * i_features.mm(self.embedding.weight)).sum(dim=1).unsqueeze(-1)
                + i_features.mm(self.visual_bias.weight)
            )

            return x_ui

    def reset_parameters(self):
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Latent factors (gamma)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)

        # Visual factors (theta)
        nn.init.xavier_uniform_(self.theta_users.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Biases (beta)
        nn.init.xavier_uniform_(self.beta_items.weight)
        nn.init.xavier_uniform_(self.visual_bias.weight)
