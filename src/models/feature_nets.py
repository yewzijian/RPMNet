"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet_util import sample_and_group_multi

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """Parameter Prediction Network with single alpha/beta as parameter.

        See: Ablation study (Table 4) in paper
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """Returns beta, gating_weights"""

        batch_size = x[0].shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha


def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net


class FeatExtractionEarlyFusion(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self, features, feature_dim, radius, num_neighbors):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(feature_dim))
        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat  # (B, N, C)


