"""
Taken from: https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
"""

import numpy as np


class CyclicalFeatureEncoder:
    def __init__(self, features_to_encode):
        """
        Initialize the encoder with specified temporal features to encode.

        Args:
            features_to_encode: List of strings specifying which features to encode.
                              Supported values: ['year', 'month', 'day', 'hour', 'minute', 'second']
        """
        self.features_to_encode = features_to_encode

        # Dictionary mapping feature names to their periods
        self.feature_periods = {
            "month": 12,  # Months in a year
            "day": 31,  # Max days in a month
            "hour": 24,  # Hours in a day
            "minute": 60,  # Minutes in an hour
            "second": 60,  # Seconds in a minute
            "weekday": 7,  # Days in a week
        }

        # Dictionary mapping feature names to their column indices in input array
        self.feature_indices = {
            "year": 0,
            "month": 1,
            "day": 2,
            "hour": 3,
            "minute": 4,
            "second": 5,
        }

        # Validate input features
        for feature in features_to_encode:
            if feature not in self.feature_periods and feature != "year":
                raise ValueError(
                    f"Unsupported feature: {feature}. Supported features are: {list(self.feature_periods.keys())}"
                )

    def encode_feature(self, values, period):
        """
        Convert a temporal feature to its cyclical encoding.

        Args:
            values: numpy array of values to encode
            period: the period of the cycle (e.g., 24 for hours)

        Returns:
            Tuple of (sin, cos) arrays
        """
        sin_values = np.sin(2 * np.pi * values / period)
        cos_values = np.cos(2 * np.pi * values / period)
        return sin_values, cos_values

    def encode_sequence(self, temporal_array):
        """
        Encode a sequence of temporal values with cyclical features.

        Args:
            temporal_array: numpy array of shape (seq_len, 6) containing temporal features
                          [year, month, day, hour, minute, second]

        Returns:
            numpy array of shape (seq_len, 2*n_features) containing encoded features
            where n_features is the number of features to encode
        """
        encoded_features = []

        for feature in self.features_to_encode:
            if feature in self.feature_periods:
                # Get the values for this feature
                feature_idx = self.feature_indices[feature]
                values = temporal_array[:, feature_idx]

                # Encode the feature
                sin_values, cos_values = self.encode_feature(
                    values, self.feature_periods[feature]
                )

                # Add encoded values to results
                encoded_features.extend([sin_values, cos_values])

        # Stack all encoded features
        if encoded_features:
            return np.stack(encoded_features, axis=1)
        else:
            return np.array([])


# class CyclicalFeatureEncoder:
#     def __init__(self):
#         self.hour_in_day = 24
#         self.day_in_week = 7

#     def encode_hour(self, hour):
#         """Convert hour to cyclical encoding"""
#         hour_sin = np.sin(2 * np.pi * hour / self.hour_in_day)
#         hour_cos = np.cos(2 * np.pi * hour / self.hour_in_day)
#         return hour_sin, hour_cos

#     def encode_day(self, day):
#         """Convert day of week to cyclical encoding"""
#         day_sin = np.sin(2 * np.pi * day / self.day_in_week)
#         day_cos = np.cos(2 * np.pi * day / self.day_in_week)
#         return day_sin, day_cos

#     def encode_sequence(self, timestamps):
#         """Encode a sequence of timestamps with cyclical features"""
#         hours = timestamps.hour.values
#         days = timestamps.dayofweek.values

#         hour_sin, hour_cos = self.encode_hour(hours)
#         day_sin, day_cos = self.encode_day(days)

#         return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] if w/o cls token and [1+grid_size*grid_size, embed_dim] if w/ cls token.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def test_get_2d_sincos_pos_embed():
    embed_dim = 4
    grid_size = 10
    cls_token = False

    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)

    print(pos_embed.shape)
    # Check that the output is a numpy array
    assert isinstance(pos_embed, np.ndarray)

    # Check that the shape of the output is correct
    expected_shape = (grid_size * grid_size, embed_dim)
    assert pos_embed.shape == expected_shape

    # Check that the output contains the expected values
    # This will depend on the implementation of get_2d_sincos_pos_embed_from_grid
    # For this test, we'll just check that the array contains no NaN or inf values
    assert not np.isnan(pos_embed).any()
    assert not np.isinf(pos_embed).any()


if __name__ == "__main__":
    test_get_2d_sincos_pos_embed()
