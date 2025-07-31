# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)
    

class SPDVectorize(nn.Module):
    """
    Extracts the lower-triangular part of SPD matrices.
    Input: (T, N, bands, C, C)
    Output: (T, N, bands, C*(C+1)//2)
    """
    def __init__(self, C: int) -> None:
        super().__init__()
        self.C = C
        idx = torch.tril_indices(C, C)
        self.register_buffer("tril_idx_row", idx[0])
        self.register_buffer("tril_idx_col", idx[1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands, C, C)
        vec = inputs[:, :, :, self.tril_idx_row, self.tril_idx_col]
        return vec  # (T, N, bands, num_tri_features)

class SPDVecNorm(nn.Module):
    def __init__(self, num_tri_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_tri_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, bands, num_tri_features)
        T, N, bands, F = x.shape
        x = x.view(-1, F)  # (T*N*bands, F)
        x = self.norm(x)
        return x.view(T, N, bands, F)


# === MLP without rotation pooling ===
class SimpleBandwiseMLP(nn.Module):
    """
    A simpler per-band MLP without rotation pooling.
    """
    def __init__(self, in_features: int, mlp_features: Sequence[int]) -> None:
        super().__init__()
        layers = []
        for out_features in mlp_features:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ])
            in_features = out_features
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands, features)
        bands = [self.mlp(inputs[:, :, i, :]) for i in range(inputs.shape[2])]
        return torch.stack(bands, dim=2)  # (T, N, bands, mlp_features[-1])


# === TDS Blocks ===
class TDSConv2dBlock(nn.Module):
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)
        T_out = x.shape[0]
        x = x + inputs[-T_out:]
        return self.layer_norm(x)


class TDSFullyConnectedBlock(nn.Module):
    def __init__(self, num_features: int, share_hand_weights: bool) -> None:
        super().__init__()
        self.share_hand_weights = share_hand_weights
        if share_hand_weights:
            half = num_features // 2
            self.fc_block = nn.Sequential(
                nn.Linear(half, half),
                nn.ReLU(),
                nn.Linear(half, half),
            )
            self.layer_norm = nn.LayerNorm(half)
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features),
            )
            self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.share_hand_weights:
            T, N, C = inputs.shape
            x = inputs.view(T, N * 2, C // 2)
            x = self.fc_block(x) + x
            x = self.layer_norm(x)
            return x.view(T, N, C)
        else:
            x = self.fc_block(inputs) + inputs
            return self.layer_norm(x)


class TDSConvEncoder(nn.Module):
    def __init__(self, num_features: int, block_channels: Sequence[int], kernel_width: int, share_hand_weights: bool) -> None:
        super().__init__()
        tds_blocks = []
        for channels in block_channels:
            assert num_features % channels == 0
            tds_blocks.append(TDSConv2dBlock(channels, num_features // channels, kernel_width))
            tds_blocks.append(TDSFullyConnectedBlock(num_features, share_hand_weights))
        self.tds_blocks = nn.Sequential(*tds_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_blocks(inputs)