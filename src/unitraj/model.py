"""UniTraj model architecture.

This file is adapted from the official UniTraj implementation:
https://github.com/Yasoz/UniTraj/blob/main/utils/unitraj.py

The encoder, decoder, tokenization, RoPE attention, masking/reordering, and
published hyperparameters are preserved. Unused imports from the research
script were removed and public type annotations were added.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding used by UniTraj attention."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        )
        positions = torch.arange(max_seq_len).float()
        sinusoid_input = torch.einsum("i , j -> i j", positions, inv_freq)
        self.register_buffer("sin", sinusoid_input.sin(), persistent=False)
        self.register_buffer("cos", sinusoid_input.cos(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        return sin, cos


class FeedForward(nn.Module):
    def __init__(
        self, embedding_dim: int, hidden_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention with the official UniTraj RoPE layout."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embedding_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embedding_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embedding_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        _, n, _ = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )

        sin, cos = self.rotary_emb(n)
        q1, q2 = q[..., : self.head_dim // 2], q[..., self.head_dim // 2 :]
        k1, k2 = k[..., : self.head_dim // 2], k[..., self.head_dim // 2 :]

        q_rotated = torch.cat(
            [q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1
        )
        k_rotated = torch.cat(
            [k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1
        )

        attn_scores = (
            torch.matmul(q_rotated, k_rotated.transpose(-1, -2)) * self.scale
        )
        attn_probs = self.dropout(self.attend(attn_scores))
        out = torch.matmul(attn_probs, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embedding_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            max_seq_len=max_seq_len,
                        ),
                        FeedForward(
                            embedding_dim, feedforward_dim, dropout=dropout
                        ),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x) + x
            x = ff_layer(x) + x
        return x


def take_indices(sequence: Tensor, indices: Tensor) -> Tensor:
    """Gather sequence values using UniTraj's [T, B] index layout."""

    return torch.gather(
        sequence, 0, repeat(indices, "t b -> t b c", c=sequence.shape[-1])
    )


def random_indices(size: int) -> tuple[np.ndarray, np.ndarray]:
    forward_indices = np.arange(size)
    np.random.shuffle(forward_indices)
    backward_indices = np.argsort(forward_indices)
    return forward_indices, backward_indices


def specified_mask_indices(
    size: int, mask_indices: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    forward_indices = np.arange(size)
    mask = np.isin(forward_indices, mask_indices, invert=True)
    remaining_indices = forward_indices[mask]
    np.random.shuffle(remaining_indices)
    forward_indices = np.concatenate([remaining_indices, np.asarray(mask_indices)])
    backward_indices = np.argsort(forward_indices)
    return forward_indices, backward_indices


class PatchShuffle(nn.Module):
    def __init__(self, mask_ratio: float) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self, patches: Tensor, mask_indices: Sequence[Sequence[int]] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        t, batch_size, _ = patches.shape
        remain_t = int(t * (1 - self.mask_ratio))

        if mask_indices is not None:
            indices = [
                specified_mask_indices(t, mask_indices[i])
                for i in range(batch_size)
            ]
            lengths = {len(mask_indices[i]) for i in range(batch_size)}
            if len(lengths) != 1:
                raise ValueError(
                    "All batch members must contain the same number of masked indices"
                )
            remain_t = t - len(mask_indices[0])
        else:
            indices = [random_indices(t) for _ in range(batch_size)]

        forward_indices = torch.as_tensor(
            np.stack([i[0] for i in indices], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indices = torch.as_tensor(
            np.stack([i[1] for i in indices], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indices(patches, forward_indices)
        patches = patches[:remain_t]
        return patches, forward_indices, backward_indices


class Encoder(nn.Module):
    def __init__(
        self,
        trajectory_length: int = 200,
        patch_size: int = 1,
        embedding_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 4,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.max_seq_len = 512
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.tokenizer = nn.Conv1d(2, embedding_dim, patch_size, patch_size)
        self.transformer = Transformer(
            embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            head_dim=embedding_dim // num_heads,
            feedforward_dim=embedding_dim * 4,
            dropout=0.0,
            max_seq_len=self.max_seq_len,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        trajectory: Tensor,
        interval_embedding: Tensor,
        mask_indices: Sequence[Sequence[int]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(trajectory)
        tokens = rearrange(tokens, "b c l -> l b c")

        interval_embedding = rearrange(interval_embedding, "b l c -> l b c")
        tokens = tokens + interval_embedding
        tokens, _, backward_indices = self.shuffle(tokens, mask_indices)

        tokens = torch.cat(
            [self.cls_token.expand(-1, tokens.shape[1], -1), tokens], dim=0
        )
        tokens = rearrange(tokens, "t b c -> b t c")
        features = self.transformer(tokens)
        features = self.layer_norm(features)
        features = rearrange(features, "b t c -> t b c")
        return features, backward_indices


class Decoder(nn.Module):
    def __init__(
        self,
        trajectory_length: int = 200,
        patch_size: int = 1,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.max_seq_len = 512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.transformer = Transformer(
            embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            head_dim=embedding_dim // num_heads,
            feedforward_dim=embedding_dim * 4,
            dropout=0.0,
            max_seq_len=self.max_seq_len,
        )

        self.head = nn.Linear(embedding_dim, 2 * patch_size)
        self.token_to_traj = Rearrange(
            "h b (c p) -> b c (h p)",
            p=patch_size,
            h=trajectory_length // patch_size,
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.time_token, std=0.02)

    def forward(
        self, features: Tensor, backward_indices: Tensor, interval_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        t, batch_size = features.shape[0], features.shape[1]

        backward_indices = torch.cat(
            [
                torch.zeros(
                    1,
                    backward_indices.shape[1],
                    dtype=backward_indices.dtype,
                    device=backward_indices.device,
                ),
                backward_indices + 1,
            ],
            dim=0,
        )

        num_masked = backward_indices.shape[0] - features.shape[0]
        features = torch.cat(
            [features, self.mask_token.expand(num_masked, batch_size, -1)],
            dim=0,
        )
        features = take_indices(features, backward_indices)

        interval_embedding = torch.cat(
            [self.time_token.expand(features.shape[1], 1, -1), interval_embedding],
            dim=1,
        )
        interval_embedding = rearrange(interval_embedding, "b t c -> t b c")
        features = features + interval_embedding

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[t - 1 :] = 1
        mask = take_indices(mask, backward_indices[1:] - 1)

        trajectory = self.token_to_traj(patches)
        mask = self.token_to_traj(mask)
        return trajectory, mask


class UniTraj(nn.Module):
    """Published UniTraj encoder-decoder configuration."""

    def __init__(
        self,
        trajectory_length: int = 200,
        patch_size: int = 1,
        embedding_dim: int = 128,
        encoder_layers: int = 8,
        encoder_heads: int = 4,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.trajectory_length = trajectory_length
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            trajectory_length,
            patch_size,
            embedding_dim,
            encoder_layers,
            encoder_heads,
            mask_ratio,
        )
        self.decoder = Decoder(
            trajectory_length,
            patch_size,
            embedding_dim,
            decoder_layers,
            decoder_heads,
        )
        self.interval_embedding = nn.Linear(1, embedding_dim)

    def forward(
        self,
        trajectory: Tensor,
        intervals: Tensor | None = None,
        mask_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if intervals is not None:
            intervals = intervals.unsqueeze(-1)
            interval_embeddings = self.interval_embedding(intervals)
        else:
            intervals_pooled = torch.zeros(
                (trajectory.shape[0], self.encoder.num_tokens),
                device=trajectory.device,
            )
            interval_embeddings = self.interval_embedding(
                intervals_pooled.unsqueeze(-1)
            )

        masks = mask_indices.cpu().numpy() if mask_indices is not None else None
        features, backward_indices = self.encoder(
            trajectory, interval_embeddings, masks
        )
        predicted_trajectory, mask = self.decoder(
            features, backward_indices, interval_embeddings
        )
        return predicted_trajectory, mask


def published_unitraj(mask_ratio: float = 0.5) -> UniTraj:
    """Construct the exact architecture used by the official training script."""

    return UniTraj(
        trajectory_length=200,
        patch_size=1,
        embedding_dim=128,
        encoder_layers=8,
        encoder_heads=4,
        decoder_layers=4,
        decoder_heads=4,
        mask_ratio=mask_ratio,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
