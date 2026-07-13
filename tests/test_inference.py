from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unitraj import UniTrajInference, parameter_count, published_unitraj
from unitraj.inference import PUBLISHED_PARAMETER_COUNT

# Avoid severe CPU oversubscription in constrained CI runners.
torch.set_num_threads(2)


def _trajectory(point_count: int = 36) -> str:
    points = []
    for index in range(point_count):
        longitude = -73.5673 + index * 0.00001
        latitude = 45.5017 + index * 0.000005
        timestamp = 1_720_000_000 + index
        points.append(f"{longitude} {latitude} {timestamp}")
    return "SRID=4326;LINESTRING M (" + ", ".join(points) + ")"


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> UniTrajInference:
    checkpoint = Path(tmp_path_factory.mktemp("checkpoint")) / "unitraj.pt"
    torch.manual_seed(7)
    torch.save(published_unitraj().state_dict(), checkpoint)
    return UniTrajInference(checkpoint, seed=123)


def test_published_parameter_count() -> None:
    assert parameter_count(published_unitraj()) == PUBLISHED_PARAMETER_COUNT


def test_embedding_is_128d_and_deterministic(engine: UniTrajInference) -> None:
    first = engine.infer(_trajectory(), task="embedding")
    second = engine.infer(_trajectory(), task="embedding")
    assert first.embedding is not None
    assert first.embedding.shape == (128,)
    np.testing.assert_array_equal(first.embedding, second.embedding)


def test_recovery_replaces_only_requested_points(engine: UniTrajInference) -> None:
    result = engine.infer(_trajectory(), task="recovery", mask_indices=[10, 11])
    assert result.coordinates is not None
    assert result.predicted_coordinates is not None
    assert result.coordinates.shape == (36, 3)
    assert result.predicted_coordinates.shape == (2, 3)
    assert result.mask_indices == (10, 11)
    assert result.linestringm_ewkt is not None
    assert result.linestringm_ewkt.startswith("SRID=4326;LINESTRING M")
    np.testing.assert_array_equal(
        result.coordinates[:, 2],
        np.arange(1_720_000_000, 1_720_000_036, dtype=np.float64),
    )


def test_prediction_appends_five_points(engine: UniTrajInference) -> None:
    result = engine.infer(_trajectory(), task="prediction", prediction_steps=5)
    assert result.coordinates is not None
    assert result.predicted_coordinates is not None
    assert result.coordinates.shape == (41, 3)
    assert result.predicted_coordinates.shape == (5, 3)
    assert result.mask_indices == (36, 37, 38, 39, 40)
    np.testing.assert_array_equal(
        result.predicted_coordinates[:, 2],
        np.arange(1_720_000_036, 1_720_000_041, dtype=np.float64),
    )


def test_requires_real_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UniTrajInference(tmp_path / "missing.pt")


def test_rejects_non_monotonic_m(engine: UniTrajInference) -> None:
    trajectory = _trajectory().replace(
        "-73.56695 45.501875 1720000035",
        "-73.56695 45.501875 1720000034",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        engine.infer(trajectory)
