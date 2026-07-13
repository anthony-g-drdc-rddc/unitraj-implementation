"""Inference-only interface from PostGIS LINESTRING M to UniTraj."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .model import UniTraj, parameter_count, published_unitraj
from .postgis import (
    ParsedLineStringM,
    PostGISLineStringM,
    parse_linestringm,
    to_ewkt,
)

InferenceTask: TypeAlias = Literal["embedding", "recovery", "prediction"]
TimestampUnit: TypeAlias = Literal["s", "ms", "us", "ns"]
OverflowStrategy: TypeAlias = Literal["error", "uniform"]

TRAJECTORY_LENGTH = 200
MIN_TRAJECTORY_POINTS = 36
EMBEDDING_DIMENSION = 128
PUBLISHED_PARAMETER_COUNT = 2_376_194
COORDINATE_MEAN = np.asarray(
    [5.3311563533497974e-05, -7.49477039789781e-05], dtype=np.float32
)
COORDINATE_STD = np.asarray(
    [0.049923088401556015, 0.040688566863536835], dtype=np.float32
)
_SECONDS_PER_TIMESTAMP_UNIT: dict[TimestampUnit, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Result returned by :meth:`UniTrajInference.infer`."""

    task: InferenceTask
    embedding: NDArray[np.float32] | None = None
    coordinates: NDArray[np.float64] | None = None
    predicted_coordinates: NDArray[np.float64] | None = None
    linestringm_ewkt: str | None = None
    mask_indices: tuple[int, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedTrajectory:
    parsed: ParsedLineStringM
    spatial: Tensor
    intervals: Tensor
    original_xy: NDArray[np.float64]
    point_count: int
    selected_source_indices: tuple[int, ...]


class UniTrajInference:
    """Reusable inference engine for the published UniTraj architecture."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        timestamp_unit: TimestampUnit = "s",
        expected_srid: int | None = 4326,
        seed: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"UniTraj checkpoint does not exist: {self.checkpoint_path}"
            )
        if timestamp_unit not in _SECONDS_PER_TIMESTAMP_UNIT:
            raise ValueError(f"Unsupported timestamp unit: {timestamp_unit}")

        self.device = torch.device(device)
        self.timestamp_unit = timestamp_unit
        self.expected_srid = expected_srid
        self.seed = int(seed)
        self.model = published_unitraj(mask_ratio=0.5)
        if parameter_count(self.model) != PUBLISHED_PARAMETER_COUNT:
            raise RuntimeError(
                "UniTraj architecture parameter count does not match the published model"
            )
        self._load_checkpoint(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)

    def infer(
        self,
        trajectory: PostGISLineStringM,
        *,
        task: InferenceTask = "embedding",
        mask_indices: Sequence[int] | None = None,
        prediction_steps: int = 5,
        future_interval_seconds: float | None = None,
        overflow: OverflowStrategy = "error",
    ) -> InferenceResult:
        """Run UniTraj inference on one PostGIS ``LINESTRING M``.

        X is longitude, Y is latitude, and M is a timestamp. Only consecutive
        timestamp differences are passed to the model. ``embedding`` returns
        the 128-dimensional encoder CLS vector. ``recovery`` reconstructs the
        requested zero-based indices. ``prediction`` appends Last-N slots; the
        paper evaluates five future points. The released architecture is fixed
        at 200 model positions.
        """

        if task not in {"embedding", "recovery", "prediction"}:
            raise ValueError(f"Unsupported inference task: {task}")
        if overflow not in {"error", "uniform"}:
            raise ValueError(f"Unsupported overflow strategy: {overflow}")

        parsed = parse_linestringm(trajectory)
        self._validate_geometry(parsed)

        if task == "prediction":
            if mask_indices is not None:
                raise ValueError("mask_indices is not used for prediction")
            parsed, prediction_indices = self._append_prediction_slots(
                parsed, prediction_steps, future_interval_seconds
            )
            prepared = self._prepare(parsed, overflow=overflow)
            if overflow == "uniform" and prepared.point_count != len(parsed.coordinates):
                raise ValueError(
                    "overflow='uniform' is not supported for prediction because "
                    "it would change the Last-N target positions"
                )
            decoded = self._decode(prepared, prediction_indices)
            result = parsed.coordinates.copy()
            result[list(prediction_indices), :2] = decoded[list(prediction_indices), :2]
            predicted = result[list(prediction_indices)]
            return self._spatial_result(
                "prediction", parsed, result, prediction_indices, predicted, prepared
            )

        prepared = self._prepare(parsed, overflow=overflow)
        if task == "embedding":
            if mask_indices is not None:
                raise ValueError("mask_indices is only valid for recovery")
            return InferenceResult(
                task="embedding",
                embedding=self._embed(prepared),
                metadata=self._metadata(prepared),
            )

        requested_masks = self._validate_mask_indices(mask_indices, prepared)
        decoded = self._decode(prepared, requested_masks)
        recovered = prepared.parsed.coordinates.copy()
        recovered[list(requested_masks), :2] = decoded[list(requested_masks), :2]
        return self._spatial_result(
            "recovery",
            prepared.parsed,
            recovered,
            requested_masks,
            recovered[list(requested_masks)],
            prepared,
        )

    def _load_checkpoint(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        state_dict: Mapping[str, Tensor]
        if _is_state_dict(payload):
            state_dict = cast(Mapping[str, Tensor], payload)
        elif isinstance(payload, Mapping):
            candidate = payload.get("state_dict") or payload.get("model_state_dict")
            if not _is_state_dict(candidate):
                raise ValueError(
                    "Checkpoint must be a state dict or contain 'state_dict' or "
                    "'model_state_dict'"
                )
            state_dict = cast(Mapping[str, Tensor], candidate)
        else:
            raise ValueError("Unsupported checkpoint payload")

        normalized = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(normalized, strict=True)

    def _validate_geometry(self, parsed: ParsedLineStringM) -> None:
        if (
            self.expected_srid is not None
            and parsed.srid is not None
            and parsed.srid != self.expected_srid
        ):
            raise ValueError(
                f"Expected SRID {self.expected_srid}, received SRID {parsed.srid}"
            )
        x = parsed.coordinates[:, 0]
        y = parsed.coordinates[:, 1]
        if self.expected_srid == 4326:
            if np.any((x < -180.0) | (x > 180.0)):
                raise ValueError("Longitude X values must be within [-180, 180]")
            if np.any((y < -90.0) | (y > 90.0)):
                raise ValueError("Latitude Y values must be within [-90, 90]")
        if np.any(np.diff(parsed.coordinates[:, 2]) <= 0):
            raise ValueError("M timestamps must be strictly increasing")

    def _prepare(
        self, parsed: ParsedLineStringM, *, overflow: OverflowStrategy
    ) -> _PreparedTrajectory:
        coordinates = parsed.coordinates
        source_indices = np.arange(len(coordinates), dtype=np.int64)
        if len(coordinates) > TRAJECTORY_LENGTH:
            if overflow == "error":
                raise ValueError(
                    f"UniTraj accepts at most {TRAJECTORY_LENGTH} points; "
                    "segment the trajectory or pass overflow='uniform'"
                )
            source_indices = np.linspace(
                0, len(coordinates) - 1, TRAJECTORY_LENGTH, dtype=np.int64
            )
            coordinates = coordinates[source_indices]
            parsed = ParsedLineStringM(coordinates, parsed.srid)

        point_count = len(coordinates)
        if point_count < MIN_TRAJECTORY_POINTS:
            raise ValueError(
                f"UniTraj was trained on trajectories with at least "
                f"{MIN_TRAJECTORY_POINTS} points; received {point_count}"
            )

        original_xy = coordinates[0, :2].copy()
        relative_xy = coordinates[:, :2] - original_xy
        normalized_xy = (
            relative_xy.astype(np.float32) - COORDINATE_MEAN
        ) / COORDINATE_STD
        padded_xy = np.zeros((TRAJECTORY_LENGTH, 2), dtype=np.float32)
        padded_xy[:point_count] = normalized_xy

        m_seconds = (
            coordinates[:, 2] * _SECONDS_PER_TIMESTAMP_UNIT[self.timestamp_unit]
        )
        intervals = np.zeros(TRAJECTORY_LENGTH, dtype=np.float32)
        intervals[1:point_count] = np.diff(m_seconds).astype(np.float32)

        return _PreparedTrajectory(
            parsed=parsed,
            spatial=torch.from_numpy(padded_xy.T).unsqueeze(0).to(self.device),
            intervals=torch.from_numpy(intervals).unsqueeze(0).to(self.device),
            original_xy=original_xy,
            point_count=point_count,
            selected_source_indices=tuple(int(i) for i in source_indices),
        )

    def _embed(self, prepared: _PreparedTrajectory) -> NDArray[np.float32]:
        old_mask_ratio = self.model.encoder.shuffle.mask_ratio
        self.model.encoder.shuffle.mask_ratio = 0.0
        try:
            with self._rng(), torch.no_grad():
                interval_embeddings = self.model.interval_embedding(
                    prepared.intervals.unsqueeze(-1)
                )
                features, _ = self.model.encoder(
                    prepared.spatial, interval_embeddings, None
                )
                embedding = features[0, 0]
        finally:
            self.model.encoder.shuffle.mask_ratio = old_mask_ratio
        return embedding.detach().cpu().numpy().astype(np.float32, copy=True)

    def _decode(
        self, prepared: _PreparedTrajectory, mask_indices: Sequence[int]
    ) -> NDArray[np.float64]:
        masks = torch.tensor(
            [list(mask_indices)], dtype=torch.long, device=self.device
        )
        with self._rng(), torch.no_grad():
            predicted, _ = self.model(prepared.spatial, prepared.intervals, masks)
        normalized = predicted[0].transpose(0, 1).detach().cpu().numpy()
        relative = normalized.astype(np.float64) * COORDINATE_STD + COORDINATE_MEAN
        xy = relative + prepared.original_xy
        output = prepared.parsed.coordinates.copy()
        output[:, :2] = xy[: prepared.point_count]
        return output

    def _validate_mask_indices(
        self,
        mask_indices: Sequence[int] | None,
        prepared: _PreparedTrajectory,
    ) -> tuple[int, ...]:
        if mask_indices is None or len(mask_indices) == 0:
            raise ValueError("recovery requires at least one mask index")
        indices = tuple(int(index) for index in mask_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("mask_indices must be unique")
        if any(index < 0 or index >= prepared.point_count for index in indices):
            raise ValueError(
                f"mask_indices must be within [0, {prepared.point_count - 1}]"
            )
        if prepared.selected_source_indices != tuple(range(prepared.point_count)):
            raise ValueError(
                "recovery cannot combine mask_indices with overflow='uniform' because "
                "source indices would be ambiguous"
            )
        return tuple(sorted(indices))

    def _append_prediction_slots(
        self,
        parsed: ParsedLineStringM,
        prediction_steps: int,
        future_interval_seconds: float | None,
    ) -> tuple[ParsedLineStringM, tuple[int, ...]]:
        if not 1 <= prediction_steps <= 32:
            raise ValueError("prediction_steps must be between 1 and 32")
        count = len(parsed.coordinates)
        if count + prediction_steps > TRAJECTORY_LENGTH:
            raise ValueError("Observed points plus prediction_steps cannot exceed 200")

        m = parsed.coordinates[:, 2]
        seconds_per_unit = _SECONDS_PER_TIMESTAMP_UNIT[self.timestamp_unit]
        if future_interval_seconds is None:
            interval_seconds = float(np.median(np.diff(m) * seconds_per_unit))
        else:
            interval_seconds = float(future_interval_seconds)
        if not np.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("future_interval_seconds must be finite and positive")
        interval_in_source_units = interval_seconds / seconds_per_unit

        future = np.empty((prediction_steps, 3), dtype=np.float64)
        future[:, :2] = parsed.coordinates[-1, :2]
        future[:, 2] = parsed.coordinates[-1, 2] + interval_in_source_units * np.arange(
            1, prediction_steps + 1, dtype=np.float64
        )
        augmented = ParsedLineStringM(
            np.vstack([parsed.coordinates, future]), parsed.srid
        )
        indices = tuple(range(count, count + prediction_steps))
        return augmented, indices

    def _spatial_result(
        self,
        task: Literal["recovery", "prediction"],
        parsed: ParsedLineStringM,
        coordinates: NDArray[np.float64],
        mask_indices: Sequence[int],
        predicted: NDArray[np.float64],
        prepared: _PreparedTrajectory,
    ) -> InferenceResult:
        result_geometry = ParsedLineStringM(coordinates, parsed.srid)
        return InferenceResult(
            task=task,
            coordinates=coordinates,
            predicted_coordinates=np.asarray(predicted, dtype=np.float64).copy(),
            linestringm_ewkt=to_ewkt(result_geometry),
            mask_indices=tuple(int(index) for index in mask_indices),
            metadata=self._metadata(prepared),
        )

    def _metadata(self, prepared: _PreparedTrajectory) -> Mapping[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "input_points": prepared.point_count,
            "model_points": TRAJECTORY_LENGTH,
            "timestamp_unit": self.timestamp_unit,
            "srid": prepared.parsed.srid or self.expected_srid,
            "selected_source_indices": prepared.selected_source_indices,
            "seed": self.seed,
        }

    @contextmanager
    def _rng(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def infer_linestringm(
    trajectory: PostGISLineStringM,
    checkpoint_path: str | Path,
    *,
    task: InferenceTask = "embedding",
    mask_indices: Sequence[int] | None = None,
    prediction_steps: int = 5,
    future_interval_seconds: float | None = None,
    device: str | torch.device = "cpu",
    timestamp_unit: TimestampUnit = "s",
    expected_srid: int | None = 4326,
    seed: int = 0,
    overflow: OverflowStrategy = "error",
) -> InferenceResult:
    """One-call interface for UniTraj inference from PostGIS LINESTRING M."""

    engine = UniTrajInference(
        checkpoint_path,
        device=device,
        timestamp_unit=timestamp_unit,
        expected_srid=expected_srid,
        seed=seed,
    )
    return engine.infer(
        trajectory,
        task=task,
        mask_indices=mask_indices,
        prediction_steps=prediction_steps,
        future_interval_seconds=future_interval_seconds,
        overflow=overflow,
    )


def _is_state_dict(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, Tensor)
        for key, tensor in value.items()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
