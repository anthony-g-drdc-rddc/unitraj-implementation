# UniTraj inference for PostGIS `LINESTRING M`

Inference-only implementation of the architecture released with **UniTraj: Learning a Universal Trajectory Foundation Model from Billion-Scale Worldwide Traces**.

The encoder, decoder, spatio-temporal tokenization, RoPE attention, masking/reordering behavior, and published hyperparameters are adapted from the authors' Apache-2.0 implementation at [`Yasoz/UniTraj`](https://github.com/Yasoz/UniTraj). The package adds a strict inference boundary for PostGIS trajectories.

## Checkpoint status

The paper repository publishes the model code, training entry point, and a 1,000-trajectory sample, but it does not currently publish a pretrained `.pt` checkpoint in the repository or document one in its README. This project therefore **never runs inference with random weights**. A compatible checkpoint is required at construction time.

A compatible checkpoint is either:

- the plain `state_dict` saved by the authors' training script;
- a dictionary containing `state_dict`; or
- a dictionary containing `model_state_dict`.

`torch.nn.DataParallel` checkpoints with `module.` prefixes are accepted.

## Public interface

```python
from pathlib import Path
from typing import Literal, Sequence

from unitraj import InferenceResult


def infer_linestringm(
    trajectory: str | bytes | bytearray | memoryview,
    checkpoint_path: str | Path,
    *,
    task: Literal["embedding", "recovery", "prediction"] = "embedding",
    mask_indices: Sequence[int] | None = None,
    prediction_steps: int = 5,
    future_interval_seconds: float | None = None,
    device: str = "cpu",
    timestamp_unit: Literal["s", "ms", "us", "ns"] = "s",
    expected_srid: int | None = 4326,
    seed: int = 0,
    overflow: Literal["error", "uniform"] = "error",
) -> InferenceResult: ...
```

### Input contract

- Geometry: PostGIS `LINESTRING M` as EWKT/WKT, WKB/EWKB bytes, or hex WKB.
- X: longitude.
- Y: latitude.
- M: timestamp. Absolute timestamp values are not embedded; consecutive differences are converted to seconds and passed to UniTraj.
- Default CRS: WGS84 / SRID 4326.
- Timestamps must be strictly increasing.
- The training distribution starts at 36 points.
- The released architecture is fixed at 200 model positions. More than 200 points fails by default. `overflow="uniform"` performs explicit deterministic downsampling for embedding only.

### Output contract

- `embedding`: 128-dimensional CLS representation.
- `recovery`: full trajectory with requested zero-based points replaced by decoder output.
- `prediction`: observed trajectory plus predicted future points. Five future points is the paper's evaluation setting.
- Spatial outputs preserve the original M values and serialize as PostGIS-compatible EWKT.

## Install

```bash
python -m pip install -e .
```

## Reusable engine

Load the checkpoint once for production use:

```python
from unitraj import UniTrajInference

engine = UniTrajInference(
    "checkpoints/unitraj.pt",
    device="cuda",
    timestamp_unit="s",
)

trajectory = (
    "SRID=4326;LINESTRING M ("
    "-73.5673 45.5017 1720000000, "
    "-73.5672 45.5018 1720000001"
    # ... at least 36 points total ...
    ")"
)

result = engine.infer(trajectory, task="embedding")
print(result.embedding.shape)  # (128,)
```

## PostGIS / psycopg example

```python
row = connection.execute(
    "SELECT ST_AsEWKB(trajectory) FROM tracks WHERE id = %s",
    (track_id,),
).fetchone()

result = engine.infer(row[0], task="prediction", prediction_steps=5)

connection.execute(
    "UPDATE tracks SET predicted = ST_GeomFromEWKT(%s) WHERE id = %s",
    (result.linestringm_ewkt, track_id),
)
```

## Recovery

```python
result = engine.infer(
    trajectory,
    task="recovery",
    mask_indices=[12, 13, 14, 15],
)
print(result.predicted_coordinates)
```

## Prediction

```python
result = engine.infer(
    trajectory,
    task="prediction",
    prediction_steps=5,
    future_interval_seconds=1.0,
)
print(result.predicted_coordinates)  # shape: (5, 3)
```

## Model fidelity

Published settings implemented here:

| Setting | Value |
|---|---:|
| trajectory positions | 200 |
| patch size | 1 |
| embedding dimension | 128 |
| encoder blocks | 8 |
| decoder blocks | 4 |
| attention heads | 4 |
| mask ratio | 0.5 |
| parameters | 2,376,194 |
| RoPE maximum sequence length | 512 |

The official coordinate preprocessing is preserved: subtract the first X/Y point, then normalize with the mean and standard deviation from the authors' dataset code. The official shuffle is seeded and its RNG state is restored after each call, making repeated inference deterministic without perturbing application-level random state.

## Post-training guidance

No post-training is needed for encoder embeddings, trajectory recovery, or Last-N prediction **after a real pretrained foundation checkpoint is supplied**.

Task-specific adaptation is needed for outputs that the foundation model does not directly produce:

- classification: pool encoder features and train a 2–3 layer MLP head; either freeze the backbone first or fine-tune end-to-end;
- anomaly detection: fit a density, distance, one-class, or supervised head over frozen embeddings;
- similarity and retrieval: optionally contrastively fine-tune the encoder, then index normalized embeddings;
- region/domain adaptation: fine-tune the backbone with the same masked reconstruction objective on representative local trajectories.

See [`docs/post-training.md`](docs/post-training.md) for a concrete staged plan.

## Validation

```bash
python -m pip install -e ".[dev]"
pytest
```

The test suite covers EWKT/WKT, ISO WKB, PostGIS EWKB, checkpoint loading, the exact published parameter count, deterministic embeddings, recovery, and five-step prediction.

## License and attribution

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
