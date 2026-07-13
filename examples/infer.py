"""Minimal PostGIS LINESTRING M inference example."""

from unitraj import UniTrajInference

CHECKPOINT_PATH = "checkpoints/unitraj.pt"
DEVICE = "cpu"

POINTS = [
    f"{-73.5673 + i * 0.00001} {45.5017 + i * 0.000005} {1720000000 + i}"
    for i in range(36)
]
TRAJECTORY = "SRID=4326;LINESTRING M (" + ", ".join(POINTS) + ")"

engine = UniTrajInference(CHECKPOINT_PATH, device=DEVICE)
result = engine.infer(TRAJECTORY, task="embedding")
print(result.embedding)
