# Suggested post-training

The foundation checkpoint is sufficient for direct encoder embeddings, masked recovery, and Last-N prediction. Post-training should be introduced only when the application needs a task-specific decision surface or domain calibration.

## Stage 1 — frozen-backbone baseline

1. Convert each `LINESTRING M` with the exact production preprocessing.
2. Extract the 128-dimensional CLS embedding.
3. Freeze UniTraj.
4. Train the smallest suitable head:
   - classification: LayerNorm → Linear(128, 256) → ReLU → Dropout → Linear(256, classes);
   - regression: LayerNorm → Linear(128, 128) → ReLU → Linear(128, targets);
   - anomaly detection: robust covariance, isolation forest, one-class SVM, or a calibrated binary head;
   - retrieval: L2-normalize embeddings and use cosine or inner-product search.
5. Evaluate by geographic and temporal holdout, not only random splits.

This is the recommended first deployment because it preserves the pretrained representation and minimizes overfitting.

## Stage 2 — partial fine-tuning

When the frozen baseline is insufficient:

1. keep the tokenizer and lower encoder blocks frozen;
2. unfreeze the final two encoder blocks, final layer norm, and task head;
3. use a backbone learning rate 10–100× lower than the head learning rate;
4. use early stopping and monitor out-of-region performance;
5. compare against the frozen baseline before accepting the added complexity.

## Stage 3 — full domain adaptation

For a materially different fleet, sensor, sampling regime, or movement mode:

1. pretrain further with the same reconstruction objective;
2. retain the four masking families: random, RDP key points, block, and Last-N;
3. preserve the published coordinate normalization unless a full retraining and compatibility break is intended;
4. mix local data with a representative global replay sample to reduce catastrophic forgetting;
5. version the adapted checkpoint with training-data window, geographic scope, normalization constants, and metric results.

## Data preparation

- WGS84 longitude/latitude.
- Strictly increasing timestamps.
- At least 36 points per training example.
- Reject duplicate points and impossible speeds before training.
- Segment or resample to 200 model positions.
- Split by vehicle/user and time to prevent leakage.

## Evaluation

- recovery/prediction: haversine MAE and RMSE in meters;
- classification: macro-F1 plus per-class recall;
- anomaly detection: precision-recall at operational alert rates;
- retrieval: Recall@K, mAP, and geographic diversity of neighbors;
- robustness: missing blocks, irregular intervals, GPS noise, and cross-region transfer.
