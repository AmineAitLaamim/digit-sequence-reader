.PHONY: infer generate ctc-infer ctc-generate ctc-train ctc-eval-extrap

# ── Seq2Seq (old, autoregressive) ──────────────────────────────────────
# Run inference locally with visualization
infer:
	python -m src.seq2seq.inference --checkpoint model/best_model.pt --image samples/test.png --visualize

# Generate local digit sequence samples
generate:
	python -m src.seq2seq.generate_samples

# ── CTC (new, parallel non-autoregressive) ─────────────────────────────
# Run CTC inference locally with visualization
ctc-infer:
	python -m src.ctc.inference --checkpoint model/best_model.pt --image samples/test.png --visualize

# Generate local CTC-style samples (enforces width >= digits*16)
ctc-generate:
	python -m src.ctc.generate_samples

# Train CTC model (configure --drive_path before running)
ctc-train:
	python -m src.ctc.train --drive_path ./model

# Length-extrapolation evaluation (synthesises L in {1,3,5,...,50} and plots)
ctc-eval-extrap:
	python -m src.ctc.evaluate_extrapolation --checkpoint ./model/checkpoints_ctc/best_ctc.pt --out_dir ./model/metrics
