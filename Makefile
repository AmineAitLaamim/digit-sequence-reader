.PHONY: infer generate ctc-infer ctc-generate ctc-gen-one ctc-train ctc-eval-extrap

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
	python -m src.ctc.inference --checkpoint model/best_ctc.pt --image samples/sample_L100_1186537628872393878000142726628586789844612808041049807894835729434881075053828820551863618760738418.png --visualize

# Generate local CTC-style samples (enforces width >= digits*16)
ctc-generate:
	python -m src.ctc.generate_samples

# Generate a SINGLE sample of an exact length L. Usage:
#   make ctc-gen-one L=7                 # → samples/sample_L7_<digits>.png
#   make ctc-gen-one L=25 OUT=my.png     # custom output path
#   make ctc-gen-one L=12 COUNT=5        # 5 different random samples of length 12
#   make ctc-gen-one L=7 AUG=1           # with training-style augmentation
ctc-gen-one:
	python -m src.ctc.generate_one --length $(L) \
		$(if $(OUT),--out $(OUT),) \
		$(if $(COUNT),--count $(COUNT),) \
		$(if $(AUG),--augment,) \
		$(if $(SEED),--seed $(SEED),)

# Train CTC model (configure --drive_path before running)
ctc-train:
	python -m src.ctc.train --drive_path ./model

# Length-extrapolation evaluation (synthesises L in {1,3,5,...,50} and plots)
ctc-eval-extrap:
	python -m src.ctc.evaluate_extrapolation --checkpoint ./model/best_ctc.pt --out_dir ./model/metrics
