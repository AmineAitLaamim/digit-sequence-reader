.PHONY: infer generate ctc-infer ctc-generate ctc-gen-one ctc-train ctc-eval-extrap \
        uncapped-infer uncapped-eval-extrap \
        seq2seq-eval-extrap

# ── Seq2Seq (old, autoregressive) ──────────────────────────────────────
# Run inference locally with visualization
infer:
	python -m src.seq2seq.inference --checkpoint model/best_model_v1.pt --image samples/test.png --visualize

# Generate local digit sequence samples
generate:
	python -m src.seq2seq.generate_samples

# ── CTC (new, parallel non-autoregressive) ─────────────────────────────
# Run CTC inference locally with visualization.
# Optional accuracy metrics:
#   make ctc-infer                                    # no GT comparison
#   make ctc-infer GT=12345                            # explicit ground truth
#   make ctc-infer IMAGE=samples/sample_L7_1234567.png # auto-extract GT from filename
#   make ctc-infer NO_GT=1                             # force skip GT comparison
#IMAGE = samples/sample_L100_1186537628872393878000142726628586789844612808041049807894835729434881075053828820551863618760738418.png
#GT = 1186537628872393878000142726628586789844612808041049807894835729434881075053828820551863618760738418
ctc-infer:
	python -m src.ctc.inference \
		--checkpoint model/best_ctc.pt \
		--image $(if $(IMAGE),$(IMAGE),samples/test.png) \
		--visualize \
		$(if $(GT),--ground-truth $(GT),) \
		$(if $(NO_GT),--no-gt,)

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

# Length-extrapolation evaluation — baseline checkpoint
ctc-eval-extrap:
	python -m src.ctc.evaluate_extrapolation --checkpoint ./model/best_ctc.pt --out_dir ./model/metrics

# Length-extrapolation evaluation — Seq2Seq checkpoint
# Outputs go to ./model/metrics_seq2seq to keep them separate from CTC results.
# Override checkpoint:  make seq2seq-eval-extrap CKPT=model/my_seq2seq.pt
# Override output dir: make seq2seq-eval-extrap OUT_DIR=./model/metrics_seq2seq
SEQ2SEQ_CKPT    = model/best_model.pt
SEQ2SEQ_OUT_DIR = ./model/metrics_seq2seq
seq2seq-eval-extrap:
	python -m src.seq2seq.evaluate_extrapolation \
		--checkpoint $(if $(CKPT),$(CKPT),$(SEQ2SEQ_CKPT)) \
		--out_dir $(if $(OUT_DIR),$(OUT_DIR),$(SEQ2SEQ_OUT_DIR))

# ── CRNN_CTC_Uncapped (ablation model) ─────────────────────────────────
# Run inference with the uncapped-receptive-field ablation checkpoint.
# Uses the same src.ctc.inference entry-point (identical forward contract).
# Optional flags — same conventions as ctc-infer:
#   make uncapped-infer                                     # no GT comparison
#   make uncapped-infer GT=12345                            # explicit ground truth
#   make uncapped-infer IMAGE=samples/my_seq.png            # custom image
#   make uncapped-infer NO_GT=1                             # skip GT comparison
UNCAPPED_CKPT = model/best_ctc_uncapped.pt
uncapped-infer:
	python -m src.ctc.inference \
		--checkpoint $(UNCAPPED_CKPT) \
		--image $(if $(IMAGE),$(IMAGE),samples/test.png) \
		--visualize \
		$(if $(GT),--ground-truth $(GT),) \
		$(if $(NO_GT),--no-gt,)

# Length-extrapolation evaluation — ablation checkpoint.
# Run this after training the uncapped model to generate the comparison table.
#   make uncapped-eval-extrap
#   make uncapped-eval-extrap OUT_DIR=./model/metrics_uncapped
UNCAPPED_OUT_DIR = ./model/metrics_uncapped
uncapped-eval-extrap:
	python -m src.ctc.evaluate_extrapolation \
		--checkpoint $(UNCAPPED_CKPT) \
		--out_dir $(if $(OUT_DIR),$(OUT_DIR),$(UNCAPPED_OUT_DIR))
