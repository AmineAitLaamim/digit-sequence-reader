.PHONY: infer generate

# Run inference locally with visualization
infer:
	python src/inference.py --checkpoint model/best_model.pt --image samples/sample_6_8675476.png --visualize

# Generate local digit sequence samples
generate:
	python src/generate_samples.py
