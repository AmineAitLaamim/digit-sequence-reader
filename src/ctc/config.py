config = {
    # ── Data ─────────────────────────────────────────────────────────────
    # Image: fixed height 64, variable width.
    # The 2D CNN downsamples width by 8x, so width must be a multiple of 8
    # for the residual 1D stack to operate cleanly. We enforce a minimum
    # width per image so that T = W//8 is always >> L (the number of digits),
    # which is required for CTC to find valid alignments.
    'img_height': 64,
    'min_seq_len': 3,
    'max_seq_len': 7,         # curriculum max during training
    'max_seq_len_final': 12,  # final max for the curriculum tail
    'min_image_width': 64,                # absolute lower bound
    'width_per_digit': 16,                # min pixels of width allocated per digit
    'gap_min': 0,
    'gap_max': 12,
    'overlap_max': 8,
    'overlap_prob_max': 0.10,
    'overlap_start_epoch': 5,
    'seq_background_noise': 0.2,
    'seq_rotation': 3,
    'train_size': 100_000,
    'val_size': 10_000,
    'test_size': 10_000,

    # ── Data sources ────────────────────────────────────────────────────
    'datasets': ['emnist_digits', 'qmnist', 'usps'],

    # ── Augmentation ────────────────────────────────────────────────────
    'aug_warmup_epochs': 10,
    'augment': True,
    'aug_noise_var': (5, 20),
    'aug_blur_limit': 3,
    'aug_erasing_p': 0.3,
    'aug_brightness': 0.3,
    'aug_contrast': 0.3,

    # ── Vocab (CTC) ─────────────────────────────────────────────────────
    # 10 digits (0-9) + 1 BLANK token. NO SOS, NO EOS, NO PAD.
    # CTC inherently handles variable-length outputs via the BLANK token,
    # which is also why we don't need explicit end-of-sequence markers.
    'vocab_size': 11,
    'BLANK_IDX': 10,  # PyTorch CTCLoss requires blank as the LAST index by convention

    # ── Model (2D CNN + 1D Dilated Residual CNN) ───────────────────────
    # 2D CNN: 4 conv blocks. The first 3 each halve H and W (H: 64→8, W: ×1/8).
    # The 4th block keeps spatial dims (stride=1, padding) and only grows
    # channels to 512 — this is what gives us 8x width downsampling and
    # exactly height=8 after the spatial collapse.
    'cnn_channels_2d': [64, 128, 256, 512],
    'cnn_kernel_size': 3,
    'cnn_dropout': 0.1,

    # 1D Dilated CNN: 4 residual blocks with dilations 1, 2, 4, 8.
    # Receptive field ≈ 2 * sum_{d} (k-1) * d = 2*(2+4+8+16) = 60 steps,
    # which is plenty of local context without ever seeing the full
    # sequence (and therefore prevents long-range length bias).
    'hidden_dim': 256,
    'num_res_blocks': 4,
    'dilations': [1, 2, 4, 8],
    'kernel_size_1d': 3,
    'resblock_dropout': 0.1,

    # ── Training ────────────────────────────────────────────────────────
    'batch_size': 64,
    'epochs': 30,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'clip_grad': 1.0,
    'num_workers': 2,
    'best_metric': 'val_seq_acc',

    # LR Scheduler (ReduceLROnPlateau)
    'lr_patience': 5,
    'lr_factor': 0.5,
    'lr_min': 1e-6,

    # Early stopping
    'early_stop_patience': 12,

    # ── Paths (overridden by argparse in train.py) ─────────────────────
    'drive_path': '/content/drive/MyDrive/digit-sequence-reader',
    'data_path': './data',
}
