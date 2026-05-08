config = {
    # Data
    'img_height': 64, 'min_seq_len': 3, 'max_seq_len': 7,
    'gap_min': 0, 'gap_max': 12,
    'overlap_max': 8, 'overlap_prob_max': 0.10,
    'overlap_start_epoch': 5,
    'seq_background_noise': 0.2,
    'seq_rotation': 3,
    'train_size': 100_000, 'val_size': 10_000, 'test_size': 10_000,

    # Data sources
    'datasets': ['emnist_digits', 'qmnist', 'usps'],

    # Augmentation
    'aug_warmup_epochs': 10,
    'augment': True,
    'aug_noise_var': (5, 20), 'aug_blur_limit': 3,
    'aug_erasing_p': 0.3,
    'aug_brightness': 0.3,
    'aug_contrast': 0.3,

    # Vocab
    'vocab_size': 13, 'SOS_IDX': 10, 'EOS_IDX': 11, 'PAD_IDX': 12,

    # Model
    'cnn_channels': [32, 64, 128], 'cnn_dropout': 0.3,
    'embed_dim': 64, 'hidden_size': 256,
    'enc_dropout': 0.3, 'dec_dropout': 0.3, 'attention_dim': 128,

    # Training
    'batch_size': 64, 'epochs': 30, 'lr': 1e-3,
    'teacher_forcing_ratio': 0.5, 'clip_grad': 1.0,
    'num_workers': 2,
    'best_metric': 'val_seq_acc',

    # LR Scheduler (ReduceLROnPlateau)
    'lr_patience': 5, 'lr_factor': 0.5, 'lr_min': 1e-6,

    # Early stopping
    'early_stop_patience': 12,

    # Paths (overridden by argparse in train.py)
    'drive_path': '/content/drive/MyDrive/digit-sequence-reader',
    'data_path': './data',
}
