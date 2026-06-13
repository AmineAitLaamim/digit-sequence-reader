config = {
    # Data
    'img_height': 64, 'min_seq_len': 3, 'max_seq_len': 7, 'max_seq_len_final': 12,
    'gap_min': 0, 'gap_max': 12,
    # Safety cap for the inference decode loop — must never limit legitimate
    # extrapolation. Set this large enough that only a truly broken model
    # (e.g. right after random init) would ever reach it.
    'max_decode_steps': 200,
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
    # Scheduled sampling: anneal teacher-forcing ratio from tf_start -> tf_end
    # across training to combat exposure bias / train-inference mismatch.
    'tf_start': 0.5, 'tf_end': 0.2, 'tf_anneal_epochs': 20,
    'num_workers': 2,
    'best_metric': 'val_seq_acc',

    # LR Scheduler (ReduceLROnPlateau on val_seq_acc, mode='max')
    'lr_patience': 5, 'lr_factor': 0.5, 'lr_min': 1e-6,

    # Divergence guard: if val_loss spikes > div_guard_mult × best loss,
    # roll back weights to best checkpoint instead of saving a bad "best".
    'div_guard_mult': 5.0,

    # Early stopping
    'early_stop_patience': 12,

    # Paths (overridden by argparse in train.py)
    'drive_path': '/content/drive/MyDrive/digit-sequence-reader',
    'data_path': './data',
}
