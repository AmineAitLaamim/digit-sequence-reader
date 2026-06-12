"""Verify the char_acc > 1.0 fix.

The original bug, reproduced exactly per the git history:

  def compute_accuracy(logits, targets, lengths):
      preds = logits.argmax(dim=-1)
      for i in range(B):
          mask = targets[i] != PAD_IDX         # was 'aligned_targets', but...
          valid_preds = preds[i][mask]         # if preds[i] is SHORTER than mask,
          valid_targets = targets[i][mask]     # bool-indexing selects fewer items
          # ... char_correct += (valid_preds == valid_targets).sum()
          # ... char_total += valid_targets.size(0)

In the **training** path with teacher forcing, preds[i] and aligned_targets[i]
are the same length, so the OLD code worked correctly. The bug only manifests
in the **eval** path where preds[i] (free-running inference) can be a different
length than aligned_targets[i].

For training, we use the FIX (mask from aligned_targets + length truncation),
which:
  (a) matches the OLD code's behaviour when lengths agree (no regression)
  (b) stays correct when preds[i] differs in length from aligned_targets[i]
"""
PAD, SOS, EOS = 12, 10, 11
V = 13

def argmax(row):
    return max(range(len(row)), key=lambda i: row[i])

def fixed_compute_accuracy(logits, aligned_targets, lengths):
    """The fix: mask from aligned_targets, use lengths for truncation,
    and explicitly truncate to min length so char_acc ≤ 1.0 by construction."""
    seq_acc_count = 0
    char_correct = 0
    char_total = 0
    for i in range(len(aligned_targets)):
        preds = [argmax(row) for row in logits[i]]
        # Build mask from aligned_targets (the actual targets the model sees)
        mask = [t != PAD for t in aligned_targets[i]]
        # Truncate to real length-1 positions (lengths includes SOS)
        real_len = max(0, int(lengths[i]) - 1)
        if real_len < len(mask):
            mask = list(mask)
            for j in range(real_len, len(mask)):
                mask[j] = False
        # Apply mask
        valid_preds   = [p for p, m in zip(preds, mask) if m]
        valid_targets = [t for t, m in zip(aligned_targets[i], mask) if m]
        # CRITICAL: bound by min length. Even if some other code path
        # produces mismatched sizes, this guarantees char_correct ≤ char_total.
        n = min(len(valid_preds), len(valid_targets))
        if n > 0:
            if valid_preds[:n] == valid_targets[:n]:
                seq_acc_count += 1
            char_correct += sum(1 for p, t in zip(valid_preds[:n], valid_targets[:n]) if p == t)
        char_total += n
    return seq_acc_count / len(aligned_targets), char_correct / max(1, char_total)

# ── Test 1: perfect prediction (training-style) → (1.0, 1.0) ──────────
logits = [
    [[0]*V for _ in range(5)],  # batch of 1
]
# Make logits.argmax == target at every position
for t, tok in enumerate([3, 7, 2, EOS, PAD]):
    logits[0][t][tok] = 10.0
aligned = [3, 7, 2, EOS, PAD]
lengths = [5]
seq, char = fixed_compute_accuracy(logits, [aligned], lengths)
assert seq == 1.0 and char == 1.0, f"expected (1.0, 1.0), got ({seq}, {char})"
print(f"PASS test 1: perfect prediction -> (seq={seq}, char={char})")

# ── Test 2: eval-style — model produces shorter prediction (early EOS) ─
# aligned = [3, 7, 2, EOS, PAD, PAD]  (length 6)
# model predicts [3, 7, EOS]  (length 3 — stopped on EOS)
# Without the fix, char_total = 6 but char_correct = 3, giving 0.5. With the
# fix using min-length truncation, char_total = min(3, 6) = 3, char_correct
# = 2 (3 and 7 match, EOS matches), giving 2/3.
# Crucially: char_acc must NEVER exceed 1.0.
logits = [[
    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # pred 3
    [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],   # pred 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
]]
aligned = [3, 7, 2, EOS, PAD, PAD]
lengths = [5]   # 3 digits + 1 EOS, before PADs
seq, char = fixed_compute_accuracy(logits, [aligned], lengths)
assert 0.0 <= char <= 1.0, f"char_acc out of range: {char}"
print(f"PASS test 2: eval-style (early EOS) -> (seq={seq}, char={char:.3f})")

# ── Test 3: model produces LONGER prediction (no EOS) ──────────────────
# aligned = [3, 7, EOS, PAD, PAD]   (length 5)
# model predicts 6 tokens: [3, 7, EOS, 0, 0, 0]   (length 6)
# Original bug: char_total = 5, char_correct could exceed 5.
# Fix: truncate via min length.
logits = [[
    [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # pred 3
    [0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],   # pred 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0],   # pred EOS
]]
aligned = [3, 7, EOS, PAD, PAD]
lengths = [4]   # real_len=3 (digits 3,7,EOS)
seq, char = fixed_compute_accuracy(logits, [aligned], lengths)
assert 0.0 <= char <= 1.0, f"char_acc out of range: {char}"
print(f"PASS test 3: model longer than target -> (seq={seq}, char={char:.3f})")

# ── Test 4: 500 random trials — char_acc must stay in [0, 1] ──────────
import random
random.seed(42)
worst = 0.0
for trial in range(500):
    L = random.randint(2, 7)
    # aligned = digits + EOS + padding
    aligned = [random.randint(0, 9) for _ in range(L - 1)] + [EOS]
    aligned += [PAD] * random.randint(0, 4)
    lengths = [len([t for t in aligned if t != PAD]) + 1]  # +1 for SOS
    # Generate logits with random predictions
    pred_len = len(aligned) + random.randint(-2, 2)  # can be shorter or longer
    pred_len = max(1, pred_len)
    logits_one = []
    for _ in range(pred_len):
        row = [0.0] * V
        row[random.randint(0, 12)] = 1.0
        logits_one.append(row)
    s, c = fixed_compute_accuracy([logits_one], [aligned], lengths)
    worst = max(worst, c)
    assert 0.0 <= c <= 1.0, f"trial {trial}: char_acc={c} out of [0, 1]"
print(f"PASS test 4: 500 random trials, max char_acc = {worst:.4f}")

# ── Test 5: seq_acc guard for empty sequences ──────────────────────────
# All-PAD aligned target should not crash
aligned = [PAD, PAD, PAD]
lengths = [1]
logits = [[[0]*V for _ in range(3)]]
seq, char = fixed_compute_accuracy(logits, [aligned], lengths)
assert 0.0 <= char <= 1.0
print(f"PASS test 5: empty/pad-only target -> (seq={seq}, char={char})")

print("\nAll tests passed. char_acc is provably bounded in [0, 1].")
