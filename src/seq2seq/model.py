import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .config import config

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = config['cnn_channels']
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=config['cnn_dropout'])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=config['cnn_dropout'])
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=config['cnn_dropout'])
        )
        
    def forward(self, x):
        # x: [B, 1, 64, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # x: [B, 128, 8, T] where T = W/8
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2)       # [B, T, 128, 8]
        x = x.reshape(B, T, C * H)      # [B, T, 1024]
        return x

class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Relative positional encoding: scalar position in [0,1] -> feature dim
        # Allows the encoder to generalise to image widths unseen during training
        self.pos_proj = nn.Linear(1, 1024)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=config['hidden_size'],
                            bidirectional=True, batch_first=True)
        # linear layers to project from 512 (256*2) to 256
        self.hidden_proj = nn.Linear(config['hidden_size'] * 2, config['hidden_size'])
        self.cell_proj   = nn.Linear(config['hidden_size'] * 2, config['hidden_size'])

    def forward(self, x):
        # x: [B, T, 1024]
        T = x.size(1)
        # Relative positions in [0, 1] regardless of actual sequence length
        positions = torch.linspace(0, 1, T, device=x.device)   # [T]
        pos_emb   = self.pos_proj(positions.unsqueeze(-1))      # [T, 1024]
        x = x + pos_emb.unsqueeze(0)                            # broadcast over B

        encoder_outputs, (h, c) = self.lstm(x)
        # h, c are [2, B, 256] -> need to concat directions to get [B, 512] then project

        # Concat fwd and bwd states
        h_concat = torch.cat((h[0], h[1]), dim=1)  # [B, 512]
        c_concat = torch.cat((c[0], c[1]), dim=1)  # [B, 512]

        hidden = torch.tanh(self.hidden_proj(h_concat))  # [B, 256]
        cell   = torch.tanh(self.cell_proj(c_concat))    # [B, 256]

        return encoder_outputs, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(config['hidden_size'], config['attention_dim'], bias=False)
        self.U = nn.Linear(config['hidden_size'] * 2, config['attention_dim'], bias=False)
        self.v = nn.Linear(config['attention_dim'], 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [B, 256] -> need [B, 1, 256] to broadcast with encoder_outputs [B, T, 512]
        hidden_expanded = hidden.unsqueeze(1)
        
        # energy: [B, T, attention_dim]
        energy = torch.tanh(self.W(hidden_expanded) + self.U(encoder_outputs))
        
        # scores: [B, T, 1] -> [B, T]
        scores = self.v(energy).squeeze(2)
        
        # alpha: [B, T]
        alpha = F.softmax(scores, dim=1)
        
        # context: [B, 1, T] @ [B, T, 512] -> [B, 1, 512] -> [B, 512]
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, alpha

class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], 
                                      embedding_dim=config['embed_dim'], 
                                      padding_idx=config['PAD_IDX'])
        
        self.attention = BahdanauAttention()
        
        self.lstm_cell = nn.LSTMCell(input_size=config['embed_dim'] + config['hidden_size'] * 2, 
                                     hidden_size=config['hidden_size'])
        
        self.dropout = nn.Dropout(p=config['dec_dropout'])
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'])
        
    def forward_step(self, prev_token, hidden, cell, encoder_outputs):
        # prev_token: [B]
        embed = self.embedding(prev_token) # [B, embed_dim]
        embed = self.dropout(embed)
        
        context, alpha = self.attention(hidden, encoder_outputs) # context: [B, 512], alpha: [B, T]
        
        rnn_input = torch.cat((embed, context), dim=1) # [B, embed_dim + 512]
        
        hidden, cell = self.lstm_cell(rnn_input, (hidden, cell))
        
        logits = self.fc(self.dropout(hidden)) # [B, vocab_size]
        
        return logits, hidden, cell, alpha

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_cnn = CNNEncoder()
        self.encoder_rnn = BiLSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, images, targets=None, teacher_forcing_ratio=None):
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = config['teacher_forcing_ratio'] if self.training else 0.0

        B = images.size(0)
        device = images.device

        cnn_features = self.encoder_cnn(images)
        encoder_outputs, hidden, cell = self.encoder_rnn(cnn_features)

        if targets is not None:
            # ── Training path ─────────────────────────────────────────────
            # Decode for exactly (target_len - 1) steps (SOS is input, not output)
            seq_len = targets.size(1) - 1
            T_enc = encoder_outputs.size(1)

            logits_seq = torch.zeros(B, seq_len, config['vocab_size'], device=device)
            alphas_seq = torch.zeros(B, seq_len, T_enc, device=device)

            prev_token = targets[:, 0]  # SOS

            for t in range(seq_len):
                logits, hidden, cell, alpha = self.decoder.forward_step(
                    prev_token, hidden, cell, encoder_outputs
                )
                logits_seq[:, t, :] = logits
                alphas_seq[:, t, :] = alpha

                teacher_force = random.random() < teacher_forcing_ratio
                prev_token = targets[:, t + 1] if teacher_force else logits.argmax(1)

        else:
            # ── Inference path ────────────────────────────────────────────
            # No length cap — run until every sequence in the batch emits EOS.
            # Output length is determined entirely by the model.
            T_enc = encoder_outputs.size(1)

            logits_list = []
            alphas_list = []

            prev_token = torch.full((B,), config['SOS_IDX'], dtype=torch.long, device=device)
            finished   = torch.zeros(B, dtype=torch.bool, device=device)

            while not finished.all():
                logits, hidden, cell, alpha = self.decoder.forward_step(
                    prev_token, hidden, cell, encoder_outputs
                )
                logits_list.append(logits.unsqueeze(1))   # [B, 1, vocab_size]
                alphas_list.append(alpha.unsqueeze(1))    # [B, 1, T_enc]

                prev_token  = logits.argmax(1)
                finished   |= (prev_token == config['EOS_IDX'])

            logits_seq = torch.cat(logits_list, dim=1)   # [B, L, vocab_size]
            alphas_seq = torch.cat(alphas_list, dim=1)   # [B, L, T_enc]

        return logits_seq, alphas_seq

if __name__ == '__main__':
    import torch
    model = Seq2Seq()

    dummy_img    = torch.zeros(2, 1, 64, 320)       # batch=2, width=320
    dummy_target = torch.randint(0, 13, (2, 9))     # batch=2, seq_len=9 (includes SOS)

    # --- Training forward (with targets) ---
    model.train()
    logits, alphas = model(dummy_img, dummy_target)
    print(f"[train] Logits shape : {logits.shape}")   # expect [2, 8, 13]
    print(f"[train] Alphas shape : {alphas.shape}")   # expect [2, 8, T_enc]

    # --- Inference forward (no targets, no length cap — stops on EOS) ---
    model.eval()
    with torch.no_grad():
        logits_inf, alphas_inf = model(dummy_img, targets=None, teacher_forcing_ratio=0.0)
    print(f"[infer] Logits shape : {logits_inf.shape}")  # [2, L, 13] where L is EOS-determined
    print(f"[infer] Alphas shape : {alphas_inf.shape}")
    print("Forward pass OK")
