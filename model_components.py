"""
model_components.py
===================
Importable Python module mirroring the cells of model_components.ipynb.
Both train_model and deploy_model import from this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# ── Type-embedding constants ────────────────────────────────────────────
# Transaction `type` is categorical, not numeric, so it gets its own
# embedding rather than being shoved through the same scaler as the
# continuous columns. These constants stay in sync with FEATURE_COLS
# below — TYPE_COL_IDX is the position of `type_enc` in that list.
TYPE_COL_IDX   = 5   # index of type_enc inside FEATURE_COLS
NUM_TYPE_CATS  = 2   # CASH_OUT and TRANSFER
TYPE_EMB_DIM   = 8   # small embedding — only two categories to represent


# ── Positional Encoding ─────────────────────────────────────────────────
class RelativePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding plus an optional time-step projection.

    The plain sinusoid handles ordering inside the window; the time
    projection lets the model see the *gap* between transactions, which
    matters because mobile-money fraud often comes in rapid bursts.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.time_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.size(1)
        pos_enc = self.pe[:, :seq_len, :]
        if time_steps is not None:
            time_enc = self.time_proj(time_steps)
            return self.dropout(x + pos_enc + time_enc)
        return self.dropout(x + pos_enc)


# ── Causal Self-Attention ───────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    The causal mask stops a transaction from attending to anything that
    happens later in the sequence — important here because we don't want
    future information to leak into a real-time score.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # Project to Q/K/V and reshape to (batch, heads, seq, head_dim).
        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # Scaled dot-product, then mask future positions before softmax.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn    = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(context)


# ── Transformer Encoder Block ───────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm encoder block: causal attention then a small FFN."""
    def __init__(self, d_model: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Main Model ──────────────────────────────────────────────────────────
class LongRangeFraudTransformer(nn.Module):
    """Encoder-only Transformer with two heads.

    The model has two prediction heads sharing the same encoder:

      * `reconstruction_head` — predicts the next transaction's features.
        Large reconstruction error means the next event was unexpected
        given the user's recent history.

      * `anomaly_head` — direct fraud-likelihood logit, trained against
        the `isFraud` label using focal loss.

    Combining the two scores at deploy time gives a more robust
    detector than either head on its own.
    """
    def __init__(
        self,
        num_features:  int,
        d_model:       int   = 128,
        num_layers:    int   = 4,
        num_heads:     int   = 8,
        ffn_dim:       int   = 512,
        context_len:   int   = 60,
        dropout:       float = 0.1,
        num_type_cats: int   = NUM_TYPE_CATS,
        type_emb_dim:  int   = TYPE_EMB_DIM,
        type_col_idx:  int   = TYPE_COL_IDX,
    ):
        super().__init__()
        self.context_len  = context_len
        self.num_features = num_features
        self.d_model      = d_model
        self.type_col_idx = type_col_idx

        # Continuous features (everything except type_enc) plus the
        # learned type embedding form the per-step input vector.
        cont_dim = (num_features - 1) + type_emb_dim

        # Treat transaction type like a word token — cheap and effective.
        self.type_embedding = nn.Embedding(num_type_cats, type_emb_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(cont_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = RelativePositionalEncoding(d_model, max_len=512, dropout=dropout)
        self.layers  = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Reconstruction head — predicts the next step's feature vector.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_features),
        )

        # Anomaly head — emits a raw logit. The training wrapper applies
        # focal loss directly on the logit; deploy applies sigmoid.
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform on linear layers; biases zeroed."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split_features(self, x: torch.Tensor):
        """Pull the type column out of the feature tensor.

        Returns the continuous columns and an integer index tensor that
        the embedding layer can consume.
        """
        idx      = self.type_col_idx
        cont     = torch.cat([x[:, :, :idx], x[:, :, idx + 1:]], dim=-1)
        type_idx = x[:, :, idx].round().long().clamp(0, 1)
        return cont, type_idx

    def forward(
        self,
        past_values:        torch.Tensor,
        past_time_features: torch.Tensor = None,
        future_values:      torch.Tensor = None,
        **kwargs,
    ) -> dict:
        # 1. Separate the type column from the continuous features.
        cont, type_idx = self._split_features(past_values)

        # 2. Look up the type embedding and concat with the continuous side.
        type_emb = self.type_embedding(type_idx)

        # 3. Project to d_model, add positional info, run the encoder stack.
        x = torch.cat([cont, type_emb], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x, past_time_features)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # 4. Two heads, two readouts:
        #    reconstruction uses the last step's hidden state;
        #    anomaly_head pools across the full window (mean pool).
        last_hidden    = x[:, -1, :]
        reconstruction = self.reconstruction_head(last_hidden).unsqueeze(1)
        pooled         = x.mean(dim=1)
        anomaly_logits = self.anomaly_head(pooled)

        output = {
            'reconstruction': reconstruction,
            'anomaly_score':  anomaly_logits,
            'hidden':         x,
        }

        # When future targets are supplied, also compute the auxiliary
        # reconstruction loss plus a small entropy regulariser that
        # discourages the anomaly head from collapsing to one class.
        if future_values is not None:
            recon_loss  = F.mse_loss(reconstruction, future_values)
            probs       = torch.sigmoid(anomaly_logits)
            entropy_reg = -torch.mean(
                probs * torch.log(probs + 1e-9)
                + (1 - probs) * torch.log(1 - probs + 1e-9)
            )
            output['loss'] = recon_loss + 0.01 * entropy_reg

        return output


# ── Feature Engineering ─────────────────────────────────────────────────
def engineer_behavioral_features(df):
    """Add per-user behavioural features to the dataframe.

    These columns capture the things human analysts actually look at when
    deciding whether a transaction is suspicious — pacing, magnitude
    relative to the user's norm, and how much of the wallet is being
    drained at once.
    """
    df = df.sort_values(by=['nameOrig', 'step']).copy()

    # Time gap since the user's previous transaction (0 for their first).
    df['prev_step'] = df.groupby('nameOrig')['step'].shift(1)
    df['time_since_last_txn'] = (df['step'] - df['prev_step']).fillna(0)

    # Running count of transactions for this user (a coarse velocity proxy).
    df['txn_velocity_5'] = (
        df.groupby('nameOrig')['step']
          .transform(lambda x: x.expanding().count())
    )

    # How unusual is this amount for this user? (z-score against their history)
    user_mean = df.groupby('nameOrig')['amount'].transform('mean')
    user_std  = df.groupby('nameOrig')['amount'].transform('std').fillna(1)
    df['amount_zscore'] = (df['amount'] - user_mean) / user_std

    # Fraction of the originating balance being moved (1.0 = full drain).
    df['balance_drain_ratio'] = np.where(
        df['oldbalanceOrg'] > 0, df['amount'] / df['oldbalanceOrg'], 0)

    # Cumulative spend per user — captures slow burns where each individual
    # transaction looks fine but the running total does not.
    df['cumulative_amount'] = df.groupby('nameOrig')['amount'].cumsum()

    df.drop(columns=['prev_step'], inplace=True)
    return df


# ── Feature column list ─────────────────────────────────────────────────
# IMPORTANT: this order is fixed — TYPE_COL_IDX above hard-codes the
# position of `type_enc`. If you reorder, update TYPE_COL_IDX too.
FEATURE_COLS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_enc',       # ← index 5
    'time_since_last_txn', 'txn_velocity_5',
    'amount_zscore', 'balance_drain_ratio', 'cumulative_amount'
]


# ── Sequence Dataset ────────────────────────────────────────────────────
class PaySimSequenceDataset(Dataset):
    """Wraps a PaySim dataframe as fixed-length sliding-window sequences.

    On training data we fit a StandardScaler; on validation/test data we
    re-use that scaler to avoid leaking future statistics into the eval.

    `mask_prob` only takes effect during training — it randomly zeroes
    out a fraction of the past steps, encouraging the model to fall back
    on broader context instead of memorising specific positions.
    """
    def __init__(self, data, scaler=None, context_len=20, train=True,
                 feature_cols=None, stride=10, mask_prob: float = 0.15):
        self.context_len  = context_len
        self.pred_len     = 1
        self.feature_cols = feature_cols or FEATURE_COLS
        self.stride       = stride
        self.train        = train
        self.mask_prob    = mask_prob

        # Fit the scaler on train data; transform-only on val/test.
        if train:
            self.scaler = StandardScaler()
            data = data.copy()
            data[self.feature_cols] = self.scaler.fit_transform(data[self.feature_cols])
        else:
            assert scaler is not None, "scaler required when train=False"
            self.scaler = scaler
            data = data.copy()
            data[self.feature_cols] = self.scaler.transform(data[self.feature_cols])

        self.sequences  = []
        self.targets    = []
        self.time_steps = []

        # Slide a window of size context_len + 1 over the time-ordered data.
        # We always keep fraud windows; non-fraud windows are sub-sampled
        # by `stride` to keep the dataset manageable at training time.
        window = self.context_len + self.pred_len
        data   = data.sort_values('step').reset_index(drop=True)
        values = data[self.feature_cols].values.astype('float32')
        labels = data['isFraud'].values if 'isFraud' in data.columns \
                 else np.zeros(len(data), dtype=int)
        steps  = data['step'].values.astype('float32')

        for start in range(0, len(values) - window + 1, self.stride):
            end      = start + window
            is_fraud = int(labels[end - 1])
            if is_fraud == 1 or start % self.stride == 0:
                self.sequences.append(values[start:end])
                self.targets.append(is_fraud)
                self.time_steps.append(steps[start:end])

        mask_info = f"  mask_prob={mask_prob}" if train else ""
        print(f"  Dataset: {len(self.sequences):,} sequences "
              f"(stride={stride}, context={context_len}, "
              f"fraud={sum(self.targets):,}){mask_info}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx].copy()
        steps = self.time_steps[idx]

        past_values   = torch.tensor(seq[:self.context_len])
        future_values = torch.tensor(seq[self.context_len:])

        # Random per-step masking during training (BERT-style).
        if self.train and self.mask_prob > 0:
            mask = torch.bernoulli(
                torch.full((self.context_len,), self.mask_prob)
            ).bool()
            past_values = past_values.masked_fill(mask.unsqueeze(-1), 0.0)

        # Normalise the time-step features into [0, 1] so they're on the
        # same scale as the projected positional encodings.
        ps = steps[:self.context_len]
        fs = steps[self.context_len:]
        past_time   = torch.tensor(ps / (ps.max() + 1e-9),
                                   dtype=torch.float32).unsqueeze(-1)
        future_time = torch.tensor(fs / (fs.max() + 1e-9),
                                   dtype=torch.float32).unsqueeze(-1)

        return {
            "past_values":            past_values,
            "past_time_features":     past_time,
            "past_observed_mask":     torch.ones(self.context_len, past_values.shape[1]),
            "future_values":          future_values,
            "future_time_features":   future_time,
            "labels":                 torch.tensor(self.targets[idx]),
        }


print("✓ Model components defined (embeddings + logits + sequence masking).")
