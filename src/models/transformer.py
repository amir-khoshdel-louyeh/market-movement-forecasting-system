"""Transformer forecasting model."""

import math
import numpy as np
from typing import Dict, Any, Tuple

from .base import BaseModel

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


if HAS_TORCH:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            return x + self.pe[:, : x.size(1), :]

    class TransformerNet(nn.Module):
        def __init__(self, input_size=5, d_model=32, nhead=4, num_layers=1, num_classes=3, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            x = x[:, -1, :]  # last token
            x = self.dropout(x)
            return self.fc(x)
else:
    TransformerNet = None  # type: ignore


class TransformerModel(BaseModel):
    """Transformer-based market direction predictor."""

    def __init__(
        self,
        seq_len: int = 30,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        threshold: float = 0.3,
        version: str = "1.0.0",
    ):
        super().__init__(name="transformer", version=version)
        self.model_type = "transformer"
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.threshold = threshold
        self.input_size = 5
        self.num_classes = 3
        self.label_map = {0: "down", 1: "neutral", 2: "up"}
        self.rev_label = {v: k for k, v in self.label_map.items()}
        self._build_model()

    def _build_model(self):
        if HAS_TORCH:
            # ensure nhead divides d_model
            if self.d_model % self.nhead != 0:
                self.nhead = 2 if self.d_model % 2 == 0 else 1
            self.net = TransformerNet(
                input_size=self.input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout,
            )
        else:
            self.net = None

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "threshold": self.threshold,
        }

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        closes = window[:, 3]
        mean = float(np.mean(closes)) if len(closes) else 1.0
        std = float(np.std(closes)) if len(closes) else 1.0
        if std < 1e-8:
            std = 1.0
        vols = window[:, 4]
        v_mean = float(np.mean(vols)) if len(vols) else 1.0
        v_std = float(np.std(vols)) if len(vols) else 1.0
        if v_std < 1e-8:
            v_std = 1.0
        norm = np.zeros_like(window, dtype=np.float32)
        for i in range(4):
            norm[:, i] = (window[:, i] - mean) / std
        norm[:, 4] = (window[:, 4] - v_mean) / (v_std + 1e-8)
        return norm

    def prepare_features(self, candles: np.ndarray) -> Dict[str, Any]:
        base = super().prepare_features(candles)
        if len(candles) == 0:
            return base
        seq = candles[-self.seq_len :] if len(candles) >= self.seq_len else candles
        if len(seq) < self.seq_len:
            pad_len = self.seq_len - len(seq)
            pad = np.tile(seq[0:1], (pad_len, 1))
            seq = np.vstack([pad, seq])
        norm_seq = self._normalize_window(seq.astype(np.float32))
        base["sequence"] = norm_seq
        base["seq_len"] = int(self.seq_len)
        base["raw_sequence"] = seq.tolist()
        # attention-friendly extras
        if len(candles) >= 2:
            closes = candles[:, 3]
            base["recent_volatility"] = float(np.std(np.diff(closes) / closes[:-1])) if len(closes) > 1 else 0.0
        return base

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        seq = features.get("sequence")
        if HAS_TORCH and self.net is not None and seq is not None:
            try:
                self.net.eval()
                with torch.no_grad():
                    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                    logits = self.net(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    label = self.label_map[idx]
                    if conf < 0.40:
                        return "neutral", float(conf)
                    return label, float(conf)
            except Exception:
                pass
        # fallback: volume-weighted momentum
        price_change_pct = features.get("price_change_pct")
        vol = features.get("recent_volatility")
        if price_change_pct is None:
            return "neutral", 0.5
        if abs(price_change_pct) < self.threshold:
            return "neutral", 0.5
        boost = min((vol or 0) * 10, 0.15) if vol else 0
        conf = min(0.55 + abs(price_change_pct) * 0.08 + boost, 0.92)
        return ("up" if price_change_pct > 0 else "down"), float(conf)

    def fit(self, candles: np.ndarray, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32) -> Dict[str, Any]:
        if not HAS_TORCH:
            return {"ok": False, "reason": "torch not installed"}
        if len(candles) < self.seq_len + 1:
            return {"ok": False, "reason": f"need >= {self.seq_len+1} candles, got {len(candles)}"}
        X, y = [], []
        for i in range(len(candles) - self.seq_len):
            window = candles[i : i + self.seq_len]
            next_close = float(candles[i + self.seq_len, 3])
            curr_close = float(candles[i + self.seq_len - 1, 3])
            pct = (next_close - curr_close) / curr_close * 100 if curr_close != 0 else 0
            if pct > self.threshold:
                label = self.rev_label["up"]
            elif pct < -self.threshold:
                label = self.rev_label["down"]
            else:
                label = self.rev_label["neutral"]
            norm = self._normalize_window(window.astype(np.float32))
            X.append(norm)
            y.append(label)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        self.net.train()
        last_loss = 0.0
        for epoch in range(epochs):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.net(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            last_loss = total / len(dataset)
        return {"ok": True, "loss": float(last_loss), "samples": int(len(X)), "epochs": int(epochs)}
