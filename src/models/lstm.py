"""LSTM forecasting model."""

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

    class LSTMNet(nn.Module):
        def __init__(self, input_size=5, hidden_size=32, num_layers=1, num_classes=3, dropout=0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # last time step
            out = self.dropout(out)
            return self.fc(out)
else:
    LSTMNet = None  # type: ignore


class LSTMModel(BaseModel):
    """LSTM-based market direction predictor."""

    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        threshold: float = 0.3,
        version: str = "1.0.0",
    ):
        super().__init__(name="lstm", version=version)
        self.model_type = "lstm"
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.threshold = threshold
        self.input_size = 5  # OHLCV
        self.num_classes = 3
        self.label_map = {0: "down", 1: "neutral", 2: "up"}
        self.rev_label = {v: k for k, v in self.label_map.items()}
        self._build_model()

    def _build_model(self):
        if HAS_TORCH:
            self.net = LSTMNet(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout,
            )
        else:
            self.net = None

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "threshold": self.threshold,
        }

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        # window: (seq_len, 5) -> z-score per feature using last close as base for price
        # Simple per-window standardization to keep inference stable without global stats
        closes = window[:, 3]
        mean = float(np.mean(closes)) if len(closes) else 1.0
        std = float(np.std(closes)) if len(closes) else 1.0
        if std < 1e-8:
            std = 1.0
        # Normalize OHLC relative to mean/std of closes, volume relative to its own mean
        vols = window[:, 4]
        v_mean = float(np.mean(vols)) if len(vols) else 1.0
        v_std = float(np.std(vols)) if len(vols) else 1.0
        if v_std < 1e-8:
            v_std = 1.0
        norm = np.zeros_like(window, dtype=np.float32)
        for i in range(4):  # OHLC
            norm[:, i] = (window[:, i] - mean) / std
        norm[:, 4] = (window[:, 4] - v_mean) / (v_std + 1e-8)
        return norm

    def prepare_features(self, candles: np.ndarray) -> Dict[str, Any]:
        base = super().prepare_features(candles)
        if len(candles) == 0:
            return base
        # Add sequence for LSTM
        seq = candles[-self.seq_len :] if len(candles) >= self.seq_len else candles
        # pad if needed
        if len(seq) < self.seq_len:
            pad_len = self.seq_len - len(seq)
            pad = np.tile(seq[0:1], (pad_len, 1))
            seq = np.vstack([pad, seq])
        norm_seq = self._normalize_window(seq.astype(np.float32))
        base["sequence"] = norm_seq  # (seq_len, 5)
        base["seq_len"] = int(self.seq_len)
        # also store raw last window for fallback
        base["raw_sequence"] = seq.tolist()
        return base

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float]:
        seq = features.get("sequence")
        if HAS_TORCH and self.net is not None and seq is not None:
            try:
                self.net.eval()
                with torch.no_grad():
                    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)
                    logits = self.net(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    label = self.label_map[idx]
                    # low confidence -> neutral
                    if conf < 0.40:
                        return "neutral", float(conf)
                    return label, float(conf)
            except Exception:
                pass
        # Fallback heuristic: momentum on last closes
        price_change_pct = features.get("price_change_pct")
        if price_change_pct is None:
            return "neutral", 0.5
        if abs(price_change_pct) < self.threshold:
            return "neutral", 0.5
        conf = min(0.55 + abs(price_change_pct) * 0.08, 0.9)
        return ("up" if price_change_pct > 0 else "down"), float(conf)

    def fit(self, candles: np.ndarray, epochs: int = 15, lr: float = 1e-3, batch_size: int = 32, patience: int = 3, seed: int = 42) -> Dict[str, Any]:
        """Train with configurable epochs, early-stop, GPU, determinism and metrics."""
        if not HAS_TORCH:
            return {"ok": False, "reason": "torch not installed"}
        if len(candles) < self.seq_len + 1:
            return {"ok": False, "reason": f"need >= {self.seq_len+1} candles, got {len(candles)}"}
        # determinism
        try:
            import random
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        # Build dataset
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
            X.append(norm); y.append(label)
        X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int64)
        n = len(X)
        split = int(n*0.85)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:] if split < n else (X[:1], y[:1])
        train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        best_val = float("inf"); wait = 0; best_state = None; last_loss = 0.0
        for epoch in range(int(epochs)):
            self.net.train()
            total = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(self.net(xb), yb); loss.backward(); opt.step()
                total += loss.item()*xb.size(0)
            last_loss = total/len(train_ds)
            # validation
            self.net.eval(); vloss=0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vloss += crit(self.net(xb), yb).item()*xb.size(0)
            vloss = vloss/len(val_ds) if len(val_ds) else last_loss
            if vloss < best_val - 1e-4:
                best_val = vloss; wait = 0; best_state = {k: v.cpu().clone() for k,v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to("cpu")
        # metrics on val
        from collections import Counter
        self.net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                logits = self.net(xb)
                preds.extend(torch.argmax(logits,1).tolist())
        # precision/recall/F1 per class
        metrics = {}
        for idx, name in self.label_map.items():
            tp = sum(1 for p,t in zip(preds, y_val) if p==idx and t==idx)
            fp = sum(1 for p,t in zip(preds, y_val) if p==idx and t!=idx)
            fn = sum(1 for p,t in zip(preds, y_val) if p!=idx and t==idx)
            prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0; f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
            metrics[name] = {"precision": prec, "recall": rec, "f1": f1}
        acc = sum(1 for p,t in zip(preds, y_val) if p==t)/len(y_val) if len(y_val) else 0
        # Sharpe proxy: mean return / std of predicted direction returns (simplified)
        sharpe = 0.0
        try:
            rets = []
            for i, p in enumerate(preds):
                # map pred to signed return: up=+1, down=-1, neutral=0 times val pct
                idx = split + i
                if idx+1 < len(candles):
                    pct = (float(candles[idx+1,3])-float(candles[idx,3]))/float(candles[idx,3])
                    sign = 1 if p==self.rev_label["up"] else -1 if p==self.rev_label["down"] else 0
                    rets.append(sign*pct)
            if rets:
                sharpe = float(np.mean(rets) / (np.std(rets)+1e-9) * (252**0.5))
        except Exception:
            pass
        return {"ok": True, "loss": float(last_loss), "val_loss": float(best_val), "samples": int(n), "epochs": int(epochs), "early_stopped": wait>=patience, "accuracy": acc, "per_class": metrics, "sharpe": sharpe, "device": str(device)}
