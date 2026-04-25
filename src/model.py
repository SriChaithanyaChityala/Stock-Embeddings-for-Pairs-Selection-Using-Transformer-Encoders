from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class StockTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.embed_head = nn.Linear(d_model, embedding_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        emb = self.embed_head(pooled)
        y_hat = self.pred_head(emb)
        return emb, y_hat


def train_model(
    model: StockTransformerEncoder,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    objective: str = "mse",
    contrastive_weight: float = 0.1,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            emb, y_hat = model(x)
            pred_loss = criterion(y_hat, y)
            loss = pred_loss

            if objective == "mse_contrastive":
                # Weakly supervise embedding similarity from target co-movement.
                emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                sim_mat = emb_norm @ emb_norm.T
                with torch.no_grad():
                    y_vec = y.squeeze(-1)
                    target_sim = 1.0 / (1.0 + (y_vec[:, None] - y_vec[None, :]).abs())
                contrastive_loss = ((sim_mat - target_sim) ** 2).mean()
                loss = pred_loss + contrastive_weight * contrastive_loss

            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}", objective=objective)

        avg_loss = running / max(1, len(train_loader))
        print(f"Epoch {epoch:02d} | train_mse={avg_loss:.6f}")


@torch.no_grad()
def extract_embeddings(
    model: StockTransformerEncoder,
    windows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    windows = windows.to(device)
    emb, _ = model(windows)
    return emb.cpu()
