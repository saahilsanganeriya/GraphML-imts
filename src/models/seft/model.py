import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings like in Transformer."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) or (batch, seq_len, 1)

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=timestamps.device) *
            (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(timestamps.size(0), timestamps.size(1), self.d_model, device=timestamps.device)
        pe[:, :, 0::2] = torch.sin(timestamps * div_term)
        pe[:, :, 1::2] = torch.cos(timestamps * div_term)

        return pe


class ObservationEncoder(nn.Module):
    """Encode (time, variable, value) → d_model."""

    def __init__(self, n_variables: int, d_model: int):
        super().__init__()
        self.time_embedding = TimeEmbedding(d_model // 2)
        self.var_embedding = nn.Embedding(n_variables, d_model // 4)
        self.value_proj = nn.Linear(1, d_model // 4)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len)
            variables: (batch, seq_len) integer indices
            values: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        time_emb = self.time_embedding(timestamps)        # (B, L, d/2)
        var_emb = self.var_embedding(variables)           # (B, L, d/4)
        val_emb = self.value_proj(values.unsqueeze(-1))  # (B, L, d/4)

        # Concatenate
        obs_emb = torch.cat([time_emb, var_emb, val_emb], dim=-1)  # (B, L, d)

        # Project to d_model
        return self.output_proj(obs_emb)


class SetEncoder(nn.Module):
    """Permutation-invariant set encoder using multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) binary mask (1=valid, 0=padding)

        Returns:
            global_emb: (batch, d_model) aggregated set representation
        """
        # Create attention mask (True = ignore)
        attn_mask = (mask == 0)  # (B, L)

        # Transformer encoder
        encoded = self.encoder(x, src_key_padding_mask=attn_mask)  # (B, L, d)

        # Aggregate: mean pooling over valid observations
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        sum_emb = (encoded * mask_expanded).sum(dim=1)  # (B, d)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        global_emb = sum_emb / count  # (B, d)

        return global_emb


class QueryDecoder(nn.Module):
    """Decode queries (t_future, v_target) to predicted values."""

    def __init__(self, n_variables: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.time_embedding = TimeEmbedding(d_model // 2)
        self.var_embedding = nn.Embedding(n_variables, d_model // 2)
        self.query_proj = nn.Linear(d_model, d_model)

        # Cross-attention: query attends to global embedding
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output head
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        query_times: torch.Tensor,
        query_vars: torch.Tensor,
        global_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_times: (batch, n_queries)
            query_vars: (batch, n_queries) variable indices
            global_emb: (batch, d_model) aggregated history

        Returns:
            predictions: (batch, n_queries)
        """
        # Encode queries
        time_emb = self.time_embedding(query_times)           # (B, Q, d/2)
        var_emb = self.var_embedding(query_vars)              # (B, Q, d/2)
        query_emb = torch.cat([time_emb, var_emb], dim=-1)   # (B, Q, d)
        query_emb = self.query_proj(query_emb)

        # Cross-attention: queries attend to global embedding
        global_emb_expanded = global_emb.unsqueeze(1)  # (B, 1, d)
        attended, _ = self.cross_attn(
            query=query_emb,
            key=global_emb_expanded,
            value=global_emb_expanded
        )  # (B, Q, d)

        # Output
        predictions = self.output(attended).squeeze(-1)  # (B, Q)

        return predictions


class SeFT(nn.Module):
    """
    SeFT: Set Functions for Time Series

    Main model combining encoder, set aggregation, and decoder.
    """

    def __init__(
        self,
        n_variables: int = 36,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = ObservationEncoder(n_variables, d_model)
        self.set_encoder = SetEncoder(d_model, n_heads, n_layers, dropout)
        self.decoder = QueryDecoder(n_variables, d_model, n_heads, dropout)

    def forward(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        query_times: torch.Tensor,
        query_vars: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) observation times
            variables: (batch, seq_len) variable indices
            values: (batch, seq_len) observed values
            mask: (batch, seq_len) binary mask
            query_times: (batch, n_queries) forecast times
            query_vars: (batch, n_queries) forecast variable indices

        Returns:
            predictions: (batch, n_queries) predicted values
        """
        # Encode observations
        obs_emb = self.encoder(timestamps, variables, values)  # (B, L, d)

        # Aggregate to set representation
        global_emb = self.set_encoder(obs_emb, mask)  # (B, d)

        # Decode queries
        predictions = self.decoder(query_times, query_vars, global_emb)  # (B, Q)

        return predictions


if __name__ == '__main__':
    print("Testing SeFT model...")

    # Dummy data
    batch_size = 4
    seq_len = 50
    n_queries = 10
    n_variables = 36

    timestamps = torch.rand(batch_size, seq_len)  # [0,1]
    variables = torch.randint(0, n_variables, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    mask = torch.rand(batch_size, seq_len) > 0.3  # 70% observed

    query_times = torch.rand(batch_size, n_queries)  # Future times
    query_vars = torch.randint(0, n_variables, (batch_size, n_queries))

    # Model
    model = SeFT(n_variables=n_variables, d_model=128, n_heads=4, n_layers=2)

    # Forward pass
    predictions = model(timestamps, variables, values, mask.float(), query_times, query_vars)

    print(f"Input shapes:")
    print(f"  Observations: {timestamps.shape}")
    print(f"  Queries: {query_times.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✓ SeFT model works!")
