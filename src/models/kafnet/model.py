import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class RBFKernel(nn.Module):
    """Radial Basis Function kernel for temporal aggregation."""

    def __init__(self, learnable_sigma: bool = True):
        super().__init__()
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_sigma', torch.tensor(0.0))

    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel weights.

        Args:
            time_diff: (batch, n_queries, n_obs) time differences

        Returns:
            weights: (batch, n_queries, n_obs) kernel weights
        """
        sigma = torch.exp(self.log_sigma)
        weights = torch.exp(-0.5 * (time_diff / sigma) ** 2)
        return weights


class KernelAggregation(nn.Module):
    """
    Aggregate observations using kernel functions.

    For each query time t_q, aggregate observations weighted by their
    temporal distance: h(t_q) = Σ_i k(t_q, t_i) * z_i
    """

    def __init__(self, d_model: int, kernel_type: str = 'rbf'):
        super().__init__()
        self.d_model = d_model

        if kernel_type == 'rbf':
            self.kernel = RBFKernel(learnable_sigma=True)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Learnable projection for aggregated features
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        obs_emb: torch.Tensor,
        obs_times: torch.Tensor,
        query_times: torch.Tensor,
        obs_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            obs_emb: (batch, n_obs, d_model) observation embeddings
            obs_times: (batch, n_obs) observation timestamps
            query_times: (batch, n_queries) query timestamps
            obs_mask: (batch, n_obs) binary mask for valid observations

        Returns:
            query_emb: (batch, n_queries, d_model) aggregated embeddings
        """
        batch_size = obs_emb.size(0)
        n_queries = query_times.size(1)
        n_obs = obs_times.size(1)

        # Compute pairwise time differences
        # (B, Q, 1) - (B, 1, O) -> (B, Q, O)
        time_diff = query_times.unsqueeze(2) - obs_times.unsqueeze(1)
        time_diff = torch.abs(time_diff)

        # Compute kernel weights
        weights = self.kernel(time_diff)  # (B, Q, O)

        # Mask out invalid observations
        weights = weights * obs_mask.unsqueeze(1)  # (B, Q, O)

        # Normalize weights
        weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum  # (B, Q, O)

        # Aggregate: (B, Q, O) @ (B, O, D) -> (B, Q, D)
        query_emb = torch.bmm(weights, obs_emb)

        # Project
        query_emb = self.proj(query_emb)

        return query_emb


class FrequencyAttention(nn.Module):
    """
    Attention mechanism in frequency domain.

    Uses FFT to transform to frequency domain, applies attention,
    then transforms back to time domain.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input sequence
            mask: (batch, seq_len) optional mask

        Returns:
            out: (batch, seq_len, d_model) attended sequence
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose: (B, H, L, D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply FFT to queries and keys for frequency-domain attention
        # Note: FFT is applied along sequence dimension
        Q_freq = torch.fft.rfft(Q, dim=2)  # (B, H, L//2+1, D) complex
        K_freq = torch.fft.rfft(K, dim=2)

        # Compute attention scores in frequency domain
        # Using magnitude for attention
        scores = torch.einsum('bhfd,bhfd->bhf', Q_freq.abs(), K_freq.abs())  # (B, H, F)
        scores = scores / math.sqrt(self.head_dim)

        # Softmax over frequencies
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, F)

        # Apply attention to values in frequency domain
        V_freq = torch.fft.rfft(V, dim=2)  # (B, H, F, D)

        # Multiply by attention weights: (B, H, F, 1) * (B, H, F, D)
        attended_freq = attn_weights.unsqueeze(-1) * V_freq  # (B, H, F, D)

        # Transform back to time domain
        attended = torch.fft.irfft(attended_freq, n=seq_len, dim=2)  # (B, H, L, D)

        # Transpose back and reshape
        attended = attended.transpose(1, 2).contiguous()  # (B, L, H, D)
        attended = attended.view(batch_size, seq_len, d_model)  # (B, L, D)

        # Output projection
        out = self.W_o(attended)
        out = self.dropout(out)

        return out


class KAFNetBlock(nn.Module):
    """
    Single KAFNet block: Frequency Attention + FFN.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.freq_attn = FrequencyAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) optional

        Returns:
            out: (batch, seq_len, d_model)
        """
        # Frequency attention with residual
        attended = self.freq_attn(x, mask)
        x = self.norm1(x + self.dropout(attended))

        # FFN with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


class KAFNet(nn.Module):
    """
    KAFNet: Kernel Aggregation + Frequency Attention Network

    Architecture:
    1. Observation encoder: (t, v, z) → d_model
    2. Kernel aggregation: irregular obs → regular grid
    3. Frequency attention blocks
    4. Per-variable decoder
    """

    def __init__(
        self,
        n_variables: int = 36,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_query_points: int = 32,
        kernel_type: str = 'rbf',
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        self.n_query_points = n_query_points

        # Observation encoder: (time, variable, value) → d_model
        self.time_proj = nn.Linear(1, d_model // 4)
        self.var_embedding = nn.Embedding(n_variables, d_model // 4)
        self.value_proj = nn.Linear(1, d_model // 4)
        self.obs_proj = nn.Linear(3 * d_model // 4, d_model)

        # Kernel aggregation
        self.kernel_agg = KernelAggregation(d_model, kernel_type)

        # Frequency attention blocks
        self.blocks = nn.ModuleList([
            KAFNetBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Variable-specific decoders
        self.var_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(n_variables)
        ])

    def encode_observations(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode observations to embeddings.

        Args:
            timestamps: (batch, n_obs)
            variables: (batch, n_obs) integer indices
            values: (batch, n_obs)

        Returns:
            obs_emb: (batch, n_obs, d_model)
        """
        t_emb = self.time_proj(timestamps.unsqueeze(-1))     # (B, O, d/4)
        v_emb = self.var_embedding(variables)                # (B, O, d/4)
        z_emb = self.value_proj(values.unsqueeze(-1))       # (B, O, d/4)

        # Concatenate and project
        obs_emb = torch.cat([t_emb, v_emb, z_emb], dim=-1)  # (B, O, 3d/4)
        obs_emb = self.obs_proj(obs_emb)                    # (B, O, d)

        return obs_emb

    def forward(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, n_obs) observation times [0, 1]
            variables: (batch, n_obs) variable indices
            values: (batch, n_obs) observed values
            mask: (batch, n_obs) binary mask

        Returns:
            predictions: (batch, n_variables) forecast for each variable
        """
        batch_size = timestamps.size(0)

        # 1. Encode observations
        obs_emb = self.encode_observations(timestamps, variables, values)  # (B, O, d)

        # 2. Create regular query grid in forecast window [0.5, 0.625]
        # Using linspace to create regular temporal grid
        query_times = torch.linspace(0.5, 0.625, self.n_query_points, device=timestamps.device)
        query_times = query_times.unsqueeze(0).expand(batch_size, -1)  # (B, Q)

        # 3. Kernel aggregation: map irregular obs to regular grid
        query_emb = self.kernel_agg(obs_emb, timestamps, query_times, mask)  # (B, Q, d)

        # 4. Frequency attention blocks
        for block in self.blocks:
            query_emb = block(query_emb)  # (B, Q, d)

        # 5. Aggregate over time (mean pool) to get single representation
        temporal_repr = query_emb.mean(dim=1)  # (B, d)

        # 6. Decode per variable
        predictions = []
        for v_idx in range(self.n_variables):
            pred_v = self.var_decoders[v_idx](temporal_repr).squeeze(-1)  # (B,)
            predictions.append(pred_v)

        predictions = torch.stack(predictions, dim=1)  # (B, V)

        return predictions


if __name__ == '__main__':
    print("Testing KAFNet model...")

    # Dummy data
    batch_size = 4
    n_obs = 50
    n_variables = 36

    timestamps = torch.rand(batch_size, n_obs)  # [0, 1]
    variables = torch.randint(0, n_variables, (batch_size, n_obs))
    values = torch.randn(batch_size, n_obs)
    mask = torch.rand(batch_size, n_obs) > 0.3  # 70% observed

    # Model
    model = KAFNet(
        n_variables=n_variables,
        d_model=128,
        n_layers=3,
        n_heads=4,
        n_query_points=32
    )

    # Forward pass
    predictions = model(timestamps, variables, values, mask.float())

    print(f"Input shapes:")
    print(f"  Observations: {timestamps.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n✓ KAFNet model works!")
