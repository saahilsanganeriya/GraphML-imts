import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionAdjacency(nn.Module):
    """
    Learn graph adjacency matrix using attention.

    Computes edge weights α_ij = attention(h_i, h_j)
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        temporal_dist: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, n_nodes, d_model) node features
            temporal_dist: (batch, n_nodes, n_nodes) pairwise time differences
            top_k: Keep only top-k neighbors per node (sparsification)

        Returns:
            adjacency: (batch, n_nodes, n_nodes) attention weights
            messages: (batch, n_nodes, d_model) aggregated messages
        """
        batch_size, n_nodes, d_model = h.shape

        # Multi-head projections
        Q = self.W_q(h).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        K = self.W_k(h).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        V = self.W_v(h).view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, n_nodes, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores: (batch, n_heads, n_nodes, n_nodes)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add temporal bias (penalize distant pairs)
        if temporal_dist is not None:
            temporal_bias = -0.1 * temporal_dist.unsqueeze(1)  # (B, 1, N, N)
            scores = scores + temporal_bias

        # Top-k sparsification (optional)
        if top_k is not None:
            # Keep only top-k highest scores per node
            topk_vals, topk_idx = torch.topk(scores, k=min(top_k, n_nodes), dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn = F.softmax(scores, dim=-1)  # (B, H, N, N)

        # Replace NaN with 0 (happens when all scores are -inf)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        # Aggregate messages
        messages = torch.matmul(attn, V)  # (B, H, N, head_dim)
        messages = messages.transpose(1, 2).contiguous()  # (B, N, H, head_dim)
        messages = messages.view(batch_size, n_nodes, d_model)  # (B, N, d_model)
        messages = self.W_o(messages)

        # Average adjacency across heads for visualization
        adjacency = attn.mean(dim=1)  # (B, N, N)

        return adjacency, messages


class GraphConvLayer(nn.Module):
    """Graph convolution with learned adjacency."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = AttentionAdjacency(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        temporal_dist: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, n_nodes, d_model)
            temporal_dist: (batch, n_nodes, n_nodes)
            top_k: Sparsification parameter

        Returns:
            h_out: (batch, n_nodes, d_model) updated features
            adjacency: (batch, n_nodes, n_nodes) learned graph
        """
        # Self-attention with learned adjacency
        adjacency, messages = self.attention(h, temporal_dist, top_k)

        # Residual + norm
        h = self.norm1(h + self.dropout(messages))

        # FFN
        h = self.norm2(h + self.dropout(self.ffn(h)))

        return h, adjacency


class LearnableGraphModel(nn.Module):
    """
    Main model: Variable-level graphs with learned adjacency.

    Architecture:
    1. Encode observations → node embeddings
    2. Aggregate per variable → V variable embeddings
    3. Learn variable-variable graph via attention
    4. GNN message passing
    5. Decode to forecasts
    """

    def __init__(
        self,
        n_variables: int = 36,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        top_k: Optional[int] = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        self.top_k = top_k

        # Node encoder: (t, v, z) → d_model
        self.time_proj = nn.Linear(1, d_model // 4)
        self.var_embedding = nn.Embedding(n_variables, d_model // 4)
        self.value_proj = nn.Linear(1, d_model // 4)
        self.node_proj = nn.Linear(3 * d_model // 4, d_model)

        # Variable aggregation (per-variable attention pooling)
        self.var_attn = nn.Linear(d_model, 1)

        # Graph convolution layers
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Decoder: variable embedding → forecast
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def encode_observations(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode observations to node embeddings.

        Args:
            timestamps: (batch, seq_len)
            variables: (batch, seq_len)
            values: (batch, seq_len)

        Returns:
            node_emb: (batch, seq_len, d_model)
        """
        t_emb = self.time_proj(timestamps.unsqueeze(-1))       # (B, L, d/4)
        v_emb = self.var_embedding(variables)                  # (B, L, d/4)
        z_emb = self.value_proj(values.unsqueeze(-1))         # (B, L, d/4)

        # Concatenate and project
        node_emb = torch.cat([t_emb, v_emb, z_emb], dim=-1)   # (B, L, 3d/4)
        node_emb = self.node_proj(node_emb)                   # (B, L, d)

        return node_emb

    def aggregate_per_variable(
        self,
        node_emb: torch.Tensor,
        variables: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate observations per variable using attention pooling.

        Args:
            node_emb: (batch, seq_len, d_model)
            variables: (batch, seq_len) variable indices
            mask: (batch, seq_len) binary mask

        Returns:
            var_emb: (batch, n_variables, d_model)
        """
        batch_size = node_emb.size(0)
        var_emb = torch.zeros(batch_size, self.n_variables, self.d_model, device=node_emb.device)

        for v in range(self.n_variables):
            # Find observations for variable v
            v_mask = (variables == v) & (mask > 0)  # (B, L)

            if v_mask.any():
                # Extract variable's observations
                v_nodes = node_emb * v_mask.unsqueeze(-1).float()  # (B, L, d)

                # Attention pooling
                attn_scores = self.var_attn(v_nodes).squeeze(-1)  # (B, L)
                attn_scores = attn_scores.masked_fill(~v_mask, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, L, 1)

                # Replace NaN with uniform weights
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

                # Weighted sum
                var_emb[:, v, :] = (v_nodes * attn_weights).sum(dim=1)  # (B, d)

        return var_emb

    def forward(
        self,
        timestamps: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        target_vars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timestamps: (batch, seq_len)
            variables: (batch, seq_len) variable indices
            values: (batch, seq_len)
            mask: (batch, seq_len)
            target_vars: (batch, n_targets) which variables to predict (optional)

        Returns:
            predictions: (batch, n_variables) or (batch, n_targets)
            adjacency: (batch, n_variables, n_variables) learned graph
        """
        # 1. Encode observations
        node_emb = self.encode_observations(timestamps, variables, values)

        # 2. Aggregate per variable
        var_emb = self.aggregate_per_variable(node_emb, variables, mask)  # (B, V, d)

        # 3. Learn variable-variable graph and propagate
        adjacencies = []
        for layer in self.graph_layers:
            var_emb, adj = layer(var_emb, temporal_dist=None, top_k=self.top_k)
            adjacencies.append(adj)

        # Final adjacency (average over layers)
        adjacency = torch.stack(adjacencies).mean(dim=0)  # (B, V, V)

        # 4. Decode
        predictions = self.decoder(var_emb).squeeze(-1)  # (B, V)

        # Select target variables if specified
        if target_vars is not None:
            batch_indices = torch.arange(predictions.size(0)).unsqueeze(1).expand_as(target_vars)
            predictions = predictions[batch_indices, target_vars]  # (B, n_targets)

        return predictions, adjacency


if __name__ == '__main__':
    print("Testing Learnable Graph Model...")

    # Dummy data
    batch_size = 4
    seq_len = 50
    n_variables = 36

    timestamps = torch.rand(batch_size, seq_len)
    variables = torch.randint(0, n_variables, (batch_size, seq_len))
    values = torch.randn(batch_size, seq_len)
    mask = torch.rand(batch_size, seq_len) > 0.3

    # Model
    model = LearnableGraphModel(
        n_variables=n_variables,
        d_model=128,
        n_layers=2,
        n_heads=4,
        top_k=12
    )

    # Forward
    predictions, adjacency = model(timestamps, variables, values, mask.float())

    print(f"Predictions shape: {predictions.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check adjacency properties
    print(f"\nAdjacency stats:")
    print(f"  Min: {adjacency.min():.4f}")
    print(f"  Max: {adjacency.max():.4f}")
    print(f"  Sparsity: {(adjacency < 0.01).float().mean():.1%}")

    print("\n✓ Learnable Graph model works!")
