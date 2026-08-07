from .layers import Linear, Embedding, MultiHeadSelfAttention, RMSNorm, SwiGLU, softmax
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            device,
            dtype,
            theta,
            max_seq_len,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
        ) -> Float[Tensor, " batch sequence_length d_model"]:
        x = in_features + self.attn(self.ln1(in_features))

        return x + self.ffn(self.ln2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device,
            dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            device,
            dtype,
        )

    def forward(
        self,
        in_indices: Int[Tensor, "batch sequence_length"],
        )-> Float[Tensor, "batch sequence_length vocab_size"]:
    
        if in_indices.ndim != 2:
            raise ValueError(
                f"Expected token IDs with shape [batch, sequence], "
                f"got {in_indices.shape}"
            )

        if in_indices.shape[-1] > self.context_length:
            raise ValueError(
                f"Input sequence length {in_indices.shape[-1]} exceeds "
                f"context length {self.context_length}."
            )

        if in_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"Token IDs must be int32 or int64, but got {in_indices.dtype}"
            )

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

