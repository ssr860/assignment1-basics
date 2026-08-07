from __future__ import annotations
from typing import Any

import math
import torch
from torch import nn

from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, 
                        in_features,
                        dtype = dtype,
                        device = device
                    )
                )

        std = math.sqrt(2/(self.in_features+self.out_features))

        nn.init.trunc_normal_(self.weight,
                                    mean = 0,
                                    std = std,
                                    a = -3*std,
                                    b = 3*std,)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return einsum(
            x, self.weight,
            "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device = device,
                dtype = dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1.0,
            a = -3.0,
            b = 3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]    


class RMSNorm(nn.Module):
    def __init__(self,
                d_model: int,
                eps: float = 1e-5,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
                    torch.ones(
                        self.d_model,
                        device=device,
                        dtype=dtype,
                    )
                )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(
            self.eps + 
            torch.mean(x**2, dim = -1, keepdim=True)
        )

        result = x * self.weight / rms_x
        return result.to(in_type)


class SwiGLU(nn.Module):
    def __init__(self,
                d_model: int,
                d_ff: int,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)

    def forward(self, x):
        s_x = self.w1(x)
        gate = torch.sigmoid(s_x)*s_x
        ffn_x = self.w2(gate*self.w3(x))
        return ffn_x


class RoPE(nn.Module):
    def __init__(self, 
                theta: float, 
                d_k: int, 
                max_seq_len: int, 
                device: torch.device | None = None) -> None:
        super().__init__()  
        assert d_k %2 == 0

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device  

        k = - torch.arange(0, d_k, 2, device = device)/d_k
        k = theta**(k)
        p = torch.arange(0, max_seq_len, device = device, dtype = torch.float32)
        m = torch.outer(p, k)

        self.cos = torch.cos(m) 
        self.sin = torch.sin(m)
        self.register_buffer("cos_cache", self.cos, persistent = False )
        self.register_buffer("sin_cache", self.sin, persistent = False)    
    
    def forward(self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # while cos.ndim < x.ndim:
        #     cos = cos.unsqueeze(-3)
        #     sin = sin.unsqueeze(-3)

        # cos = cos.to(device=x.device, dtype=x.dtype)
        # sin = sin.to(device=x.device, dtype=x.dtype)

        x_e = cos*x_even - sin*x_odd
        x_o = sin*x_even + cos*x_odd

        output = torch.stack(
            [x_e,x_o],
            dim = -1
        ).flatten(-2)

        return output

def softmax(x: torch.tensor, i:int):
    sub = torch.max(x, dim = i, keepdim = True).values
    x = x - sub
    e_x = torch.exp(x)
    d = torch.sum(e_x, dim = i, keepdim = True)
    return e_x/d


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]):
    m = torch.max(inputs, dim = -1, keepdim = True).values
    shift_l = inputs - m

    a = torch.log(torch.sum(torch.exp(shift_l),dim = -1, keepdim = True))
    j = torch.gather(
        shift_l,
        dim = -1,
        index = targets.unsqueeze(-1),
    ).squeeze(1)

    return (a - j).mean()

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        a = softmax(scores, -1) @ V
        return a


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        theta: float | None = None,
        max_len: int | None = None) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.output_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

        if theta is not None and max_len is not None:
            self.rope = RoPE(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_len,
                device=device,
            )
        else:
            self.rope = None


    def forward(self, 
    x:torch.tensor, 
    token_positions: torch.tensor | None = None,) -> torch.tensor:
        sequence_length = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(
            Q,
            "... sequence_length (num_heads d_head) "
            "-> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
        )

        K = rearrange(
            K,
            "... sequence_length (num_heads d_head) "
            "-> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
        )

        V = rearrange(
            V,
            "... sequence_length (num_heads d_head) "
            "-> ... num_heads sequence_length d_head",
            num_heads=self.num_heads,
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(
            torch.ones(sequence_length,
            sequence_length,
            dtype = torch.bool,
            device = x.device)
        )

        output = scaled_dot_product_attention(Q, K, V, mask)

        output = rearrange(
            output,
            "... num_heads sequence_length d_head "
            "-> ... sequence_length (num_heads d_head)"
        )

        return self.output_proj(output)
