import torch
import torch.nn as nn
import torch.nn.functional as F


class DGPrototypeAttention(nn.Module):
    def __init__(self, dim, heads, num_learnable_protos):
        super().__init__()

        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.d = dim // heads

        # Standard attention scaling factor
        self.scale = self.d ** -0.5

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        self.Wk_proto = nn.Linear(dim, dim)
        self.Wv_proto = nn.Linear(dim, dim)

        self.prototype = nn.Parameter(
            torch.randn(num_learnable_protos, dim) * 0.02
        )

        self.out = nn.Linear(dim, dim)

    def _split(self, x):
        """
        Input:  [B, N, D]
        Output: [B, H, N, d]
        """
        B, N, D = x.shape
        return x.view(B, N, self.heads, self.d).transpose(1, 2)

    def forward(self, x, memory=None):
        """
        x: [B, N, D]
        """

        B, N, D = x.shape

        # Token queries / keys / values
        q = self._split(self.Wq(x))  # [B,H,N,d]
        k = self._split(self.Wk(x))  # [B,H,N,d]
        v = self._split(self.Wv(x))  # [B,H,N,d]

        # Learnable prototypes
        proto = self.prototype.unsqueeze(0).expand(B, -1, -1)  # [B,P,D]

        kp = self._split(self.Wk_proto(proto))  # [B,H,P,d]
        vp = self._split(self.Wv_proto(proto))  # [B,H,P,d]

        # Concatenate token + prototype keys/values
        k_all = torch.cat([k, kp], dim=2)  # [B,H,N+P,d]
        v_all = torch.cat([v, vp], dim=2)  # [B,H,N+P,d]

        # Attention over both tokens and prototypes
        attn = torch.matmul(q, k_all.transpose(-2, -1))  # [B,H,N,N+P]
        attn = F.softmax(attn * self.scale, dim=-1)

        out = torch.matmul(attn, v_all)  # [B,H,N,d]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out(out), attn