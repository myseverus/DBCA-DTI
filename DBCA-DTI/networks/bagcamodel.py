
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class BAGCA(nn.Module):

    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.1,      # 0.1
                 use_dynamic_alpha: bool = True, use_context_gating: bool = True):
        super().__init__()
        self.positional_drug = PositionalEncoding(hidden_size, max_len=1500, learnable=True)
        self.positional_prot = PositionalEncoding(hidden_size, max_len=1500, learnable=True)

        self.attn_map = FixedBidirectionalCrossAttn(
            hidden_size, heads, dropout=dropout,
            use_dynamic_alpha=use_dynamic_alpha
        )

        self.attention_fc_dp = nn.Linear(heads, hidden_size)
        self.attention_fc_pd = nn.Linear(heads, hidden_size)

        self.use_context_gating = use_context_gating
        if use_context_gating:
            self.gate_drug = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )
            self.gate_prot = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        drug_encoded = self.positional_drug(drug)
        prot_encoded = self.positional_prot(protein)
        # attn_matrix:[16,8,100,1200]、drug_context：[16,100,256], prot_context：[16,1200,256]
        attn_matrix, drug_context, prot_context = self.attn_map(drug_encoded, prot_encoded)
        # print("att_matrix")
        attn_dp = torch.mean(attn_matrix, dim=-1).transpose(-1, -2)
        drug_attn = self.attention_fc_dp(attn_dp)

        attn_pd = torch.mean(attn_matrix, dim=-2).transpose(-1, -2)
        prot_attn = self.attention_fc_pd(attn_pd)

        if self.use_context_gating:
            drug_gate = self.gate_drug(drug_encoded)
            prot_gate = self.gate_prot(prot_encoded)
            drug_context = drug_context * drug_gate
            prot_context = prot_context * prot_gate

        # drug_updated = drug_encoded + self.residual_dropout(drug_context * torch.sigmoid(drug_attn))
        # prot_updated = prot_encoded + self.residual_dropout(prot_context * torch.sigmoid(prot_attn))

        drug_updated = drug_encoded + drug_context * torch.sigmoid(drug_attn)
        prot_updated = prot_encoded + prot_context * torch.sigmoid(prot_attn)

        pair_features = torch.cat([drug_updated, prot_updated], dim=1)

        return pair_features


class FixedBidirectionalCrossAttn(nn.Module):

    def __init__(self, hid_dim: int, heads: int, dropout: float = 0.1,
                 use_dynamic_alpha: bool = True):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = heads
        self.d_k = hid_dim // heads
        self.use_dynamic_alpha = use_dynamic_alpha

        assert hid_dim % heads == 0, "Hidden dimension must be divisible by number of heads"

        self.q_d = nn.Linear(hid_dim, hid_dim)
        self.k_p = nn.Linear(hid_dim, hid_dim)
        self.v_p = nn.Linear(hid_dim, hid_dim)

        self.q_p = nn.Linear(hid_dim, hid_dim)
        self.k_d = nn.Linear(hid_dim, hid_dim)
        self.v_d = nn.Linear(hid_dim, hid_dim)

        if use_dynamic_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.alpha = 0.5

        self.attn_dropout = nn.Dropout(dropout)
        # self.4 = nn.LayerNorm(hid_dim)
        self.out_proj_d = nn.Linear(hid_dim, hid_dim)
        self.out_proj_p = nn.Linear(hid_dim, hid_dim)

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = drug.size(0)
        d_len, p_len = drug.size(1), protein.size(1)

        Q_d = self.q_d(drug).view(batch_size, d_len, self.n_heads, self.d_k).transpose(1, 2)
        K_p = self.k_p(protein).view(batch_size, p_len, self.n_heads, self.d_k).transpose(1, 2)
        V_p = self.v_p(protein).view(batch_size, p_len, self.n_heads, self.d_k).transpose(1, 2)

        Q_p = self.q_p(protein).view(batch_size, p_len, self.n_heads, self.d_k).transpose(1, 2)
        K_d = self.k_d(drug).view(batch_size, d_len, self.n_heads, self.d_k).transpose(1, 2)
        V_d = self.v_d(drug).view(batch_size, d_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_dp = torch.matmul(Q_d, K_p.transpose(-2, -1)) / math.sqrt(self.d_k)  # [bs, heads, d_len, p_len]
        attn_pd = torch.matmul(Q_p, K_d.transpose(-2, -1)) / math.sqrt(self.d_k)  # [bs, heads, p_len, d_len]

        attn_pd = attn_pd.transpose(2, 3)  # [bs, heads, d_len, p_len]

        if self.use_dynamic_alpha:
            alpha = torch.sigmoid(self.alpha)
            attn_matrix = alpha * attn_dp + (1 - alpha) * attn_pd
        else:
            attn_matrix = 0.5 * attn_dp + 0.5 * attn_pd

        attn_matrix = self.attn_dropout(F.softmax(attn_matrix, dim=-1)) # [16,8,100,1200]
        # attn_matrix = F.softmax(attn_matrix, dim=-1)

        context_d = torch.matmul(attn_matrix, V_p)  # [bs, heads, d_len, d_k]
        context_p = torch.matmul(attn_matrix.transpose(-2, -1), V_d)  # [bs, heads, p_len, d_k]

        context_d = context_d.transpose(1, 2).contiguous().view(batch_size, d_len, self.hid_dim)
        context_p = context_p.transpose(1, 2).contiguous().view(batch_size, p_len, self.hid_dim)

        context_d = self.out_proj_d(context_d)
        context_p = self.out_proj_p(context_p)

        return attn_matrix, context_d, context_p


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.scale = nn.Parameter(torch.ones(1)) if learnable else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, :x.size(1)]
        return x + self.scale * pe
