"""Mixture of experts that adds a per-sample correction to MPLMM's generator.

For a missing target modality m with available sources S (subset of the
other two modalities):

    expert_k(x)[m] = mean over s in S of  Resample_{T_m}( Translator_k^{s->m}(x_s) )
    router(x)      = softmax( MLP([summary of available modalities, mode]) / tau )
    moe(x)[m]      = sum_k p_k(x)[m] * expert_k(x)[m]                 [prompt_dim, T_m]

MoEPromptModel adds tanh(gate_m) * moe(x)[m] to the baseline generator output.

Design choices:
  * Translators are low-rank (LayerNorm -> D_s x r -> GELU -> r x prompt_dim):
    a full copy of the baseline generator per expert would be 57k-275k
    parameters each, too many for ~1.3k training samples (MOSI, CH-SIMS).
  * Time alignment is parameter-free (average pooling to shorten, linear
    interpolation to lengthen); the baseline's learned time-mixing conv is
    the heaviest part of the generator and stays shared in the base path.
  * The expert output is linear (no final activation), so the correction can
    be negative; the baseline output ends in GELU and cannot go below -0.17.
  * Modalities marked missing are never read: they are excluded from the
    router summary and from the source mean with torch.where, which also
    blocks NaN/inf and gradients from the unused branch.
  * The router's last layer starts at zero: p = 1/K for every sample.
"""

import torch
import torch.nn.functional as F
from torch import nn

MODALITIES = ("l", "a", "v")
MOD_INDEX = {m: j for j, m in enumerate(MODALITIES)}

# Missing-mode code -> missing modalities (MPLMM convention).
MISSING = {0: ("l",), 1: ("a",), 2: ("v",), 3: ("l", "a"), 4: ("l", "v"), 5: ("a", "v"), 6: ()}

# Source modalities of each target, in the order MPLMM concatenates them.
SOURCES = {"l": ("a", "v"), "a": ("l", "v"), "v": ("l", "a")}

# AVAILABLE[mode, j] = modality j is present in that mode.
AVAILABLE = torch.tensor(
    [[m not in MISSING[mode] for m in MODALITIES] for mode in range(7)], dtype=torch.bool
)


def resample(x, length):
    """[B, C, T] -> [B, C, length]: average-pool to shorten, interpolate to lengthen."""
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return F.adaptive_avg_pool1d(x, length)
    return F.interpolate(x, size=length, mode="linear", align_corners=False)


class Translator(nn.Module):
    """Low-rank map of one source modality into the target feature space, per time step."""

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, rank), nn.GELU(), nn.Linear(rank, out_dim)
        )

    def forward(self, x):  # [B, D, T] -> [B, out_dim, T]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class Router(nn.Module):
    """Per-sample expert weights for each target modality: [B, 3, K]."""

    def __init__(self, in_dims, n_experts, hidden=16, tau=1.0, n_modes=7):
        super().__init__()
        self.n_experts, self.tau = n_experts, tau
        self.summaries = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden)) for d in in_dims]
        )
        self.mode_embedding = nn.Embedding(n_modes, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, 3 * n_experts)
        )
        nn.init.zeros_(self.mlp[-1].weight)  # uniform routing at initialisation
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, inputs, missing_mod, available):
        """inputs: [B, D_j, T_j] per modality; available: [B, 3] bool."""
        parts = []
        for j, x in enumerate(inputs):
            h = self.summaries[j](x.mean(dim=2))  # time mean -> [B, hidden]
            parts.append(torch.where(available[:, j:j + 1], h, torch.zeros_like(h)))
        parts.append(self.mode_embedding(missing_mod))
        logits = self.mlp(torch.cat(parts, dim=1)).view(-1, 3, self.n_experts)
        return torch.softmax(logits / self.tau, dim=-1)


class MoEGenerator(nn.Module):
    def __init__(self, in_dims, seq_lens, out_dim, n_experts=3, rank=16, router_hidden=16,
                 tau=1.0):
        super().__init__()
        self.seq_lens, self.n_experts = tuple(seq_lens), n_experts
        self.translators = nn.ModuleDict({
            f"{s}2{m}": nn.ModuleList(
                [Translator(in_dims[MOD_INDEX[s]], out_dim, rank) for _ in range(n_experts)]
            )
            for m in MODALITIES for s in SOURCES[m]
        })
        self.router = Router(in_dims, n_experts, router_hidden, tau)
        self.register_buffer("available_table", AVAILABLE.clone(), persistent=False)

    def forward(self, x_l, x_a, x_v, missing_mod):
        """Inputs [B, D, T] (as inside PromptModel.forward); missing_mod [B] long.

        Returns ([moe_l, moe_a, moe_v] each [B, prompt_dim, T_m], probs [B, 3, K]).
        Rows of a target that is not missing are computed but never used.
        """
        inputs = {"l": x_l, "a": x_a, "v": x_v}
        available = self.available_table[missing_mod]  # [B, 3]
        probs = self.router((x_l, x_a, x_v), missing_mod, available)

        outputs = []
        for j, m in enumerate(MODALITIES):
            length = self.seq_lens[j]
            total, count = 0.0, 0.0
            for s in SOURCES[m]:
                mask = available[:, MOD_INDEX[s]].view(-1, 1, 1, 1)
                experts = torch.stack(
                    [resample(t(inputs[s]), length) for t in self.translators[f"{s}2{m}"]], dim=1
                )  # [B, K, out_dim, T_m]
                total = total + torch.where(mask, experts, torch.zeros_like(experts))
                count = count + mask.float()
            mixed = total / count.clamp(min=1.0)  # mean over the available sources
            outputs.append((probs[:, j].view(-1, self.n_experts, 1, 1) * mixed).sum(dim=1))
        return outputs, probs
