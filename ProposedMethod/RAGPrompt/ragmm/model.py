"""RAGPromptModel: MPLMM's PromptModel with a per-sample generative prompt.

Baseline: the generative prompt for missing modality j is one learned tensor
    P_G[j]                                   (prompt_dim x prompt_length)
shared by every sample.

Here it becomes
    P_G(x)[j] = P_G[j] + tanh(g_j) * Attn_j( query  = P_G[j] tokens,
                                             keys   = proj_j(retrieved values),
                                             bias   = log retrieval weights )
where the retrieved values are modality j taken from the top-K training samples
most similar to x on its AVAILABLE modalities (see ragmm.bank).

g_j starts at 0, so at initialisation the model is exactly the baseline
(checked by scripts/sanity_checks.py). The frozen MOSEI backbone and every
other baseline component are reused unchanged.

get_complete_data() and forward() are copied from MPLMM/src/model.py because
the baseline reads self.generative_prompt inside them; lines that differ from
the original are marked "# RAG".
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from ragmm.bank import MOD_INDEX, MODE_SPEC
from ragmm.bridge import PromptModel, transfer_model
from ragmm.config import MODEL_DEFAULTS, PRETRAINED_BACKBONE, dataset_cfg

DEFAULT_RAG = {"enabled": True, "k": 5, "tau": 0.1, "random": False, "heads": 3}


class RAGPromptModel(PromptModel):
    def __init__(self, hyp_params, rag_cfg=None):
        super().__init__(hyp_params)
        self.rag_cfg = {**DEFAULT_RAG, **(rag_cfg or {})}
        self.bank = None  # attached with attach_bank(); never saved in the state_dict

        pd = self.prompt_dim
        in_dims = (self.orig_d_l, self.orig_d_a, self.orig_d_v)
        # LayerNorm first: raw features differ wildly in scale (e.g. MOSEI audio).
        self.rag_value_proj = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, pd)) for d in in_dims]
        )
        self.rag_attn = nn.ModuleList(
            [nn.MultiheadAttention(pd, self.rag_cfg["heads"], batch_first=True) for _ in in_dims]
        )
        self.rag_gate = nn.Parameter(torch.zeros(3))

    def attach_bank(self, bank):
        self.bank = bank.to(next(self.parameters()).device)
        return self

    def rag_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith("rag_")]

    # ------------------------------------------------------- dynamic prompts
    def dynamic_prompts(self, x_l, x_a, x_v, missing_mod, exclude_groups=None,
                        exclude_indices=None):
        """Per-sample generative prompts [B, 3, prompt_dim, prompt_length].

        x_*: raw model inputs [B, T, D]. Samples with mode 6 (nothing missing)
        keep the static prompt; the prompt is unused for them anyway.
        """
        batch = x_l.shape[0]
        static = self.generative_prompt  # [3, pd, pl]
        prompts = static.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        if not self.rag_cfg["enabled"]:
            return prompts
        if self.bank is None:
            raise RuntimeError("RAG is enabled but no memory bank is attached")

        cfg = self.rag_cfg
        queries = self.bank.query_keys(x_l, x_a, x_v)
        n_seg = self.bank.meta["n_segments"]
        prompt_len = static.shape[2]

        for mode, (available, missing) in MODE_SPEC.items():
            sel = (missing_mod == mode).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            indices, log_w = self.bank.retrieve(
                {m: queries[m][sel] for m in available}, available, cfg["k"], cfg["tau"],
                exclude_groups=exclude_groups[sel] if exclude_groups is not None else None,
                exclude_indices=exclude_indices[sel] if exclude_indices is not None else None,
                random=cfg["random"],
            )
            # retrieval weight of each token = weight of the sample it came from
            bias = log_w.repeat_interleave(n_seg, dim=1)  # [b, K*S]
            attn_bias = bias.unsqueeze(1).expand(-1, prompt_len, -1)
            attn_bias = attn_bias.repeat_interleave(cfg["heads"], dim=0)  # [b*heads, pl, K*S]

            for m in missing:
                j = MOD_INDEX[m]
                retrieved = self.bank.values[m][indices]  # [b, K, S, D_m]
                tokens = self.rag_value_proj[j](retrieved.flatten(1, 2))  # [b, K*S, pd]
                query = static[j].T.unsqueeze(0).expand(sel.numel(), -1, -1)  # [b, pl, pd]
                context, _ = self.rag_attn[j](
                    query, tokens, tokens, attn_mask=attn_bias, need_weights=False
                )
                prompts[sel, j] = static[j] + torch.tanh(self.rag_gate[j]) * context.transpose(1, 2)
        return prompts

    # ------------------------------------- copied from PromptModel, see header
    def get_complete_data(self, x_l, x_a, x_v, missing_mode, gen_prompt):  # RAG: gen_prompt
        x_l, x_a, x_v = x_l.unsqueeze(dim=0), x_a.unsqueeze(dim=0), x_v.unsqueeze(dim=0)
        if missing_mode == 0:
            x_l = torch.cat(
                [gen_prompt[0], self.a2l(x_a)[0], self.v2l(x_v)[0]], dim=1  # RAG
            ).unsqueeze(dim=0)
            x_l = self.l_avp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = self.proj_a(x_a) + self.prompta_nm
            x_v = self.proj_v(x_v) + self.promptv_nm
        elif missing_mode == 1:
            x_a = torch.cat(
                [gen_prompt[1], self.l2a(x_l)[0], self.v2a(x_v)[0]], dim=1  # RAG
            ).unsqueeze(dim=0)
            x_a = self.a_lvp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.proj_v(x_v) + self.promptv_nm
            x_l = self.proj_l(x_l) + self.promptl_nm
        elif missing_mode == 2:
            x_v = torch.cat(
                [gen_prompt[2], self.l2v(x_l)[0], self.a2v(x_a)[0]], dim=1  # RAG
            ).unsqueeze(dim=0)
            x_v = self.v_alp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = self.proj_l(x_l) + self.promptl_nm
            x_a = self.proj_a(x_a) + self.prompta_nm
        elif missing_mode == 3:
            x_l = torch.cat([gen_prompt[0], self.v2l(x_v)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_a = torch.cat([gen_prompt[1], self.v2a(x_v)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_l = self.l_vp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = self.a_vp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.proj_v(x_v) + self.promptv_nm
        elif missing_mode == 4:
            x_l = torch.cat([gen_prompt[0], self.a2l(x_a)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_v = torch.cat([gen_prompt[2], self.a2v(x_a)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_l = self.l_ap(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_v = self.v_ap(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_a = self.proj_a(x_a) + self.prompta_nm
        elif missing_mode == 5:
            x_a = torch.cat([gen_prompt[1], self.l2a(x_l)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_v = torch.cat([gen_prompt[2], self.l2v(x_l)[0]], dim=1).unsqueeze(dim=0)  # RAG
            x_a = self.a_lp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.v_lp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = self.proj_l(x_l) + self.promptl_nm
        else:
            x_a = self.proj_a(x_a) + self.prompta_nm
            x_l = self.proj_l(x_l) + self.promptl_nm
            x_v = self.proj_v(x_v) + self.promptv_nm

        return x_l, x_a, x_v

    def forward(self, x_l, x_a, x_v, missing_mod, exclude_groups=None,
                exclude_indices=None):  # RAG: exclude_*
        missing_mod = torch.as_tensor(missing_mod, device=x_l.device).long()  # RAG
        gen_prompts = self.dynamic_prompts(  # RAG
            x_l, x_a, x_v, missing_mod, exclude_groups, exclude_indices
        )

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        xx_l, xx_a, xx_v = [], [], []
        for idx in range(len(x_l)):
            x_l_temp, x_a_temp, x_v_temp = self.get_complete_data(
                x_l[idx], x_a[idx], x_v[idx], missing_mod[idx], gen_prompts[idx]  # RAG
            )
            xx_l.append(x_l_temp)
            xx_a.append(x_a_temp)
            xx_v.append(x_v_temp)
        xx_l, xx_a, xx_v = torch.cat(xx_l), torch.cat(xx_a), torch.cat(xx_v)

        proj_x_a = xx_a.permute(2, 0, 1)
        proj_x_v = xx_v.permute(2, 0, 1)
        proj_x_l = xx_l.permute(2, 0, 1)

        self.get_proj_matrix()
        batch_prompt = torch.stack(
            [torch.matmul(self.missing_type_prompt, self.mp[missing_mod[idx]])
             for idx in range(len(x_l))]
        )
        batch_prompt = batch_prompt.transpose(0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = torch.cat([h_ls, batch_prompt[0].transpose(0, 1)], dim=0)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = torch.cat([h_as, batch_prompt[1].transpose(0, 1)], dim=0)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as[-1]

        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = torch.cat([h_vs, batch_prompt[2].transpose(0, 1)], dim=0)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training)
        )
        last_hs_proj += last_hs
        return self.out_layer(last_hs_proj)



# ------------------------------------------------------------ construction
def make_hyp_params(dataset, train_split):
    """The hyper-parameter namespace PromptModel expects (as built by MPLMM/main.py)."""
    orig_d_l, orig_d_a, orig_d_v = train_split.get_dim()
    return SimpleNamespace(
        **MODEL_DEFAULTS,
        orig_d_l=orig_d_l,
        orig_d_a=orig_d_a,
        orig_d_v=orig_d_v,
        seq_len=tuple(train_split.get_seq_len()),
        output_dim=dataset_cfg(dataset)["output_dim"],
        dataset=dataset,
    )


def build_model(dataset, train_split, rag_cfg, device):
    """RAGPromptModel with the MOSEI backbone loaded and frozen.

    transfer_model() (MPLMM) copies every backbone weight except the input
    projections and output head, and freezes what it copied. The rag_* layers
    are new, so they stay trainable.
    """
    hyp_params = make_hyp_params(dataset, train_split)
    model = RAGPromptModel(hyp_params, rag_cfg).to(device)
    model = transfer_model(model, str(PRETRAINED_BACKBONE))
    return model, hyp_params


# ------------------------------------------------------------- checkpoints
def save_checkpoint(model, path, hyp_params, extra=None):
    """State dict + the settings needed to rebuild the model (bank stored separately)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyp_params": vars(hyp_params),
            "rag_cfg": model.rag_cfg,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device, bank=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = RAGPromptModel(SimpleNamespace(**ckpt["hyp_params"]), ckpt["rag_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    if model.rag_cfg["enabled"]:
        if bank is None:
            raise ValueError(f"{path} uses RAG; pass the memory bank it was trained with")
        model.attach_bank(bank)
    return model.eval(), ckpt
