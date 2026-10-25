"""Memory bank of complete TRAINING samples, and top-K retrieval over it.

For every training sample and modality m in {l, a, v} the bank stores
  key[m]    [N, D_m]       mean over time of the features, standardised per
                           dimension with bank statistics, then L2-normalised
                           -> cosine similarity is a plain dot product
  value[m]  [N, S, D_m]    the sequence average-pooled into S time segments
                           (S = 16 = prompt length)
plus labels [N] and group ids [N] (source video / dialogue).

Keys and values are computed from exactly the tensors the model receives, so a
query at training or test time is built the same way as the bank.

Retrieval for a sample with missing mode `mode`:
  1. score every bank entry by cosine similarity on the AVAILABLE modalities
     only (averaged over them),
  2. during training, mask out entries from the sample's own group,
  3. keep the top-K, weight them by softmax(score / tau),
  4. return the MISSING modality's values of those K entries.
"""

import math

import torch
import torch.nn.functional as F

from ragmm.data import group_names, load_split, make_loader, to_device

MODALITIES = ("l", "a", "v")
MOD_INDEX = {"l": 0, "a": 1, "v": 2}  # same order as PromptModel.generative_prompt

# MPLMM missing-mode code -> (available modalities, missing modalities)
MODE_SPEC = {
    0: (("a", "v"), ("l",)),
    1: (("l", "v"), ("a",)),
    2: (("l", "a"), ("v",)),
    3: (("v",), ("l", "a")),
    4: (("a",), ("l", "v")),
    5: (("l",), ("a", "v")),
}


def summarize(x):
    """[B, T, D] -> [B, D]: mean over time."""
    return x.float().mean(dim=1)


def segment_pool(x, n_segments):
    """[B, T, D] -> [B, n_segments, D]: average-pool the time axis into segments."""
    return F.adaptive_avg_pool1d(x.float().transpose(1, 2), n_segments).transpose(1, 2)


class MemoryBank:
    def __init__(self, keys, key_mean, key_std, values, labels, groups, meta):
        self.keys = keys
        self.key_mean = key_mean
        self.key_std = key_std
        self.values = values
        self.labels = labels
        self.groups = groups
        self.meta = meta

    def __len__(self):
        return self.labels.shape[0]

    # ------------------------------------------------------------ build / io
    @classmethod
    @torch.no_grad()
    def build(cls, dataset, audio_len=350, n_segments=16, batch_size=64):
        """Build from the full training split, in dataset order (index i = sample i)."""
        train = load_split(dataset, "train", full_data=True, audio_len=audio_len)
        loader = make_loader(train, batch_size=batch_size, shuffle=False)
        device = torch.device("cpu")

        raw_keys = {m: [] for m in MODALITIES}
        values = {m: [] for m in MODALITIES}
        labels = []
        for batch in loader:
            (text, audio, vision), y, _ = to_device(batch, device)
            for m, x in zip(MODALITIES, (text, audio, vision)):
                raw_keys[m].append(summarize(x))
                values[m].append(segment_pool(x, n_segments))
            labels.append(y.reshape(-1).cpu())

        raw_keys = {m: torch.cat(v) for m, v in raw_keys.items()}
        key_mean = {m: k.mean(dim=0) for m, k in raw_keys.items()}
        key_std = {m: k.std(dim=0).clamp_min(1e-6) for m, k in raw_keys.items()}
        keys = {
            m: F.normalize((raw_keys[m] - key_mean[m]) / key_std[m], dim=-1)
            for m in MODALITIES
        }
        values = {m: torch.cat(v) for m, v in values.items()}
        labels = torch.cat(labels)

        names = group_names(dataset, "train")
        if len(names) != labels.shape[0]:
            raise RuntimeError(f"{len(names)} group ids for {labels.shape[0]} samples")
        unique = {name: i for i, name in enumerate(dict.fromkeys(names))}
        groups = torch.tensor([unique[n] for n in names], dtype=torch.long)

        meta = {
            "dataset": dataset,
            "split": "train",
            "audio_len": audio_len,
            "n_segments": n_segments,
            "n_groups": len(unique),
            "group_names": list(unique),  # group id i <-> group_names[i]
        }
        return cls(keys, key_mean, key_std, values, labels, groups, meta)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": self.keys,
                "key_mean": self.key_mean,
                "key_std": self.key_std,
                "values": self.values,
                "labels": self.labels,
                "groups": self.groups,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path, device):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run scripts/build_bank.py first")
        state = torch.load(path, map_location=device, weights_only=True)
        return cls(
            state["keys"], state["key_mean"], state["key_std"], state["values"],
            state["labels"], state["groups"], state["meta"],
        )

    def size_mb(self):
        tensors = [*self.keys.values(), *self.values.values(), self.labels, self.groups]
        return sum(t.numel() * t.element_size() for t in tensors) / 2**20

    # ------------------------------------------------------------- retrieval
    def query_keys(self, text, audio, vision):
        """Model inputs [B, T, D] -> normalised query keys, one [B, D] per modality."""
        raw = dict(zip(MODALITIES, (summarize(text), summarize(audio), summarize(vision))))
        return {
            m: F.normalize((raw[m] - self.key_mean[m]) / self.key_std[m], dim=-1)
            for m in MODALITIES
        }

    def group_ids(self, names):
        """Group names -> bank group ids; -1 for groups that are not in the bank."""
        lookup = {name: i for i, name in enumerate(self.meta["group_names"])}
        return torch.tensor([lookup.get(n, -1) for n in names], dtype=torch.long,
                            device=self.groups.device)

    def retrieve(self, queries, available, k, tau, exclude_groups=None, exclude_indices=None,
                 random=False):
        """Top-K bank entries for each query.

        queries: {m: [b, D_m]} for at least the `available` modalities
        exclude_groups: [b] group ids whose bank entries must not be returned
                        (-1 = nothing to exclude)
        exclude_indices: [b] bank index of the query itself (training only)
        random: control condition; K random entries with uniform weights
        Returns (indices [b, K], log_weights [b, K]).
        """
        b = queries[available[0]].shape[0]
        device = self.labels.device
        if random:
            scores = torch.rand(b, len(self), device=device)
        else:
            scores = sum(queries[m] @ self.keys[m].T for m in available) / len(available)
        if exclude_groups is not None:
            same_group = self.groups.unsqueeze(0) == exclude_groups.unsqueeze(1)
            scores = scores.masked_fill(same_group, float("-inf"))
        if exclude_indices is not None:
            scores = scores.scatter(1, exclude_indices.unsqueeze(1), float("-inf"))

        top_scores, indices = scores.topk(k, dim=1)
        if random:
            log_weights = torch.full_like(top_scores, -math.log(k))
        else:
            log_weights = F.log_softmax(top_scores / tau, dim=1)
        return indices, log_weights

    def to(self, device):
        move = lambda d: {m: t.to(device) for m, t in d.items()}  # noqa: E731
        self.keys, self.key_mean, self.key_std, self.values = (
            move(self.keys), move(self.key_mean), move(self.key_std), move(self.values)
        )
        self.labels, self.groups = self.labels.to(device), self.groups.to(device)
        return self
