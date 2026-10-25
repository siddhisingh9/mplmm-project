import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np


class SIMSData(Dataset):
    def __init__(self, data_path, split, drop_rate, full_data=False, normalize=True, clip=10.0):
        super(SIMSData, self).__init__()
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        self.data = data[split]
        self.split = split
        self.drop_rate = drop_rate
        self.full_data = full_data

        # Raw SIMS audio/vision features are on wildly different scales
        # (vision std ~1e5, max ~3e7). Z-score per feature with TRAIN-split
        # statistics, then clip outliers.
        self.feats = {}
        for key in ['text', 'audio', 'vision']:
            feat = np.asarray(self.data[key], dtype=np.float32)
            feat[~np.isfinite(feat)] = 0
            if normalize and key in ['audio', 'vision']:
                train = np.asarray(data['train'][key], dtype=np.float64)
                train[~np.isfinite(train)] = 0
                train = train.reshape(-1, train.shape[-1])
                mean = train.mean(axis=0).astype(np.float32)
                std = train.std(axis=0).astype(np.float32)
                std[std < 1e-6] = 1.0
                feat = np.clip((feat - mean) / std, -clip, clip)
            self.feats[key] = feat.astype(np.float32)

        # Shape labels as (N, 1, 1) to match MOSI/MOSEI, so that
        # batch_Y.squeeze(-1) gives (B, 1) == preds shape in train.py.
        self.labels = np.asarray(self.data['regression_labels'], dtype=np.float32).reshape(-1, 1, 1)

        self.orig_dims = [
            self.feats['text'].shape[2],
            self.feats['audio'].shape[2],
            self.feats['vision'].shape[2]
        ]

    def get_dim(self):
        return self.orig_dims

    def get_seq_len(self):
        return self.feats['text'].shape[1], self.feats['audio'].shape[1], self.feats['vision'].shape[1]

    def __len__(self):
        return self.labels.shape[0]

    def get_missing_mode(self):
        if self.full_data:
            return 6
        if random.random() < self.drop_rate:
            return random.randint(0, 5)
        else:
            return 6


    def __getitem__(self, idx):
        L_feat = torch.from_numpy(self.feats['text'][idx])
        A_feat = torch.from_numpy(self.feats['audio'][idx])
        V_feat = torch.from_numpy(self.feats['vision'][idx])
        label = torch.from_numpy(self.labels[idx])
        X = (L_feat, A_feat, V_feat)
        missing_code = self.get_missing_mode()

        return X, label, missing_code
