import torch.nn.functional as F
from torch import nn


class SpatialVisualEncoder(nn.Module):
    """Learnable 2D convolution over per-frame spatial feature maps (Technique A).

    x_V^t = Proj(Flatten(ReLU(W_2D * X_V^t + b)))   (proposal Eqs. 3-4)

    Input:  (B, T, C, H, W) frame feature maps from tools/extract_mosi_spatial.py.
    Output: (B, T, out_dim) visual sequence consumed by PromptModel.

    With pool=True the map is averaged to 1x1 before the convolution. This removes
    the spatial layout but keeps everything else the same, as a no-spatial control.

    Raw ResNet maps (std ~1.1, max ~22) are much larger than the FACET features the
    pretrained network was built around (std ~0.12), and without normalization the
    encoder output grew to std ~3.4 and training was unstable. The input is
    therefore batch-normalized and the output layer-normalized.
    """

    def __init__(self, in_channels, map_size, conv_channels=32, out_dim=64, pool=False, dropout=0.1):
        super(SpatialVisualEncoder, self).__init__()
        self.pool = pool
        self.dropout = dropout
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1)
        cells = 1 if pool else map_size[0] * map_size[1]
        self.proj = nn.Linear(conv_channels * cells, out_dim)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        batch, frames = x.shape[:2]
        x = self.input_norm(x.flatten(0, 1))
        if self.pool:
            x = x.mean(dim=(-2, -1), keepdim=True)
        x = F.relu(self.conv(x)).flatten(1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_norm(self.proj(x)).view(batch, frames, -1)
