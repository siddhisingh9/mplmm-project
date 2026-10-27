"""The only place that touches the MPLMM baseline code.

MPLMM's modules are imported as top-level `src` and `modules` packages, and the
pickled checkpoints (e.g. pretrained/mosei.pt) refer to `src.model.MULTModel`,
so the MPLMM directory itself has to be on sys.path.

Side effect to be aware of: importing src.utils imports src.mosidata, which
calls torch.set_default_device("cuda") when a GPU is available.
"""

import sys

from moemm.config import MPLMM_DIR

if str(MPLMM_DIR) not in sys.path:
    sys.path.insert(0, str(MPLMM_DIR))

from src.model import PromptModel  # noqa: E402
from src.utils import get_data, transfer_model  # noqa: E402

__all__ = ["PromptModel", "get_data", "transfer_model"]
