"""Extract per-frame spatial feature maps for CMU-MOSI (input to the 2D spatial encoder).

For every sample in mosi_data.pkl, T frames are sampled uniformly from the raw
clip, the speaker's face is cropped (OpenCV Haar cascade), and a frozen CNN is
run up to its last 7x7 stage: ImageNet ResNet-18 (512 channels) or the HSEmotion
AffectNet EfficientNet-B0 (320 channels). The map is kept without global
pooling, so the learnable 2D convolution in src/spatial.py sees the spatial
layout of the face.

Output, one file per split in the same sample order as mosi_data.pkl:
    <out_dir>/<split>.npy   float16, shape (N, T, C, 7, 7)
    <out_dir>/meta.json     settings and face-detection statistics

Example:
    python tools/extract_mosi_spatial.py --data_path C:/mplmm_data/mosi_data.pkl \
        --raw_dir C:/mplmm_data/CMU-MOSI/Raw --out_dir C:/mplmm_data/mosi_spatial_t16
"""
import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torchvision
from torch import nn

SPLITS = ["train", "valid", "test"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frames(path, n_frames):
    """Decode n_frames uniformly spaced frames (indices repeat for short clips)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n_frames).round().astype(int)
    wanted, decoded, i = set(idxs.tolist()), {}, 0
    while i <= idxs[-1] and cap.grab():
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                decoded[i] = frame
        i += 1
    cap.release()
    if not decoded:
        raise RuntimeError(f"no frame could be decoded: {path}")
    # The container's frame count can overshoot what is decodable; use the nearest decoded frame.
    keys = np.array(sorted(decoded))
    return [decoded[keys[np.abs(keys - j).argmin()]] for j in idxs]


def detect_face(cascade, frame):
    """Largest face as (x, y, w, h) in full-resolution pixels, or None."""
    h, w = frame.shape[:2]
    scale = min(1.0, 480.0 / max(h, w))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=np.float32) / scale


def crop_square(frame, box, margin, size):
    """Square crop around box enlarged by margin, zero-padded at the borders."""
    x, y, w, h = box
    side = int(round(max(w, h) * margin))
    cx, cy = int(round(x + w / 2)), int(round(y + h / 2))
    padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = cx - side // 2 + side, cy - side // 2 + side
    return cv2.resize(padded[y0 : y0 + side, x0 : x0 + side], (size, size), interpolation=cv2.INTER_AREA)


def face_crops(cascade, frames, margin, size):
    """Crop every frame; frames without a detection use the clip's median face box."""
    boxes = [detect_face(cascade, f) for f in frames]
    found = [b for b in boxes if b is not None]
    if found:
        fallback = np.median(np.stack(found), axis=0)
    else:  # no face in the whole clip: centre square crop
        h, w = frames[0].shape[:2]
        side = min(h, w) / margin
        fallback = np.array([(w - side) / 2, (h - side) / 2, side, side], dtype=np.float32)
    crops = [crop_square(f, b if b is not None else fallback, margin, size) for f, b in zip(frames, boxes)]
    return crops, len(found)


def build_backbone(name, device):
    """Frozen CNN trunk producing a (C, 7, 7) map for a 224x224 face crop; returns (trunk, C)."""
    if name == "resnet18":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        trunk, channels = nn.Sequential(*list(net.children())[:-2]), 512  # drop avgpool + fc
    elif name == "hsemotion":
        trunk, channels = load_hsemotion(), 320
    elif name == "hsemotion_head":
        trunk, channels = load_hsemotion(head=True), 1280
    else:
        raise ValueError(f"unknown backbone {name}")
    return trunk.to(device).eval(), channels


def load_hsemotion(model_name="enet_b0_8_va_mtl", head=False):
    """EfficientNet-B0 trained on AffectNet for expressions + valence/arousal (HSEmotion).

    The download is a pickled timm model that predates current timm. It is loaded
    with weights_only=True, allowing only the torch.nn / timm layer classes it
    references, and its weights are copied into a fresh tf_efficientnet_b0 (same
    TF-style padding and BN eps; outputs match the original exactly). The trunk
    ends at the last block stage, a (320, 7, 7) map, or with head=True after
    conv_head + bn2, the (1280, 7, 7) map the expression/valence classifier pools.
    """
    import importlib

    import timm
    from hsemotion.facial_emotions import get_model_path

    path = get_model_path(model_name)
    names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    assert all(n.startswith(("torch.nn.", "timm.")) for n in names), f"unexpected classes: {names}"
    classes = [getattr(importlib.import_module(mod), cls) for mod, cls in (n.rsplit(".", 1) for n in names)]
    with torch.serialization.safe_globals(classes):
        old = torch.load(path, map_location="cpu", weights_only=True)
    net = timm.create_model("tf_efficientnet_b0", num_classes=10, pretrained=False)
    net.load_state_dict(old.state_dict())
    layers = [net.conv_stem, net.bn1, net.blocks] + ([net.conv_head, net.bn2] if head else [])
    return nn.Sequential(*layers)


def to_tensor(crops, device):
    rgb = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", required=True, help="mosi_data.pkl (defines samples and split order)")
    parser.add_argument("--raw_dir", required=True, help="CMU-MOSI Raw/ folder: <video_id>/<clip_id>.mp4")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "hsemotion", "hsemotion_head"])
    parser.add_argument("--frames", type=int, default=16, help="frames sampled per clip (T)")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=1.4, help="face box enlargement before cropping")
    parser.add_argument("--max_samples", type=int, default=None, help="per split, for quick tests")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    trunk, channels = build_backbone(args.backbone, device)
    data = pickle.load(open(args.data_path, "rb"))

    meta = {"frames": args.frames, "crop_size": args.crop_size, "margin": args.margin,
            "backbone": args.backbone, "channels": channels, "dtype": "float16", "splits": {}}
    for split in SPLITS:
        ids = [row[0].decode() for row in data[split]["id"]][: args.max_samples]
        out = np.lib.format.open_memmap(os.path.join(args.out_dir, f"{split}.npy"), mode="w+",
                                        dtype=np.float16, shape=(len(ids), args.frames, channels, 7, 7))
        frames_with_face, clips_without_face, start = 0, [], time.time()
        for n, sample_id in enumerate(ids):
            video, clip = sample_id.rsplit("_", 1)
            frames = read_frames(os.path.join(args.raw_dir, video, f"{clip}.mp4"), args.frames)
            crops, n_faces = face_crops(cascade, frames, args.margin, args.crop_size)
            frames_with_face += n_faces
            if n_faces == 0:
                clips_without_face.append(sample_id)
            with torch.inference_mode():
                out[n] = trunk(to_tensor(crops, device)).cpu().numpy().astype(np.float16)
            if (n + 1) % 200 == 0 or n + 1 == len(ids):
                print(f"{split}: {n + 1}/{len(ids)} clips | {time.time() - start:.0f}s", flush=True)
        out.flush()
        del out
        meta["splits"][split] = {
            "samples": len(ids),
            "face_detection_rate": frames_with_face / (len(ids) * args.frames),
            "clips_without_face": clips_without_face,
        }
        print(f"{split}: face found in {meta['splits'][split]['face_detection_rate']:.1%} of frames, "
              f"{len(clips_without_face)} clips with no face (centre crop used)")
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
