#!/usr/bin/env python3
"""
Debug the Refcoco dataloader outputs and visualize a sample to verify correctness.

- Loads the same TSV as training (by default), builds the task and dataset
- Iterates one small batch and prints detailed tensor shapes and stats
- Verifies that `coordinates` aligns with decoder inputs
- Saves a visualization image overlaying coordinates on the input image

Usage:
  python tools/debug_dataloader.py \
    --data /data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco+g_train_shuffled.tsv,/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco/refcoco_val.tsv \
    --selected-cols 0,5,6,2,4,3,7 \
    --bpe-dir ./utils/BPE \
    --num-bins 64 \
    --patch-image-size 512 \
    --max-src-length 80 \
    --max-tgt-length 420 \
    --batch-size 2 \
    --print-steps 80 \
    --sample-index 0
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.refcoco import RefcocoTask, RefcocoConfig
from fairseq.data import Dictionary
from types import SimpleNamespace


def to_cpu_numpy(t):
    return t.detach().cpu().numpy()


def denormalize(img_tensor: torch.Tensor, mean=None, std=None):
    # img_tensor: [3, H, W]
    if mean is None:
        mean = [0.5, 0.5, 0.5]
    if std is None:
        std = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp_(0, 1)


def build_task_from_args(args) -> RefcocoTask:
    cfg = RefcocoConfig()
    cfg.data = args.data
    cfg.selected_cols = args.selected_cols
    cfg.bpe_dir = args.bpe_dir
    cfg.num_bins = args.num_bins
    cfg.patch_image_size = args.patch_image_size
    cfg.max_src_length = args.max_src_length
    cfg.max_tgt_length = args.max_tgt_length

    # Manually build dictionaries to avoid extra deps
    src_dict = Dictionary()
    tgt_dict = Dictionary()
    for i in range(cfg.num_bins):
        for j in range(cfg.num_bins):
            src_dict.add_symbol(f"<bin_{i}_{j}>")
            tgt_dict.add_symbol(f"<bin_{i}_{j}>")

    task = RefcocoTask(cfg, src_dict, tgt_dict)
    # Minimal BPE stub to satisfy dataset.encode_text interface
    task.bpe = SimpleNamespace(encode=lambda s: s)
    task.load_dataset("train")
    return task


def print_batch_info(batch):
    print("Batch keys:", list(batch.keys()))
    print("Net input keys:", list(batch["net_input"].keys()))

    for k, v in batch["net_input"].items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

    target = batch["target"]
    print("target:", tuple(target.shape), target.dtype)
    print("token_type:", tuple(batch["token_type"].shape), batch["token_type"].dtype)

    # Basic stats
    coords = batch["net_input"].get("coordinates")
    if coords is not None:
        cmin = float(coords.min().item()) if coords.numel() else 0.0
        cmax = float(coords.max().item()) if coords.numel() else 0.0
        print(f"coordinates stats: min={cmin:.4f} max={cmax:.4f}")


def validate_alignment(batch):
    ni = batch["net_input"]
    coords = ni.get("coordinates")
    if coords is None:
        print("coordinates not present in net_input")
        return

    seq_len = int(ni["prev_output_tokens_11"].size(1))
    assert coords.size(1) == seq_len, (
        f"coordinates len {coords.size(1)} != prev_output seq_len {seq_len}")
    # BOS coordinate should be zeros
    bos_ok = torch.all(coords[:, 0].eq(0))
    print("BOS coord all zeros:", bool(bos_ok))

    # Compare against target (continuous) up to available length
    target = batch["target"]  # [B, T_cont, 2]
    tok = batch["token_type"].long()  # [B, T]
    B = target.size(0)

    # Original check: coords[t] vs target[t-1] (expected alignment for previous-step feeding)
    diffs_prev = []
    for i in range(B):
        copy_len = min(seq_len - 1, target[i].size(0))
        if copy_len <= 0:
            continue
        diff_prev = torch.abs(coords[i, 1:1+copy_len] - target[i, :copy_len]).max().item()
        diffs_prev.append(diff_prev)
    print("Max abs diff per-sample (coords[t] vs target[t-1]) [first 5]:", [round(x, 6) for x in diffs_prev[:5]])

    # Leakage check: coords[t] vs target[t] on coordinate-token positions
    leakage_ratios = []
    leakage_max_abs = []
    for i in range(B):
        # Limit up to min lengths
        cur_len = min(seq_len, target[i].size(0))
        if cur_len <= 0:
            leakage_ratios.append(0.0)
            leakage_max_abs.append(0.0)
            continue
        mask_coord = (tok[i, :cur_len] == 0)
        if not mask_coord.any():
            leakage_ratios.append(0.0)
            leakage_max_abs.append(0.0)
            continue
        cur_coords = coords[i, :cur_len][mask_coord]
        cur_target = target[i, :cur_len][mask_coord]
        abs_diff = torch.abs(cur_coords - cur_target)
        # Count positions that are extremely close (potential leakage)
        near_eq = (abs_diff.max(dim=1).values < 1e-6).float()
        ratio = near_eq.mean().item()
        leakage_ratios.append(ratio)
        leakage_max_abs.append(abs_diff.max().item())
    print("Leakage ratio (coords[t]==target[t]) per-sample [first 5]:", [round(x, 4) for x in leakage_ratios[:5]])
    print("Leakage max abs diff per-sample [first 5]:", [round(x, 6) for x in leakage_max_abs[:5]])


def validate_values(batch):
    ni = batch["net_input"]
    coords = ni["coordinates"].float()  # [B,T,2]
    dxi = ni["delta_x1"].float()
    dyi = ni["delta_y1"].float()
    dx2 = ni["delta_x2"].float()
    dy2 = ni["delta_y2"].float()
    tok = batch["token_type"].long()   # [B,T]

    # Ranges and NaNs
    def rng_stats(t, name):
        tmin = float(t.min().item()) if t.numel() else 0.0
        tmax = float(t.max().item()) if t.numel() else 0.0
        isnan = bool(torch.isnan(t).any())
        isinf = bool(torch.isinf(t).any())
        print(f"{name}: min={tmin:.6f} max={tmax:.6f} nan={isnan} inf={isinf}")

    rng_stats(coords, "coordinates")
    rng_stats(dxi, "delta_x1")
    rng_stats(dx2, "delta_x2")
    rng_stats(dyi, "delta_y1")
    rng_stats(dy2, "delta_y2")

    # Delta consistency
    tol = 1e-5
    max_err_x = torch.abs(dxi + dx2 - 1).max().item()
    max_err_y = torch.abs(dyi + dy2 - 1).max().item()
    print(f"delta sums: max|dx1+dx2-1|={max_err_x:.6e} max|dy1+dy2-1|={max_err_y:.6e}")

    # Token types
    unique_types = sorted(list(set(tok.flatten().tolist())))
    print("token_type unique values:", unique_types)
    bsz, seqlen = tok.shape

    # For non-coordinate (tok != 0 and tok != -1) positions, coords should be zeros
    non_coord_mask = (tok != 0) & (tok != -1)
    if non_coord_mask.any():
        max_non_coord_val = coords[non_coord_mask].abs().max().item()
        print(f"non-coordinate coord abs max (should be 0): {max_non_coord_val:.6f}")
    else:
        print("non-coordinate positions: none")

    # For coordinate positions (tok == 0), coords should be within [0,1]
    coord_mask = (tok == 0)
    if coord_mask.any():
        cvals = coords[coord_mask]
        cmin = float(cvals.min().item())
        cmax = float(cvals.max().item())
        print(f"coordinate positions range: min={cmin:.6f} max={cmax:.6f}")
    else:
        print("coordinate positions: none")

    # Check shapes alignment across all prev_output sequences
    p11 = ni["prev_output_tokens_11"].size(1)
    p12 = ni["prev_output_tokens_12"].size(1)
    p21 = ni["prev_output_tokens_21"].size(1)
    p22 = ni["prev_output_tokens_22"].size(1)
    print("sequence lengths: prev_11/12/21/22, token_type, coordinates:", p11, p12, p21, p22, tok.size(1), coords.size(1))


def print_token_alignment(batch, index: int = 0, max_steps: int = 80, n_bins: int = 64):
    ni = batch["net_input"]
    tok = batch["token_type"][index].long()
    coords = ni["coordinates"][index].float()
    target = batch["target"][index].float()
    p11 = ni["prev_output_tokens_11"][index].long()
    p12 = ni["prev_output_tokens_12"][index].long()
    p21 = ni["prev_output_tokens_21"][index].long()
    p22 = ni["prev_output_tokens_22"][index].long()
    dxi = ni["delta_x1"][index].float()
    dyi = ni["delta_y1"][index].float()
    dx2 = ni["delta_x2"][index].float()
    dy2 = ni["delta_y2"][index].float()

    seq_len = int(tok.size(0))
    steps = min(seq_len, max_steps)
    print(f"\nPer-token alignment (sample {index}, showing {steps}/{seq_len} steps):")
    header = (
        " t   tok    p11    p12    p21    p22   p11.x  p11.y     x        y    "
        "tgt[t-1].x  tgt[t-1].y   tgt[t].x   tgt[t].y    dx1     dy1     dx2     dy2"
    )
    print(header)
    print("-" * len(header))
    for t in range(steps):
        tt = int(tok[t].item())
        v11 = int(p11[t].item()) if t < p11.size(0) else -1
        v12 = int(p12[t].item()) if t < p12.size(0) else -1
        v21 = int(p21[t].item()) if t < p21.size(0) else -1
        v22 = int(p22[t].item()) if t < p22.size(0) else -1
        # Decode p11 token to normalized bin corner coordinates if applicable
        if v11 >= 4:
            idx = v11 - 4
            xb = idx // n_bins
            yb = idx % n_bins
            p11x = float(xb) / float(n_bins - 1) if n_bins > 1 else 0.0
            p11y = float(yb) / float(n_bins - 1) if n_bins > 1 else 0.0
        else:
            p11x = float('nan')
            p11y = float('nan')
        x = float(coords[t, 0].item())
        y = float(coords[t, 1].item())
        # Safe indexing for target[t-1] and target[t]
        if t - 1 >= 0 and (t - 1) < target.size(0):
            tx_prev = float(target[t - 1, 0].item())
            ty_prev = float(target[t - 1, 1].item())
        else:
            tx_prev = 0.0
            ty_prev = 0.0
        if t < target.size(0):
            tx_cur = float(target[t, 0].item())
            ty_cur = float(target[t, 1].item())
        else:
            tx_cur = 0.0
            ty_cur = 0.0
        vx1 = float(dxi[t].item())
        vy1 = float(dyi[t].item())
        vx2 = float(dx2[t].item())
        vy2 = float(dy2[t].item())
        print(
            f"{t:3d}  {tt:3d}  {v11:5d}  {v12:5d}  {v21:5d}  {v22:5d}  "
            f"{p11x:6.3f} {p11y:6.3f}  {x:7.4f}  {y:7.4f}    {tx_prev:7.4f}   {ty_prev:7.4f}   "
            f"{tx_cur:7.4f}   {ty_cur:7.4f}   {vx1:6.3f}  {vy1:6.3f}  {vx2:6.3f}  {vy2:6.3f}"
        )

def visualize_sample(batch, out_dir: str, index: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    ni = batch["net_input"]
    img = denormalize(ni["patch_images"][index])  # [3,H,W] in [0,1]
    img_np = (to_cpu_numpy(img).transpose(1, 2, 0) * 255).astype(np.uint8)
    H, W = img_np.shape[:2]
    coords = ni.get("coordinates")
    if coords is None:
        print("No coordinates found; skipping visualization of points.")
        return
    pts = to_cpu_numpy(coords[index])  # [T,2] in [0,1]

    # Convert to pixel space and filter out exact zeros (BOS, seps/eos placeholders)
    xs = (pts[:, 0] * W)
    ys = (pts[:, 1] * H)
    mask = ~((pts[:, 0] == 0) & (pts[:, 1] == 0))
    xs = xs[mask]
    ys = ys[mask]
    # Draw using PIL to avoid matplotlib/numpy ABI issues
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    r = 2
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    out_path = os.path.join(out_dir, f"sample_{index}_inputs.png")
    pil_img.save(out_path)
    print("Saved visualization:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco+g_train_shuffled.tsv,/data0/arshkon/checkpoints/polyform_rl/datasets/finetune/refcoco/refcoco_val.tsv')
    parser.add_argument('--selected-cols', type=str, default='0,5,6,2,4,3,7')
    parser.add_argument('--bpe-dir', type=str, default='./utils/BPE')
    parser.add_argument('--num-bins', type=int, default=64)
    parser.add_argument('--patch-image-size', type=int, default=512)
    parser.add_argument('--max-src-length', type=int, default=80)
    parser.add_argument('--max-tgt-length', type=int, default=420)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='./debug_outputs')
    parser.add_argument('--print-steps', type=int, default=80)
    parser.add_argument('--sample-index', type=int, default=0)
    args = parser.parse_args()

    task = build_task_from_args(args)
    dataset = task.datasets['train']

    # Get a small iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_sentences=args.batch_size,
        num_workers=0,
        epoch=1,
    ).next_epoch_itr(shuffle=False)

    batch = next(iter(itr))
    print_batch_info(batch)
    validate_alignment(batch)
    validate_values(batch)
    print_token_alignment(batch, index=args.sample_index, max_steps=args.print_steps, n_bins=args.num_bins)
    visualize_sample(batch, args.out_dir, index=args.sample_index)


if __name__ == '__main__':
    main()


