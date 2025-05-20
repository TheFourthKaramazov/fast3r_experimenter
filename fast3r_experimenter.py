#!/usr/bin/env python3
"""
fast3r_experimenter.py   •   20-May-2025
----------------------------------------
Fast3R runner with timings, PnP poses, and a coloured fused cloud,
with optional view-dropping and pointmap-head choice (local/global).

Single scene
------------
python fast3r_experimenter.py \
    --images            /path/to/scene/images \
    --workspace         /results/scene \
    --size              512 \
    --dtype             fp32 \
    --pointmap_head     local \
    --conf_threshold    1.5 \
    --per_view_percentile 10

Batch
-----
python fast3r_experimenter.py \
    --dataset_root      /data/multi_scenes \
    --workspace         /results/all \
    --size              512 \
    --dtype             bf16 \
    --pointmap_head     global \
    --conf_threshold    1.5 \
    --per_view_percentile 10
"""

import argparse, csv, time
from pathlib import Path

import numpy as np
from PIL import Image
import open3d as o3d
import torch

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


def run_timed(func, *args, **kwargs):
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    return out, time.perf_counter() - t0


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


def process_scene(
    img_dir: Path,
    ws_dir: Path,
    size: int,
    device,
    dtype,
    pointmap_head: str,
    conf_threshold: float,
    per_view_percentile: float
):
    ws_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    T0 = time.perf_counter()

    # 1) Load model & lit_module
    model, timings["load_model"] = run_timed(
        Fast3R.from_pretrained, "jedyang97/Fast3R_ViT_Large_512"
    )
    model.to(device).eval()
    lit = MultiViewDUSt3RLitModule.load_for_inference(model).eval()

    # 2) Gather images & originals
    files = sorted(p for p in img_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"⚠️  No images found in {img_dir}")
        return
    orig_imgs = [Image.open(fp).convert("RGB") for fp in files]

    # 3) Preprocess
    views, timings["load_imgs"] = run_timed(
        load_images, [str(p) for p in files], size=size, verbose=False
    )

    # 4) Inference
    out, timings["inference"] = run_timed(
        inference, views, model, device,
        dtype=dtype, verbose=False, profiling=False
    )

    # 4.1) Drop low-confidence views exactly like the demo UI
    confs = [pred["conf"][0].mean().item() for pred in out["preds"]]
    thr_view = np.percentile(confs, per_view_percentile)
    keep = [i for i, c in enumerate(confs) if c >= thr_view]
    dropped = len(confs) - len(keep)
    if dropped:
        timings["dropped_views"] = dropped
    views = [views[i] for i in keep]
    out["preds"] = [out["preds"][i] for i in keep]

    # 5) Align (there is only local‐head alignment in the API)
    _, t_align = run_timed(
        lit.align_local_pts3d_to_global,
        preds=out["preds"],
        views=views,
        min_conf_thr_percentile=85
    )
    timings["align_pts"] = t_align

    # 6) PnP poses
    pose_batch, timings["pose_pnp"] = run_timed(
        MultiViewDUSt3RLitModule.estimate_camera_poses,
        out["preds"],
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head",
    )
    poses = pose_batch[0]
    np.save(ws_dir / "poses_c2w.npy", np.stack(poses))

    # 7) Fuse with per-point confidence and chosen head
    t_fuse = time.perf_counter()
    xyz_list, rgb_list = [], []
    for orig, pred in zip(orig_imgs, out["preds"]):
        # pick the “local” head if requested and available, else fallback
        if pointmap_head == "local" and "local_pts3d_global_est" in pred:
            xyz = pred["local_pts3d_global_est"][0].cpu().numpy()
        else:
            xyz = pred["pts3d_in_other_view"][0].cpu().numpy()

        H, W, _ = xyz.shape
        mask = np.isfinite(xyz).all(-1)

        for key in ("mask_photometric", "mask_geometric", "mask_global"):
            if key in pred:
                mask &= pred[key][0].cpu().numpy().astype(bool)

        conf_map = pred["conf"][0].cpu().numpy()
        mask &= (conf_map > conf_threshold)

        if not mask.any():
            continue

        rgb = np.asarray(orig.resize((W, H), Image.LANCZOS))
        xyz_list.append(xyz[mask])
        rgb_list.append(rgb[mask])

    xyz_all = np.concatenate(xyz_list, 0) if xyz_list else np.zeros((0, 3))
    rgb_all = np.concatenate(rgb_list, 0) if rgb_list else np.zeros((0, 3), dtype=np.uint8)
    timings["fuse_pts"] = time.perf_counter() - t_fuse

    if xyz_all.shape[0]:
        save_ply(xyz_all, rgb_all, ws_dir / "fused_fast3r.ply")

    timings["total"] = time.perf_counter() - T0

    # 8) Write CSV + console summary
    with (ws_dir / "timings.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["stage","seconds"], *timings.items()])

    print(f"\n=== {ws_dir.name} – timings (s) ===")
    for k, v in timings.items():
        print(f"{k:<12}: {v:7.2f}")
    print("=== stats ===")
    print(f"views     : {len(poses)}")
    print(f"points3D  : {xyz_all.shape[0]}")
    if xyz_all.size:
        print(f"mean |XYZ|: {np.linalg.norm(xyz_all,axis=1).mean():.2f} units")


def main():
    ap = argparse.ArgumentParser(
        description="Fast3R experimenter with per-view filtering & head choice"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images",        help="folder of images for ONE scene")
    g.add_argument("--dataset_root",  help="batch: parent folder of scenes")
    ap.add_argument(
        "--workspace", required=True,
        help="where to write outputs"
    )
    ap.add_argument(
        "--size", type=int, default=512,
        help="resize longer edge (default 512)"
    )
    ap.add_argument(
        "--dtype", choices=("fp32","bf16"), default="fp32",
        help="data type for inference"
    )
    ap.add_argument(
        "--pointmap_head", choices=("local","global"), default="local",
        help="which head to fuse (global falls back to in_other_view)"
    )
    ap.add_argument(
        "--conf_threshold", type=float, default=0,
        help="per-point confidence cutoff (default 0)"
    )
    ap.add_argument(
        "--per_view_percentile", type=float, default=0,
        help="drop bottom P%% of views by mean confidence (default 0)"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32 if args.dtype=="fp32" else torch.bfloat16

    ws = Path(args.workspace).expanduser().resolve()
    ws.mkdir(parents=True, exist_ok=True)

    if args.images:
        process_scene(
            Path(args.images), ws,
            size=args.size,
            device=device,
            dtype=dtype,
            pointmap_head=args.pointmap_head,
            conf_threshold=args.conf_threshold,
            per_view_percentile=args.per_view_percentile
        )
    else:
        for scene in sorted(Path(args.dataset_root).iterdir()):
            if scene.is_dir():
                process_scene(
                    scene, ws/scene.name,
                    size=args.size,
                    device=device,
                    dtype=dtype,
                    pointmap_head=args.pointmap_head,
                    conf_threshold=args.conf_threshold,
                    per_view_percentile=args.per_view_percentile
                )


if __name__ == "__main__":
    main()
