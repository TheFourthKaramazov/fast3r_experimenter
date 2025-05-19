#!/usr/bin/env python3
"""
colmap_experimenter.py
======================

Step-by-step COLMAP runner with per-stage timing, automatic stats,
and single-scene or batch mode.

Single scene
------------
python colmap_experimenter.py \
    --images         /path/to/scene/images \
    --workspace      /results/scene \
    --matcher        exhaustive \
    --patchmatch     high \
    --gpu            true \
    --single_camera  false

Batch (one sub-dir per scene)
-----------------------------
python colmap_experimenter.py \
    --dataset_root   /data/multiscene \
    --workspace      /results/all_scenes \
    --matcher        exhaustive \
    --patchmatch     high \
    --gpu            true \
    --single_camera  false
"""

import argparse, csv, os, subprocess, sys, time
from pathlib import Path
import numpy as np

def run(cmd: str, env=None) -> float:
    """Execute *cmd*; return elapsed seconds; abort on error."""
    print(f"\n$ {cmd}\n", flush=True)
    t0 = time.perf_counter()
    if subprocess.call(cmd, shell=True, env=env) != 0:
        sys.exit("⛔  command failed – aborting")
    return time.perf_counter() - t0

def quick_stats(sparse_root: Path) -> dict:
    """Pick the reconstruction folder with most registered images."""
    best = dict(folder="?", registered=0, points3D=0, mean_err_px=float("nan"))
    for sub in sorted(sparse_root.iterdir()):
        if not (sub / "images.txt").exists():
            continue
        imgs = sum(1 for l in (sub/"images.txt").open()
                   if l.strip() and not l.startswith("#"))
        if imgs == 0:
            continue
        pts  = sum(1 for l in (sub/"points3D.txt").open()
                   if l.strip() and not l.startswith("#"))
        errs = [float(l.split()[2]) for l in (sub/"points3D.txt").open()
                if l.strip() and not l.startswith("#")]
        if imgs > best["registered"]:
            best = dict(folder=sub.name,
                        registered=imgs,
                        points3D=pts,
                        mean_err_px=np.mean(errs) if errs else float("nan"))
    return best

def process_scene(images_dir: Path, workspace_dir: Path, *,
                  matcher: str, overlap: int, gpu: bool,
                  patchmatch: str, threads: int, single_camera: bool):

    # prepare workspace dirs
    workspace_dir.mkdir(parents=True, exist_ok=True)
    db     = workspace_dir / "database.db"
    sparse = workspace_dir / "sparse"; sparse.mkdir(exist_ok=True)
    dense  = workspace_dir / "dense";  dense.mkdir(exist_ok=True)

    timings = {}
    t_all = time.perf_counter()

    # 1) Feature extraction
    timings["feat"] = run(
        f"colmap feature_extractor "
        f"--database_path {db} "
        f"--image_path    {images_dir} "
        f"--ImageReader.single_camera {int(single_camera)} "
        f"--SiftExtraction.use_gpu {int(gpu)} "
        f"--SiftExtraction.num_threads {threads}"
    )

    # 2) Matching
    if matcher == "sequential":
        cmd = (
            f"colmap sequential_matcher --database_path {db} "
            f"--SequentialMatching.overlap {overlap} "
            f"--SiftMatching.use_gpu {int(gpu)} "
            f"--SiftMatching.num_threads {threads}"
        )
    elif matcher == "exhaustive":
        cmd = (
            f"colmap exhaustive_matcher --database_path {db} "
            f"--SiftMatching.use_gpu {int(gpu)} "
            f"--SiftMatching.num_threads {threads}"
        )
    else:  # vocab_tree
        cmd = (
            f"colmap vocab_tree_matcher --database_path {db} "
            f"--SiftMatching.use_gpu {int(gpu)} "
            f"--SiftMatching.num_threads {threads}"
        )
    timings["match"] = run(cmd)

    # 3) Sparse mapping
    timings["map"] = run(
        f"colmap mapper "
        f"--database_path {db} "
        f"--image_path    {images_dir} "
        f"--output_path   {sparse} "
        f"--Mapper.num_threads {threads} "
        f"--Mapper.ba_refine_focal_length 0"
    )

    # 4) Undistort
    timings["undist"] = run(
        f"colmap image_undistorter "
        f"--image_path  {images_dir} "
        f"--input_path  {sparse}/0 "
        f"--output_path {dense}/undist "
        f"--output_type COLMAP"
    )

    # 5) Patch-Match stereo
    geom = "true" if patchmatch=="high" else "false"
    env  = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    timings["pm"] = run(
        f"colmap patch_match_stereo "
        f"--workspace_path {dense}/undist "
        f"--PatchMatchStereo.geom_consistency {geom} "
        f"--PatchMatchStereo.filter true",
        env=env
    )

    # 6) Fusion
    timings["fusion"] = run(
        f"colmap stereo_fusion "
        f"--workspace_path {dense}/undist "
        f"--output_path    {dense/'fused.ply'}"
    )

    timings["total"] = time.perf_counter() - t_all

    # write CSV
    with (workspace_dir/"timings.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["stage","seconds"], *timings.items()])

    # console summary
    stats = quick_stats(sparse)
    print(f"\n=== {workspace_dir.name} – timings (s) ===")
    for k,v in timings.items():
        print(f"{k:<7}: {v:6.1f}")
    print("=== reconstruction stats ===")
    if stats["registered"] == 0:
        print("No sparse model found")
    else:
        print(f"model folder   : sparse/{stats['folder']}")
        for k in ("registered","points3D","mean_err_px"):
            print(f"{k:<12}: {stats[k]:.3f}")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images",      help="folder of images for ONE scene")
    g.add_argument("--dataset_root",help="batch: parent folder of scenes")
    ap.add_argument("--workspace",   required=True,
                    help="output root; per-scene dirs auto-made")
    ap.add_argument("--matcher",     default="exhaustive",
                    choices=("sequential","exhaustive","vocab_tree"))
    ap.add_argument("--overlap",     type=int, default=5,
                    help="±k for sequential matcher")
    ap.add_argument("--patchmatch",  choices=("fast","high"), default="high",
                    help="'high' = geom_consistency on")
    ap.add_argument("--single_camera", type=lambda x: x.lower()=="true",
                    default=False,
                    help="true = force single intrinsics (identical resolution)")
    ap.add_argument("--gpu",         type=lambda x: x.lower()=="true",
                    default=True)
    ap.add_argument("--threads",     type=int, default=os.cpu_count())
    args = ap.parse_args()

    ws_root = Path(args.workspace).resolve()
    ws_root.mkdir(parents=True, exist_ok=True)

    if args.images:
        process_scene(Path(args.images).resolve(), ws_root,
                      matcher=args.matcher,
                      overlap=args.overlap,
                      gpu=args.gpu,
                      patchmatch=args.patchmatch,
                      threads=args.threads,
                      single_camera=args.single_camera)
    else:
        ds = Path(args.dataset_root).resolve()
        scenes = [d for d in ds.iterdir() if d.is_dir()]
        if not scenes:
            sys.exit("No sub-directories found in dataset_root")
        for scene in scenes:
            out = ws_root/scene.name
            process_scene(scene, out,
                          matcher=args.matcher,
                          overlap=args.overlap,
                          gpu=args.gpu,
                          patchmatch=args.patchmatch,
                          threads=args.threads,
                          single_camera=args.single_camera)

if __name__=="__main__":
    main()
