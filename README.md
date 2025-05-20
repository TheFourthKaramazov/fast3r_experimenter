# [Fast3R](https://github.com/facebookresearch/fast3r) Experimenter

A Python CLI tool to run Fast3R over single or multiple scenes with timings, PnP poses, and coloured fused point clouds, with optional view‐dropping and pointmap‐head choice.

## Requirements

- Python 3.7+
- PyTorch
- numpy
- Pillow
- Open3D
- fast3r (and its dust3r dependencies)

## Installation

```bash
pip install torch numpy Pillow open3d
```
Please visit the [official Fast3r implementation repo](https://github.com/facebookresearch/fast3r) for installation instructions.

## Usage

### Single Scene

```bash



python fast3r_experimenter.py \
  --images              /path/to/scene/images \
  --workspace           /results/scene \
  --size                512 \
  --dtype               fp32 \
  --pointmap_head       local \
  --conf_threshold      0 \
  --per_view_percentile 0

```


### Batch

```bash



python fast3r_experimenter.py \
  --dataset_root        /data/multi_scenes \
  --workspace           /results/all \
  --size                512 \
  --dtype               bf16 \
  --pointmap_head       global \
  --conf_threshold      0 \
  --per_view_percentile 0

```
## Arguments

| Flag                   | Required?                            | Default | Description                                                                                       |
|------------------------|--------------------------------------|---------|---------------------------------------------------------------------------------------------------|
| `--images`             | either this or `--dataset_root`      | —       | Folder containing your scene images (single-scene mode)                                           |
| `--dataset_root`       | either this or `--images`            | —       | Parent folder of scene subdirectories (batch mode)                                                |
| `--workspace`          | yes                                  | —       | Where to write per-scene outputs                                                                  |
| `--size`               | no                                   | `512`   | Resize longer image edge to this value                                                            |
| `--dtype`              | no                                   | `fp32`  | Inference data type: `fp32` or `bf16`                                                              |
| `--pointmap_head`      | no                                   | `local` | Which point-map head to fuse: `local` (uses `local_pts3d_global_est`) or `global` (falls back to `pts3d_in_other_view`) |
| `--conf_threshold`     | no                                   | `0`     | Per-point confidence cutoff (points with `conf <= threshold` are dropped)                         |
| `--per_view_percentile`| no                                   | `0`     | Drop bottom _P_% of views by mean confidence (views with mean `conf` below this percentile are skipped) |

## Outputs (per scene)

- **`fused_fast3r.ply`**  
  Coloured fused point cloud (if any points remain after masking).

- **`poses_c2w.npy`**  
  NumPy array of camera-to-world poses.

- **`timings.csv`**  
  CSV listing elapsed seconds for each pipeline stage (`load_model`, `load_imgs`, `inference`, `dropped_views`, `align_pts`, `pose_pnp`, `fuse_pts`, `total`).

- **Console summary**  
  Prints number of views processed, total 3D points, and mean distance of points.


