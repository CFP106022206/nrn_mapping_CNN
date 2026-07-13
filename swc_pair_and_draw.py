from __future__ import annotations

import argparse
from pathlib import Path

from candidate_matching import run_matching
from standard_draw import run_standard_draw
from swc_descriptor_batch import batch_run


def run_pipeline(
    *,
    fc_swc_dir: str | Path,
    em_swc_dir: str | Path,
    descriptor_root: str | Path = "./data",
    pairs_out_dir: str | Path = "./data/pairs_label/",
    views_root: str | Path = "./data/standard_views/",
    centroid_th: float = 100.0,
    ratio_th: float = 0.4,
    recursive: bool = True,
    fail_fast: bool = False,
    scale_um_per_px: float = 5.0,
    normalize: str = "p99",
    skip_existing: bool = True,
) -> Path:
    descriptor_root = Path(descriptor_root)
    fc_desc_dir = descriptor_root / "descriptors_FC"
    em_desc_dir = descriptor_root / "descriptors_EM"

    print("[1/4] Building FC descriptors")
    batch_run(fc_swc_dir, fc_desc_dir, source="FC", recursive=recursive, fail_fast=fail_fast)

    print("[2/4] Building EM descriptors")
    batch_run(em_swc_dir, em_desc_dir, source="EM", recursive=recursive, fail_fast=fail_fast)

    print("[3/4] Matching candidates")
    pairs_csv = run_matching(
        fc_dir=fc_desc_dir,
        em_dir=em_desc_dir,
        out_dir=pairs_out_dir,
        centroid_th=centroid_th,
        ratio_th=ratio_th,
    )

    views_root = Path(views_root)
    fc_views_dir = views_root / "FC"
    em_views_dir = views_root / "EM"

    print("[4/4] Rendering FC views")
    run_standard_draw(
        swc_dir=fc_swc_dir,
        neuron_list=pairs_csv,
        output_dir=fc_views_dir,
        csv_id_col="fc_id",
        scale_um_per_px=scale_um_per_px,
        normalize=normalize,
        skip_existing=skip_existing,
    )

    print("[4/4] Rendering EM views")
    run_standard_draw(
        swc_dir=em_swc_dir,
        neuron_list=pairs_csv,
        output_dir=em_views_dir,
        csv_id_col="em_id",
        scale_um_per_px=scale_um_per_px,
        normalize=normalize,
        skip_existing=skip_existing,
    )

    return pairs_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full SWC -> descriptor -> matching -> standard view pipeline.")

    # Stage 1: descriptor extraction inputs
    ap.add_argument("--fc_swc_dir", default="./data/SWC/FC", help="FC SWC directory")
    ap.add_argument("--em_swc_dir", default="./data/SWC/EM", help="EM SWC directory")
    ap.add_argument("--descriptor_root", default="./data", help="Root folder for descriptors_FC / descriptors_EM")
    ap.add_argument("--no-recursive", action="store_true", help="Only scan the top level of each SWC directory")
    ap.add_argument("--fail-fast", action="store_true", help="Stop immediately if one SWC fails during descriptor extraction")

    # Stage 2: candidate matching thresholds and output
    ap.add_argument("--pairs_out_dir", default="./data/pairs_label/", help="Folder for candidate pairs CSV")
    ap.add_argument("--centroid_th", type=float, default=100.0, help="Centroid distance threshold for candidate filtering")
    ap.add_argument("--ratio_th", type=float, default=0.4, help="(r21,r31) ratio distance threshold for candidate filtering")

    # Stage 3: rendering settings
    ap.add_argument("--views_root", default="./data/standard_views/", help="Root folder for rendered views")
    ap.add_argument("--scale_um_per_px", type=float, default=5.0, help="Rendering scale in micrometers per pixel")
    ap.add_argument("--normalize", choices=["max", "p99"], default="p99", help="Normalization mode for rendered views")
    ap.add_argument("--no-skip-existing", action="store_true", help="Re-render views even if output files already exist")

    args = ap.parse_args()

    run_pipeline(
        fc_swc_dir=args.fc_swc_dir,
        em_swc_dir=args.em_swc_dir,
        descriptor_root=args.descriptor_root,
        pairs_out_dir=args.pairs_out_dir,
        views_root=args.views_root,
        centroid_th=args.centroid_th,
        ratio_th=args.ratio_th,
        recursive=not args.no_recursive,
        fail_fast=args.fail_fast,
        scale_um_per_px=args.scale_um_per_px,
        normalize=args.normalize,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()