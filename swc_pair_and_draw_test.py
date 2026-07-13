from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_matching import run_matching
from standard_draw import run_standard_draw
from swc_descriptor_batch import batch_run


def _sort_descriptor_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "neuron_id" not in df.columns:
        raise ValueError("Descriptor dataframe is missing neuron_id column")
    return df.sort_values("neuron_id", kind="mergesort").reset_index(drop=True)


def _assert_same_descriptor_table(temp_parquet: Path, ref_parquet: Path, label: str) -> None:
    if not ref_parquet.exists():
        raise FileNotFoundError(f"Missing reference descriptor parquet: {ref_parquet}")

    temp_df = _sort_descriptor_frame(pd.read_parquet(temp_parquet))
    ref_df = _sort_descriptor_frame(pd.read_parquet(ref_parquet))

    if list(temp_df.columns) != list(ref_df.columns):
        raise AssertionError(f"{label}: column mismatch\nTEMP={list(temp_df.columns)}\nREF={list(ref_df.columns)}")

    if temp_df.shape != ref_df.shape:
        raise AssertionError(f"{label}: shape mismatch TEMP={temp_df.shape} REF={ref_df.shape}")

    if not temp_df["neuron_id"].astype(str).equals(ref_df["neuron_id"].astype(str)):
        raise AssertionError(f"{label}: neuron_id order/content mismatch")

    numeric_cols = [c for c in temp_df.columns if c != "neuron_id"]
    for col in numeric_cols:
        temp_vals = temp_df[col].to_numpy()
        ref_vals = ref_df[col].to_numpy()
        if np.issubdtype(temp_vals.dtype, np.number) and np.issubdtype(ref_vals.dtype, np.number):
            np.testing.assert_allclose(temp_vals, ref_vals, rtol=0.0, atol=0.0, err_msg=f"{label}: column {col}")
        else:
            if not np.array_equal(temp_vals, ref_vals):
                raise AssertionError(f"{label}: column {col} mismatch")


def _assert_same_npy(temp_path: Path, ref_path: Path, label: str) -> None:
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference npy file: {ref_path}")

    temp_arr = np.load(temp_path, allow_pickle=True)
    ref_arr = np.load(ref_path, allow_pickle=True)

    if temp_arr.dtype.kind in {"U", "S", "O"} or ref_arr.dtype.kind in {"U", "S", "O"}:
        if not np.array_equal(temp_arr.astype(str), ref_arr.astype(str)):
            raise AssertionError(f"{label}: string array mismatch")
        return

    np.testing.assert_allclose(temp_arr, ref_arr, rtol=0.0, atol=0.0, err_msg=label)


def _assert_same_pairs_csv(temp_csv: Path, ref_csv: Path) -> None:
    if not ref_csv.exists():
        raise FileNotFoundError(f"Missing reference pairs csv: {ref_csv}")

    temp_df = pd.read_csv(temp_csv, dtype=str).sort_values(["fc_id", "em_id"], kind="mergesort").reset_index(drop=True)
    ref_df = pd.read_csv(ref_csv, dtype=str).sort_values(["fc_id", "em_id"], kind="mergesort").reset_index(drop=True)

    if list(temp_df.columns) != list(ref_df.columns):
        raise AssertionError(f"Pairs CSV column mismatch\nTEMP={list(temp_df.columns)}\nREF={list(ref_df.columns)}")
    if temp_df.shape != ref_df.shape:
        raise AssertionError(f"Pairs CSV shape mismatch TEMP={temp_df.shape} REF={ref_df.shape}")
    if not temp_df.equals(ref_df):
        raise AssertionError("Pairs CSV content mismatch")


def _assert_pairs_schema(temp_csv: Path) -> None:
    if not temp_csv.exists():
        raise FileNotFoundError(f"Missing generated pairs csv: {temp_csv}")

    temp_df = pd.read_csv(temp_csv, dtype=str)
    expected_cols = ["fc_id", "em_id"]
    if list(temp_df.columns) != expected_cols:
        raise AssertionError(f"Pairs CSV schema mismatch TEMP={list(temp_df.columns)} EXPECTED={expected_cols}")
    if temp_df.empty:
        raise AssertionError("Pairs CSV is empty")


def _assert_same_views(temp_views_dir: Path, ref_views_dir: Path) -> None:
    temp_files = sorted(temp_views_dir.glob("*_views.npz"))
    ref_files = sorted(ref_views_dir.glob("*_views.npz"))

    temp_names = [p.name for p in temp_files]
    ref_names = [p.name for p in ref_files]
    if temp_names != ref_names:
        raise AssertionError(
            f"View filename set mismatch\nTEMP={temp_names[:20]}\nREF={ref_names[:20]}"
        )

    for temp_file in temp_files:
        ref_file = ref_views_dir / temp_file.name
        if not ref_file.exists():
            raise FileNotFoundError(f"Missing reference view file: {ref_file}")

        temp_npz = np.load(temp_file, allow_pickle=True)
        ref_npz = np.load(ref_file, allow_pickle=True)

        if not np.array_equal(temp_npz["nid"].astype(str), ref_npz["nid"].astype(str)):
            raise AssertionError(f"View nid mismatch: {temp_file.name}")
        if int(temp_npz["grid_size"]) != int(ref_npz["grid_size"]):
            raise AssertionError(f"Grid size mismatch: {temp_file.name}")
        np.testing.assert_array_equal(temp_npz["views"], ref_npz["views"], err_msg=temp_file.name)


def run_pipeline_and_compare(
    *,
    fc_swc_dir: str | Path,
    em_swc_dir: str | Path,
    reference_root: str | Path = "./data",
    centroid_th: float = 100.0,
    ratio_th: float = 0.4,
    pairs_ref: str | Path | None = None,
    recursive: bool = True,
    fail_fast: bool = False,
    scale_um_per_px: float = 5.0,
    normalize: str = "p99",
) -> None:
    reference_root = Path(reference_root)
    fc_desc_ref = reference_root / "descriptors_FC" / "descriptors_FC.parquet"
    em_desc_ref = reference_root / "descriptors_EM" / "descriptors_EM.parquet"
    fc_desc_ref_npy = {
        "centroids": reference_root / "descriptors_FC" / "centroids_FC.npy",
        "eigvecs": reference_root / "descriptors_FC" / "eigvecs_FC.npy",
        "ratios": reference_root / "descriptors_FC" / "eigvals_ratio_FC.npy",
        "ids": reference_root / "descriptors_FC" / "neuron_ids_FC.npy",
    }
    em_desc_ref_npy = {
        "centroids": reference_root / "descriptors_EM" / "centroids_EM.npy",
        "eigvecs": reference_root / "descriptors_EM" / "eigvecs_EM.npy",
        "ratios": reference_root / "descriptors_EM" / "eigvals_ratio_EM.npy",
        "ids": reference_root / "descriptors_EM" / "neuron_ids_EM.npy",
    }
    fc_views_ref = reference_root / "standard_views" / "FC"
    em_views_ref = reference_root / "standard_views" / "EM"

    with tempfile.TemporaryDirectory(prefix="swc_pair_and_draw_test_") as tmp_root:
        tmp_root = Path(tmp_root)
        tmp_desc_root = tmp_root / "descriptors"
        tmp_pairs_root = tmp_root / "pairs_label"
        tmp_views_root = tmp_root / "standard_views"
        tmp_desc_root.mkdir(parents=True, exist_ok=True)

        print("[1/4] Building FC descriptors into temp dir")
        fc_desc_dir = tmp_desc_root / "descriptors_FC"
        batch_run(fc_swc_dir, fc_desc_dir, source="FC", recursive=recursive, fail_fast=fail_fast)

        print("[2/4] Building EM descriptors into temp dir")
        em_desc_dir = tmp_desc_root / "descriptors_EM"
        batch_run(em_swc_dir, em_desc_dir, source="EM", recursive=recursive, fail_fast=fail_fast)

        print("[compare] Checking FC descriptors against reference")
        _assert_same_descriptor_table(fc_desc_dir / "descriptors_FC.parquet", fc_desc_ref, "FC descriptors")
        _assert_same_npy(fc_desc_dir / "centroids_FC.npy", fc_desc_ref_npy["centroids"], "FC centroids")
        _assert_same_npy(fc_desc_dir / "eigvecs_FC.npy", fc_desc_ref_npy["eigvecs"], "FC eigvecs")
        _assert_same_npy(fc_desc_dir / "eigvals_ratio_FC.npy", fc_desc_ref_npy["ratios"], "FC eigvals_ratio")
        _assert_same_npy(fc_desc_dir / "neuron_ids_FC.npy", fc_desc_ref_npy["ids"], "FC neuron_ids")

        print("[compare] Checking EM descriptors against reference")
        _assert_same_descriptor_table(em_desc_dir / "descriptors_EM.parquet", em_desc_ref, "EM descriptors")
        _assert_same_npy(em_desc_dir / "centroids_EM.npy", em_desc_ref_npy["centroids"], "EM centroids")
        _assert_same_npy(em_desc_dir / "eigvecs_EM.npy", em_desc_ref_npy["eigvecs"], "EM eigvecs")
        _assert_same_npy(em_desc_dir / "eigvals_ratio_EM.npy", em_desc_ref_npy["ratios"], "EM eigvals_ratio")
        _assert_same_npy(em_desc_dir / "neuron_ids_EM.npy", em_desc_ref_npy["ids"], "EM neuron_ids")

        print("[3/4] Matching candidates into temp dir")
        pairs_csv = run_matching(
            fc_dir=fc_desc_dir,
            em_dir=em_desc_dir,
            out_dir=tmp_pairs_root,
            centroid_th=centroid_th,
            ratio_th=ratio_th,
        )

        print("[compare] Checking pairs CSV against reference")
        if pairs_ref is None:
            _assert_pairs_schema(pairs_csv)
        else:
            pairs_ref = Path(pairs_ref)
            _assert_same_pairs_csv(pairs_csv, pairs_ref)

        print("[4/4] Rendering FC and EM views into temp dir")
        fc_views_dir = tmp_views_root / "FC"
        em_views_dir = tmp_views_root / "EM"

        run_standard_draw(
            swc_dir=fc_swc_dir,
            neuron_list=pairs_csv,
            output_dir=fc_views_dir,
            csv_id_col="fc_id",
            scale_um_per_px=scale_um_per_px,
            normalize=normalize,
            skip_existing=False,
        )
        run_standard_draw(
            swc_dir=em_swc_dir,
            neuron_list=pairs_csv,
            output_dir=em_views_dir,
            csv_id_col="em_id",
            scale_um_per_px=scale_um_per_px,
            normalize=normalize,
            skip_existing=False,
        )

        print("[compare] Checking FC views against reference")
        _assert_same_views(fc_views_dir, fc_views_ref)
        print("[compare] Checking EM views against reference")
        _assert_same_views(em_views_dir, em_views_ref)

    print("[done] Temporary run matched the reference outputs and has been cleaned up.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full pipeline in temp dirs and compare results against existing reference outputs.")
    ap.add_argument("--fc_swc_dir", default="./data/SWC/FC", help="FC SWC directory")
    ap.add_argument("--em_swc_dir", default="./data/SWC/EM", help="EM SWC directory")
    ap.add_argument("--reference_root", default="./data", help="Reference root containing descriptors_*, pairs_label, standard_views")
    ap.add_argument("--pairs_ref", default="", help="Optional archived pairs CSV for strict comparison; if omitted, only validate schema")
    ap.add_argument("--centroid_th", type=float, default=100.0, help="Centroid distance threshold")
    ap.add_argument("--ratio_th", type=float, default=0.4, help="(r21,r31) distance threshold")
    ap.add_argument("--no-recursive", action="store_true", help="Only scan top level SWC files")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first SWC error")
    ap.add_argument("--scale_um_per_px", type=float, default=5.0, help="Rendering scale in micrometers per pixel")
    ap.add_argument("--normalize", choices=["max", "p99"], default="p99", help="Normalization mode")
    args = ap.parse_args()

    run_pipeline_and_compare(
        fc_swc_dir=args.fc_swc_dir,
        em_swc_dir=args.em_swc_dir,
        reference_root=args.reference_root,
        centroid_th=args.centroid_th,
        ratio_th=args.ratio_th,
        pairs_ref=args.pairs_ref or None,
        recursive=not args.no_recursive,
        fail_fast=args.fail_fast,
        scale_um_per_px=args.scale_um_per_px,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()