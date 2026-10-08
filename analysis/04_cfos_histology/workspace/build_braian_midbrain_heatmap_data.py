from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(r"F:\Sleep\Figure\Fig4_BraiAn_midbrain_heatmap_Origin_20260903")
SOURCE_DERIVED = Path(
    r"F:\Sleep\outputs\fig4_bba_projectpython_rerun\projectpython_braian_midbrain_animal_density.csv"
)
RAW_CONTROL = Path(r"F:\1.Sleep\bba_braian\Control_sum.csv")
RAW_SURGERY = Path(r"F:\1.Sleep\bba_braian\Surgery_sum.csv")
REGION_LIST = Path(r"F:\1.Sleep\bba_braian\ProjectPython\mb_subregions.txt")

ANIMALS = [
    "Control_3",
    "Control_4",
    "Control_5",
    "Control_6",
    "Surgery_2",
    "Surgery_3",
    "Surgery_4",
    "Surgery_5",
]

# Exact leaf-nucleus order used in the earlier Python matrix figure.
REGIONS = [
    "ICe", "SAG", "SCzo", "SCsg", "SCop", "NB", "PBG", "SCO",
    "SCig", "SCiw", "SCdg", "SCdw", "III", "INC", "Su3", "ND",
    "MT", "LT", "DT", "MRN", "RN", "APN", "MPT", "NOT", "NPC",
    "OP", "PPT", "RPF", "RR", "SNr", "VTA", "EW", "Pa4", "PN",
    "MA3", "IF", "IPI", "IPDM", "IPDL", "IPRL", "IPR", "IPC",
    "IPA", "IPL", "RL", "CLI", "DR", "SNc", "PPN",
]


def main() -> None:
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    (OUT / "derived_data").mkdir(parents=True, exist_ok=True)
    (OUT / "exports").mkdir(parents=True, exist_ok=True)
    (OUT / "scripts").mkdir(parents=True, exist_ok=True)

    for source in (RAW_CONTROL, RAW_SURGERY, REGION_LIST, SOURCE_DERIVED):
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / "source_data" / source.name)

    long = pd.read_csv(SOURCE_DERIVED)
    long = long[long["animal"].isin(ANIMALS) & long["region"].isin(REGIONS)].copy()
    density = long.pivot_table(index="region", columns="animal", values="density", aggfunc="first")
    density = density.reindex(index=REGIONS, columns=ANIMALS)

    means = density.mean(axis=1, skipna=True)
    sds = density.std(axis=1, skipna=True, ddof=1)
    z = density.sub(means, axis=0).div(sds.replace(0, np.nan), axis=0)
    zero_sd = sds.eq(0) & means.notna()
    if zero_sd.any():
        z.loc[zero_sd] = density.loc[zero_sd].notna().astype(float) * 0.0

    density_out = density.reset_index().rename(columns={"region": "Region"})
    z_out = z.reset_index().rename(columns={"region": "Region"})
    missing_out = density.isna().astype(int).reset_index().rename(columns={"region": "Region"})

    density_out.to_csv(OUT / "derived_data" / "midbrain_density_by_animal.csv", index=False)
    z_out.to_csv(OUT / "derived_data" / "midbrain_row_zscore_origin_input.csv", index=False)
    # Origin display matrix: insert a visual group gap and code missing cells as 0
    # only for rendering. The analytical Z-score table above retains true NaNs.
    # Native Origin Heatmap renders the final worksheet row at the top.
    # Reverse only the display table so the exported figure reads ICe -> PPN
    # from top to bottom, matching the approved Python reference.
    display = z.iloc[::-1].copy()
    display.columns = ["CON1", "CON2", "CON3", "CON4", "SUR1", "SUR2", "SUR3", "SUR4"]
    display.insert(4, " ", np.nan)
    display = display.fillna(0.0)
    display.reset_index().rename(columns={"region": "Region"}).to_csv(
        OUT / "derived_data" / "midbrain_row_zscore_origin_display.csv", index=False
    )
    missing_out.to_csv(OUT / "derived_data" / "midbrain_missingness_matrix.csv", index=False)

    summary = pd.DataFrame({
        "Region": REGIONS,
        "n_available": density.notna().sum(axis=1).values,
        "n_missing": density.isna().sum(axis=1).values,
        "row_mean_density": means.values,
        "row_sd_density": sds.values,
    })
    summary.to_csv(OUT / "derived_data" / "row_scaling_audit.csv", index=False)

    with (OUT / "source_paths.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["role", "path"])
        writer.writerow(["raw_control", str(RAW_CONTROL)])
        writer.writerow(["raw_surgery", str(RAW_SURGERY)])
        writer.writerow(["canonical_animal_level_density", str(SOURCE_DERIVED)])
        writer.writerow(["region_order_reference", str(REGION_LIST)])

    readme = f"""# BraiAn midbrain animal-by-region heatmap

- Groups: CON = Control_3–Control_6; SUR = Surgery_2–Surgery_5.
- Surgery_1_g is excluded to match the previously approved animal set.
- Value before scaling: animal-level FITC-positive cell density after hemisphere merging (BraiAn output).
- Heatmap scaling: within-region Z-score across available animals, sample SD (`ddof=1`).
- Missing measurements remain missing in the analytical table and are displayed as white cells in the Origin render.
- The analytical Z-score CSV retains missing values. The separate Origin display matrix uses 0 only as a visual code for missing cells and for the blank CON/SUR divider; consult the missingness matrix before interpreting a white cell.
- Display order: 49 leaf midbrain nuclei matching the earlier Python matrix figure (ICe at top, PPN at bottom). The analytical CSV keeps the canonical order; only the Origin display table is reversed to compensate for Origin's heatmap row direction.
- Origin color scale: diverging blue–white–red RGB palette with the displayed range derived from the row Z-scores.
- Group separator: a white divider between Control_6 and Surgery_2.
- Display labels: CON1–CON4 and SUR1–SUR4; the internal source-animal mapping is retained in the analytical tables and provenance record.

Rows with missing data: {(density.isna().any(axis=1)).sum()} of {len(REGIONS)}.
Total missing cells: {int(density.isna().sum().sum())} of {density.size}.
"""
    (OUT / "README_analysis.md").write_text(readme, encoding="utf-8")

    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    print(OUT)
    print(f"matrix={density.shape}; missing={int(density.isna().sum().sum())}")


if __name__ == "__main__":
    main()
