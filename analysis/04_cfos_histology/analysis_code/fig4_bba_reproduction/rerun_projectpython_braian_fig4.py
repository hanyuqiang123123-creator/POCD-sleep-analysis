import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(r"F:\Sleep\bba_braian\Qupath_projects")
PROJECT_PYTHON = Path(r"F:\Sleep\bba_braian\ProjectPython")
VENVSITE = PROJECT_PYTHON / ".venv" / "Lib" / "site-packages"
OUTDIR = Path(r"F:\Sleep\outputs\fig4_bba_projectpython_rerun")


def bootstrap_projectpython_venv() -> None:
    """Use the original ProjectPython site-packages with a Python 3.12 runtime."""
    for rel in [
        "pywin32_system32",
        "win32",
        "win32/lib",
        "vtkmodules",
        "vtk.libs",
    ]:
        dll_dir = VENVSITE / rel
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

    for path in [
        VENVSITE,
        VENVSITE / "win32",
        VENVSITE / "win32" / "lib",
        VENVSITE / "win32com",
    ]:
        sys.path.insert(0, str(path))


def get_mb_leaf_regions(root_dir: Path) -> list[str]:
    ontology_path = root_dir / "allen_mouse_10um-Ontology.json"
    with ontology_path.open("r", encoding="utf-8") as f:
        ontology = json.load(f)

    def walk(node: dict, found: bool = False) -> list[str]:
        acronym = node.get("data", {}).get("acronym", "")
        children = node.get("children", [])
        now_found = found or acronym == "MB"
        regions = []
        if now_found and acronym and not children:
            regions.append(acronym)
        for child in children:
            regions.extend(walk(child, now_found))
        return regions

    return walk(ontology.get("root", ontology))


def main() -> None:
    bootstrap_projectpython_venv()

    import pandas as pd
    from scipy import stats
    import braian
    import braian.config

    OUTDIR.mkdir(parents=True, exist_ok=True)
    os.environ["BRAINRENDER_DEBUG"] = "false"

    midbrain_regions = get_mb_leaf_regions(ROOT)
    config = braian.config.BraiAnConfig(ROOT / "config_example.yml", tempfile.gettempdir())
    experiment_sliced = config.from_qupath(sliced=True)

    rows = []
    loaded_animals = []
    merged_groups = []
    for group in experiment_sliced.groups:
        loaded_animals.append(
            {"group": group.name, "animals": ";".join(animal.name for animal in group.animals)}
        )
        merged_animals = []
        for sliced_animal in group.animals:
            merged_sliced = braian.SlicedBrain.merge_hemispheres(sliced_animal)
            animal_density = braian.AnimalBrain.from_slices(
                merged_sliced,
                metric=braian.SliceMetrics.SUM,
                densities=True,
            )
            merged_animals.append(animal_density)
        merged_groups.append(braian.AnimalGroup(group.name, merged_animals, "density"))

    for group in merged_groups:
        for animal in group.animals:
            marker = "FITC"
            if marker not in animal.markers:
                continue
            marker_data = animal[marker]
            for region in animal.regions:
                if region not in midbrain_regions:
                    continue
                try:
                    density = marker_data[region]
                except (KeyError, IndexError):
                    continue
                rows.append(
                    {
                        "group": group.name,
                        "animal": animal.name,
                        "region": region,
                        "density": density,
                        "included_density_gt0": bool(pd.notna(density) and density > 0),
                    }
                )

    source = pd.DataFrame(rows)
    source.to_csv(OUTDIR / "projectpython_braian_midbrain_animal_density.csv", index=False)
    pd.DataFrame(loaded_animals).to_csv(OUTDIR / "projectpython_braian_loaded_animals.csv", index=False)

    included = source[source["included_density_gt0"]].copy()
    stats_rows = []
    filtered_rows = []
    for region in midbrain_regions:
        control = included[(included["group"] == "Control") & (included["region"] == region)]["density"].astype(float)
        surgery = included[(included["group"] == "Surgery") & (included["region"] == region)]["density"].astype(float)
        if control.empty or surgery.empty:
            continue

        total_n = len(control) + len(surgery)
        if len(control) < 2 or len(surgery) < 2 or total_n < 4:
            filtered_rows.append(
                {
                    "region": region,
                    "control_n": len(control),
                    "surgery_n": len(surgery),
                    "total_n": total_n,
                }
            )
            continue

        t_stat, p_value = stats.ttest_ind(surgery.to_numpy(), control.to_numpy())
        control_mean = control.mean()
        surgery_mean = surgery.mean()
        fold_change = surgery_mean / control_mean if control_mean else np.nan
        log2_fc = np.log2(fold_change) if fold_change and fold_change > 0 else np.nan
        stats_rows.append(
            {
                "region": region,
                "control_n": len(control),
                "surgery_n": len(surgery),
                "control_mean": control_mean,
                "surgery_mean": surgery_mean,
                "log2_fc": log2_fc,
                "p_value": p_value,
                "neg_log10_p": -np.log10(p_value) if p_value > 0 else np.inf,
                "t_statistic": t_stat,
                "significance": "Up (Surgery)"
                if p_value < 0.05 and log2_fc > 0
                else ("Down (Surgery)" if p_value < 0.05 and log2_fc < 0 else "Not Significant"),
            }
        )

    result = pd.DataFrame(stats_rows)
    filtered = pd.DataFrame(filtered_rows)
    result.to_csv(OUTDIR / "projectpython_braian_volcano_results_rerun.csv", index=False)
    filtered.to_csv(OUTDIR / "projectpython_braian_filtered_regions.csv", index=False)

    historical_path = ROOT / "volcano_results.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path)
        compare = result.merge(
            historical,
            on="region",
            how="outer",
            suffixes=("_rerun", "_historical"),
            indicator=True,
        )
        compare.to_csv(OUTDIR / "projectpython_braian_rerun_vs_historical.csv", index=False)

    target_regions = ["MT", "RPF", "VTA", "EW", "RL"]
    print("Loaded animals:")
    for item in loaded_animals:
        print(f"  {item['group']}: {item['animals']}")
    print("\nTarget regions:")
    cols = ["region", "control_n", "surgery_n", "control_mean", "surgery_mean", "log2_fc", "p_value", "significance"]
    print(result[result["region"].isin(target_regions)][cols].to_string(index=False))
    print(f"\nWrote: {OUTDIR}")


if __name__ == "__main__":
    main()
