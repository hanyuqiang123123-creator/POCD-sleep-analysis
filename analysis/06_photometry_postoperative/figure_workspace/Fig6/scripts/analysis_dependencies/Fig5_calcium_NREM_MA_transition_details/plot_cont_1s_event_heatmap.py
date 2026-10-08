"""Draw the Cont NREM-to-MA event heatmap using manually refined 1-s scores."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_nrem_ma_transition_details import BIN_CENTERS, OUT, load_event_curves, save_figure
from analyze_nrem_ma_transitions import ROOT


PROVENANCE_ROOT = Path(r"F:\Sleep\outputs\figure_provenance")
FIGURE_ID = "fig5_calcium_nrem_ma_transition_Cont_1s_heatmap"
SUR_CURRENT_DATA = OUT / "data/SUR_1s_alignment_corrected_event_level_transition_timecourses.csv"
CONT_1S_RECORDS = [
    {"group": "Cont", "animal": "Cont1", "plot": ROOT / "Cont/1/Plot.mat", "stage": ROOT / "Cont/1/EEG/stages_aligned_epoch_1s.db3", "stage_origin": 0.0, "offset": 0.0, "calcium_shift_s": -2.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont2", "plot": ROOT / "Cont/2/Plot.mat", "stage": ROOT / "Cont/2/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 29.786, "calcium_shift_s": 6.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont3", "plot": ROOT / "Cont/3/Plot.mat", "stage": ROOT / "Cont/3/EEG/stages_aligned_epoch_1s.db3", "stage_origin": 0.0, "offset": 0.0, "calcium_shift_s": 3.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont4", "plot": ROOT / "Cont/4/Plot.mat", "stage": ROOT / "Cont/4/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 29.544, "calcium_shift_s": 3.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont5", "plot": ROOT / "Cont/5/Plot.mat", "stage": ROOT / "Cont/5/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 29.827, "calcium_shift_s": 3.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont6", "plot": ROOT / "Cont/6/Plot.mat", "stage": ROOT / "Cont/6/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 29.231, "calcium_shift_s": -4.0, "expected_stage_epoch_s": 1.0},
    {"group": "Cont", "animal": "Cont7", "plot": ROOT / "Cont/7/Plot.mat", "stage": ROOT / "Cont/7/EEG/stages_aligned.db3", "stage_origin": 0.0, "offset": 0.0, "calcium_shift_s": 3.0, "expected_stage_epoch_s": 1.0},
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "pdf.compression": 0,
    "font.size": 7,
    "axes.linewidth": 0.75,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for folder in (OUT / "panels", OUT / "data", OUT / "scripts", OUT / "documentation", PROVENANCE_ROOT):
        folder.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    epochs: dict[str, float] = {}
    for record in CONT_1S_RECORDS:
        event_rows, epoch_s = load_event_curves(record)
        rows.extend(event_rows)
        epochs[str(record["animal"])] = epoch_s
        for role in ("plot", "stage"):
            path = Path(record[role])
            sources.append({"animal": record["animal"], "role": role, "path": str(path), "sha256": sha256(path)})

    event_df = pd.DataFrame(rows)
    pivot = event_df.pivot(
        index=["animal", "event_index", "onset_s", "ma_duration_s", "calcium_shift_s"],
        columns="time_s",
        values="delta_signal",
    ).sort_index(level=[0, 2])
    row_manifest = pivot.reset_index()[["animal", "event_index", "onset_s", "ma_duration_s", "calcium_shift_s"]]
    matrix = pivot.to_numpy(float)

    if not SUR_CURRENT_DATA.exists():
        raise FileNotFoundError(f"Current SUR comparison data not found: {SUR_CURRENT_DATA}")
    sur_values = pd.read_csv(SUR_CURRENT_DATA)["delta_signal"].to_numpy(float)
    limit = float(np.nanpercentile(np.abs(sur_values), 98))

    fig, ax = plt.subplots(figsize=(3.65, 3.35))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=[BIN_CENTERS[0] - 0.5, BIN_CENTERS[-1] + 0.5, len(matrix), 0],
    )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    group_sizes = row_manifest.groupby("animal", sort=False).size()
    boundaries = group_sizes.cumsum().to_numpy()
    starts = np.r_[0, boundaries[:-1]]
    centers = (starts + boundaries) / 2
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="white", lw=0.9)
    ax.set_yticks(centers, group_sizes.index)
    ax.set_xlabel("Time from MA onset (s)")
    ax.set_ylabel("Cont mouse / eligible events")
    ax.set_title("Cont NREM-to-MA calcium events (1-s scoring; corrected alignment)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.18, fraction=0.07)
    colorbar.set_label("Δ signal vs −20 to −10 s baseline (stored units)")
    fig.text(0.99, 0.015, f"n=7 mice; {len(matrix)} events; scale matched to SUR", ha="right", va="bottom", fontsize=6)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.22)

    base = OUT / "panels" / "panel_f_Cont_event_heatmap_1s_alignment_corrected"
    outputs = save_figure(fig, base, tiff=True)
    plt.close(fig)

    event_path = OUT / "data" / "Cont_1s_alignment_corrected_event_level_transition_timecourses.csv"
    rows_path = OUT / "data" / "Cont_1s_alignment_corrected_event_heatmap_row_manifest.csv"
    source_path = OUT / "data" / "Cont_1s_alignment_corrected_heatmap_source_manifest.csv"
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    row_manifest.to_csv(rows_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sources).to_csv(source_path, index=False, encoding="utf-8-sig")

    script_copy = OUT / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy)
    shifts = {str(record["animal"]): float(record["calcium_shift_s"]) for record in CONT_1S_RECORDS}
    readme_path = OUT / "documentation" / "Cont_1s_alignment_corrected_heatmap_README.md"
    readme_path.write_text(
        f"""# Cont 1-s scoring event heatmap

- Purpose: visualize Cont NREM-to-MA calcium events using the manually refined 1-s scores.
- Sample: seven independent Cont mice; {len(matrix)} eligible events.
- Score resolution: 1 s in every mouse ({json.dumps(epochs, ensure_ascii=False)}).
- Alignment: established Plot–EEG offsets retained; user-specified calcium shifts (positive = right/later): {json.dumps(shifts, ensure_ascii=False)} s.
- Event definition: Wake ≤20 s flanked by NREM; ≥20 s continuous preceding NREM.
- Window and baseline: −20 to +30 s; event-specific −20 to −10 s baseline; 1-s calcium bins.
- Color scale: ±{limit:.4g} stored units, exactly matched to the current corrected SUR heatmap.
- Inference: none. Event rows are descriptive repeated observations, not independent n.
""",
        encoding="utf-8",
    )

    provenance = f"""# {FIGURE_ID}

- figure_title: Cont NREM-to-MA calcium event heatmap using 1-s scoring and corrected alignment
- manuscript_context: Candidate detail panel for visual comparison with the corrected SUR 1-s heatmap.
- status: candidate
- generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}
- sample_definition: seven independent Cont mice; {len(matrix)} descriptive event rows.
- analysis_window: −20 to +30 s around MA onset; −20 to −10 s baseline; 1-s scoring and calcium bins.
- analysis_steps: Wake ≤20 s flanked by NREM; ≥20 s preceding NREM; calcium shifts Cont1 −2 s, Cont2 +6 s, Cont3 +3 s, Cont4 +3 s, Cont5 +3 s, Cont6 −4 s, Cont7 +3 s.
- statistics: none in this heatmap.
- scripts: {script_copy}
- source_data: {event_path} | {rows_path} | {source_path}
- outputs: {' | '.join(str(path) for path in outputs)}
- review_notes: shared color scale with current SUR heatmap; event rows are not inferential replicates.
"""
    provenance_path = PROVENANCE_ROOT / f"{FIGURE_ID}_provenance.md"
    provenance_path.write_text(provenance, encoding="utf-8")
    shutil.copy2(provenance_path, OUT / "documentation" / provenance_path.name)

    manifest_path = OUT / "documentation" / "Cont_1s_alignment_corrected_heatmap_outputs_manifest.csv"
    manifest_files = [*outputs, event_path, rows_path, source_path, script_copy, readme_path, provenance_path]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "path", "status", "sha256"])
        writer.writeheader()
        for path in manifest_files:
            writer.writerow({"type": "figure" if path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff"} else "supporting", "path": str(path), "status": "current", "sha256": sha256(path)})

    for source in sources:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise AssertionError(f"Source changed: {source['path']}")
    print(json.dumps({"output": str(base.with_suffix('.png')), "events": len(matrix), "events_per_mouse": group_sizes.to_dict(), "score_epochs_s": epochs, "calcium_shift_s": shifts, "shared_color_limit": limit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
