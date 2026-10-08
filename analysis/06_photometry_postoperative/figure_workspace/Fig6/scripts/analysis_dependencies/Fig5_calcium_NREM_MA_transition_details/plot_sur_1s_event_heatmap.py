"""Draw the SUR NREM-to-MA event heatmap using manually refined 1-s scores."""
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
FIGURE_ID = "fig5_calcium_nrem_ma_transition_SUR_1s_heatmap"
SUR_1S_RECORDS = [
    {"group": "SUR", "animal": "SUR1", "plot": ROOT / "SUR/1/Aligned_EEG_56m20s_to84m/Plot.mat", "stage": ROOT / "SUR/1/EEG/scores_epoch_1s.db3", "stage_origin": 3380.0, "offset": 0.0, "calcium_shift_s": 0.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR2", "plot": ROOT / "SUR/2/Plot.mat", "stage": ROOT / "SUR/2/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 11.568, "calcium_shift_s": 4.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR3", "plot": ROOT / "SUR/3/Aligned_EEGminus21s/Plot.mat", "stage": ROOT / "SUR/3/Aligned_EEGminus21s/stages_aligned.db3", "stage_origin": 0.0, "offset": 0.0, "calcium_shift_s": 6.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR4", "plot": ROOT / "SUR/4/Plot.mat", "stage": ROOT / "SUR/4/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 7.306, "calcium_shift_s": 8.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR5", "plot": ROOT / "SUR/5/Plot.mat", "stage": ROOT / "SUR/5/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 4.047, "calcium_shift_s": 3.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR6", "plot": ROOT / "SUR/6_/Plot.mat", "stage": ROOT / "SUR/6_/EEG/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 4.046, "calcium_shift_s": 4.0, "expected_stage_epoch_s": 1.0},
    {"group": "SUR", "animal": "SUR7", "plot": ROOT / "SUR/7/Plot.mat", "stage": ROOT / "SUR/7/EEG1/scores_epoch_1s.db3", "stage_origin": 0.0, "offset": 4.766, "calcium_shift_s": 0.0, "expected_stage_epoch_s": 1.0},
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
    for record in SUR_1S_RECORDS:
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
    limit = float(np.nanpercentile(np.abs(matrix), 98))

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
    ax.set_ylabel("SUR mouse / eligible events")
    ax.set_title("SUR NREM-to-MA calcium events (1-s scoring; corrected alignment)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.18, fraction=0.07)
    colorbar.set_label("Δ signal vs −20 to −10 s baseline (stored units)")
    fig.text(0.99, 0.015, f"n=7 mice; {len(matrix)} events", ha="right", va="bottom", fontsize=6)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.22)

    base = OUT / "panels" / "panel_f_SUR_event_heatmap_1s_alignment_corrected"
    outputs = save_figure(fig, base, tiff=True)
    plt.close(fig)

    event_path = OUT / "data" / "SUR_1s_alignment_corrected_event_level_transition_timecourses.csv"
    rows_path = OUT / "data" / "SUR_1s_alignment_corrected_event_heatmap_row_manifest.csv"
    source_path = OUT / "data" / "SUR_1s_alignment_corrected_heatmap_source_manifest.csv"
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")
    row_manifest.to_csv(rows_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sources).to_csv(source_path, index=False, encoding="utf-8-sig")

    script_copy = OUT / "scripts" / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy)
    shifts = {str(record["animal"]): float(record["calcium_shift_s"]) for record in SUR_1S_RECORDS}
    readme = f"""# SUR 1-s scoring event heatmap with user-specified alignment corrections

- Purpose: redraw the SUR NREM-to-MA event heatmap using the manually refined 1-s scoring files.
- Sample: seven independent SUR mice; {len(matrix)} eligible events.
- Score resolution: 1 s in every included mouse ({json.dumps(epochs, ensure_ascii=False)}).
- Calcium shifts relative to EEG (positive = later): {json.dumps(shifts, ensure_ascii=False)} s.
- MA definition: Wake ≤20 s flanked by NREM; at least 20 s continuous preceding NREM.
- Calcium window: −20 to +30 s around the 1-s-defined MA onset, summarized in 1-s bins.
- Baseline: −20 to −10 s for each event.
- Row order: animal, then event onset time; white horizontal separators mark mouse boundaries.
- Color scale: shared symmetric ±{limit:.4g} stored units, determined by the 98th percentile of absolute event-bin values.
- Inference: none. Rows are repeated events and this panel is descriptive only.
- Raw Plot and score files were read only; source hashes are recorded in `{source_path}`.
"""
    readme_path = OUT / "documentation" / "SUR_1s_alignment_corrected_heatmap_README.md"
    readme_path.write_text(readme, encoding="utf-8")

    provenance = f"""# {FIGURE_ID}

- figure_title: SUR NREM-to-MA calcium event heatmap using 1-s scoring and corrected alignment
- manuscript_context: Candidate detail panel replacing the coarse 5-s SUR event-onset visualization.
- status: candidate
- generated_at: {datetime.now().astimezone().isoformat(timespec='seconds')}
- sample_definition: seven independent SUR mice; event rows are descriptive and not independent statistical replicates.
- analysis_window: −20 to +30 s around MA onset; −20 to −10 s baseline; 1-s score and 1-s calcium display bins.
- analysis_steps: Wake ≤20 s flanked by NREM; ≥20 s preceding NREM; user-specified calcium shifts SUR2 +4 s, SUR3 +6 s, SUR4 +8 s, SUR5 +3 s, SUR6 +4 s; rows ordered within mouse by onset.
- statistics: none in this heatmap.
- scripts: {script_copy}
- source_data: {event_path} | {rows_path} | {source_path}
- outputs: {' | '.join(str(path) for path in outputs)}
- review_notes: SUR-only high-resolution visualization; do not use event rows as inferential n.
"""
    provenance_path = PROVENANCE_ROOT / f"{FIGURE_ID}_provenance.md"
    provenance_path.write_text(provenance, encoding="utf-8")
    shutil.copy2(provenance_path, OUT / "documentation" / provenance_path.name)

    manifest_path = OUT / "documentation" / "SUR_1s_alignment_corrected_heatmap_outputs_manifest.csv"
    manifest_files = [*outputs, event_path, rows_path, source_path, script_copy, readme_path, provenance_path]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "path", "status", "sha256"])
        writer.writeheader()
        for path in manifest_files:
            writer.writerow({"type": "figure" if path.suffix.lower() in {".png", ".pdf", ".svg", ".tiff"} else "supporting", "path": str(path), "status": "current", "sha256": sha256(path)})

    for source in sources:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise AssertionError(f"Source changed: {source['path']}")
    print(json.dumps({"output": str(base.with_suffix('.png')), "events": len(matrix), "events_per_mouse": group_sizes.to_dict(), "score_epochs_s": epochs, "calcium_shift_s": shifts, "color_limit": limit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
