from __future__ import annotations

import hashlib
import heapq
import json
import math
import shutil
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import redraw_fig8_requested_16panel_set as score_core  # noqa: E402


WORK = Path(r"F:\1.Sleep\PHD稿件\Figure_Workspace\Fig8_hM3Dq_chemogenetics")
FORMAL = Path(r"F:\1.Sleep\PHD稿件\Figures\Fig8_hM3Dq_chemogenetics")
GLOBAL_PROV = Path(r"F:\Sleep\outputs\figure_provenance")
RAW = Path(r"F:\1.Sleep\eXdata\hm3Dq\CNO")
ANALYSIS = WORK / "data" / "Latest_C_H_reanalysis" / "analysis"
DETECTOR = ANALYSIS / "animal_level_all_metrics_normalized.tsv"
EVENTS = ANALYSIS / "events"
GROUPS = ["mCherry+SAL", "mCherry+CNO", "hmdq3+SAL", "hmdq3+CNO"]
XPOS = np.array([0.78, 1.22, 2.78, 3.22])
COLORS = ["#D2D2D2", "#A6A6A6", "#F2B59A", "#EF854B"]
EDGES = ["#666666", "#666666", "#D95420", "#D95420"]
LABELS = ["SAL", "CNO", "SAL", "CNO"]

FOUR = [
    ("G_Spindle_density", "spindle_density_per_nrem_min", "Spindles / NREM min", "Spindle density", 1),
    ("J_SO_density", "so_density_per_nrem_min", "SOs / NREM min", "SO density", 0),
    ("Fig8_Coupled_spindle_density", "coupled_density_per_nrem_min", "SO-coupled spindles / NREM min", "Coupled spindle density", 1),
    ("K_SO_Spindle_coupling", "coupled_fraction_percent", "SO-coupled spindles (%)", "SO-spindle temporal coupling", 0),
]
# Fig8 mirrors Fig9 and therefore displays only the three count/density
# endpoints.  Coupled fraction is still calculated and retained in source and
# statistics tables for auditability, but is intentionally not plotted.
DISPLAY_THREE = FOUR[:3]
SIX = [
    ("C_MA", "ma_frequency_per_nrem_h", "MAs / NREM hour", "MA frequency", 0),
    ("D_NREM_duration", "nrem_mean_bout_seconds", "Duration (s)", "NREM", 0),
    ("Fig8_bout_nrem_frequency", "nrem_bout_frequency_per_valid_h", "Frequency / h", "NREM", 0),
    FOUR[0], FOUR[1], FOUR[2],
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def exact_permutation(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.r_[a, b]
    observed = abs(float(np.mean(a) - np.mean(b)))
    hits = total = 0
    for idx in combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idx)] = True
        difference = abs(float(np.mean(pooled[mask]) - np.mean(pooled[~mask])))
        total += 1
        hits += difference >= observed - 1e-12
    return hits / total


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.maximum.accumulate(np.minimum(1.0, p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def couple_events(spindles: pd.DataFrame, slow: pd.DataFrame) -> list[dict[str, object]]:
    slow = slow.sort_values("start_seconds").reset_index(drop=True)
    starts = slow.start_seconds.to_numpy(float)
    ends = slow.end_seconds.to_numpy(float)
    down = slow.down_peak_seconds.to_numpy(float)
    active: set[int] = set()
    end_heap: list[tuple[float, int]] = []
    cursor = 0
    output = []
    ordered = spindles.sort_values("peak_seconds")
    for spindle in ordered.itertuples(index=False):
        peak = float(spindle.peak_seconds)
        while cursor < len(starts) and starts[cursor] <= peak:
            active.add(cursor)
            heapq.heappush(end_heap, (ends[cursor], cursor))
            cursor += 1
        while end_heap and end_heap[0][0] < peak:
            _, idx = heapq.heappop(end_heap)
            active.discard(idx)
        if not active:
            continue
        idx = min(active, key=lambda j: abs(down[j] - peak))
        so = slow.iloc[idx]
        output.append({
            "spindle_event_id": spindle.event_id,
            "so_event_id": so.event_id,
            "spindle_peak_seconds": peak,
            "so_start_seconds": float(so.start_seconds),
            "so_end_seconds": float(so.end_seconds),
            "so_down_peak_seconds": float(so.down_peak_seconds),
            "so_up_peak_seconds": float(so.up_peak_seconds),
            "down_peak_lag_ms": 1000.0 * (peak - float(so.down_peak_seconds)),
        })
    return output


def load_refresh() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detector = pd.read_csv(DETECTOR, sep="\t")
    detector = detector[detector.group.isin(GROUPS)].copy()
    if detector.groupby("group").size().to_dict() != {g: 6 for g in GROUPS}:
        raise RuntimeError(detector.groupby("group").size().to_dict())
    rows, coupled_rows, paths = [], [], []
    for rec in detector.sort_values(["group", "subject_for_sensitivity"]).itertuples(index=False):
        group, subject = str(rec.group), str(rec.subject_for_sensitivity)
        spindle_path = EVENTS / f"{group}_{subject}_spindle_events.csv"
        so_root = RAW / group / subject / "export_SO200Hz_so"
        so_events_path = so_root / "export_SO200Hz_so_events.tsv"
        so_summary_path = so_root / "export_SO200Hz_so_summary.tsv"
        so_metadata_path = so_root / "export_SO200Hz_so_metadata.json"
        for path in (spindle_path, so_events_path, so_summary_path, so_metadata_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        spindle = pd.read_csv(spindle_path)
        spindle = spindle[spindle.accepted.map(truth)].copy()
        slow = pd.read_csv(so_events_path, sep="\t")
        slow = slow[slow.accepted.map(truth)].copy()
        summary = pd.read_csv(so_summary_path, sep="\t").iloc[0]
        metadata = json.loads(so_metadata_path.read_text(encoding="utf-8"))
        if len(spindle) != int(rec.spindle_count):
            raise RuntimeError(f"Spindle count mismatch: {group}/{subject}")
        if len(slow) != int(summary.accepted_count):
            raise RuntimeError(f"SO count mismatch: {group}/{subject}")
        if abs(float(summary.nrem_minutes) - float(rec.nrem_minutes_artifact_free)) > 1e-8:
            raise RuntimeError(f"NREM denominator mismatch: {group}/{subject}")
        coupled = couple_events(spindle, slow)
        for event in coupled:
            event.update({"group": group, "subject": subject})
        coupled_rows.extend(coupled)
        count = len(coupled)
        rows.append({
            "group": group, "virus": rec.virus, "drug": rec.drug,
            "subject_for_sensitivity": subject, "animal": rec.animal,
            "nrem_minutes_artifact_free": float(summary.nrem_minutes),
            "spindle_count": len(spindle),
            "spindle_density_per_nrem_min": len(spindle) / float(summary.nrem_minutes),
            "so_count": len(slow),
            "so_density_per_nrem_min": len(slow) / float(summary.nrem_minutes),
            "coupled_count": count,
            "coupled_density_per_nrem_min": count / float(summary.nrem_minutes),
            "coupled_fraction_percent": 100.0 * count / len(spindle),
            "coupling_definition": "accepted spindle peak within accepted SO start-end interval",
            "phase_metrics_reported": False,
            "spindle_reviews_restored": rec.spindle_reviews_restored,
            "spindle_reviews_excluded": rec.spindle_reviews_excluded,
        })
        for role, path in (("spindle_events", spindle_path), ("new_so_events", so_events_path),
                           ("new_so_summary", so_summary_path), ("new_so_metadata", so_metadata_path)):
            paths.append({"group": group, "subject": subject, "role": role, "path": str(path),
                          "sha256": sha256(path), "last_write": datetime.fromtimestamp(path.stat().st_mtime).isoformat()})
        paths[-1]["phase_safe_metadata"] = bool(metadata.get("source", {}).get("phase_safe", False))
    frame = pd.DataFrame(rows)
    inventory = detector.copy()
    score, _, score_audit = score_core.prepare_score_metrics(inventory)
    score = score.rename(columns={"subject": "subject_for_sensitivity"})
    frame = frame.merge(score[["group", "subject_for_sensitivity", "ma_frequency_per_nrem_h",
                               "nrem_mean_bout_seconds", "nrem_bout_frequency_per_valid_h"]],
                        on=["group", "subject_for_sensitivity"], validate="one_to_one")
    frame["group"] = pd.Categorical(frame.group, GROUPS, ordered=True)
    frame = frame.sort_values(["group", "subject_for_sensitivity"]).reset_index(drop=True)
    return frame, pd.DataFrame(coupled_rows), pd.DataFrame(paths), score_audit


def statistics(frame: pd.DataFrame, specs: list[tuple]) -> pd.DataFrame:
    rows = []
    for stem, metric, *_ in specs:
        arrays = [frame[frame.group == g].sort_values("subject_for_sensitivity")[metric].to_numpy(float) for g in GROUPS]
        primary = [
            ("SAL cross-virus", 0, 2, exact_permutation(arrays[0], arrays[2])),
            ("CNO cross-virus", 1, 3, exact_permutation(arrays[1], arrays[3])),
            ("virus x drug interaction", -1, -1, exact_permutation(arrays[1] - arrays[0], arrays[3] - arrays[2])),
        ]
        adjusted = holm([r[3] for r in primary])
        for (name, i, j, p), p_adj in zip(primary, adjusted):
            rows.append({"panel": stem, "metric": metric, "role": "primary", "comparison": name,
                         "first_group": GROUPS[i] if i >= 0 else "mCherry paired change",
                         "second_group": GROUPS[j] if j >= 0 else "hM3Dq paired change",
                         "p_raw": p, "p_holm_primary_family": p_adj,
                         "display_p": p_adj, "displayed": p_adj < 0.05})
        for name, i, j in (("mCherry SAL vs CNO paired", 0, 1), ("hM3Dq SAL vs CNO paired", 2, 3)):
            result = wilcoxon(arrays[i], arrays[j], alternative="two-sided", method="exact")
            rows.append({"panel": stem, "metric": metric, "role": "paired sensitivity", "comparison": name,
                         "first_group": GROUPS[i], "second_group": GROUPS[j], "p_raw": float(result.pvalue),
                         "wilcoxon_w": float(result.statistic), "display_p": float(result.pvalue),
                         "displayed": result.pvalue < 0.05})
    return pd.DataFrame(rows)


def pairs(stats: pd.DataFrame, stem: str) -> list[tuple[int, int, float]]:
    out = []
    for name, i, j in (("SAL cross-virus", 0, 2), ("CNO cross-virus", 1, 3),
                       ("mCherry SAL vs CNO paired", 0, 1), ("hM3Dq SAL vs CNO paired", 2, 3)):
        row = stats[(stats.panel == stem) & (stats.comparison == name)]
        if len(row) and bool(row.iloc[0].displayed):
            out.append((i, j, float(row.iloc[0].display_p)))
    return out


def draw(ax: plt.Axes, frame: pd.DataFrame, stats: pd.DataFrame, spec: tuple) -> None:
    stem, metric, ylabel, title, decimals = spec
    arrays = [frame[frame.group == g].sort_values("subject_for_sensitivity")[metric].to_numpy(float) for g in GROUPS]
    annotations = pairs(stats, stem)
    top = max(float(np.max(v)) for v in arrays)
    target = top * (1.22 + 0.10 * len(annotations))
    rough = target / 5
    mag = 10 ** math.floor(math.log10(rough)) if rough > 0 else 1
    step = min([1, 2, 2.5, 5, 10], key=lambda x: abs(x * mag - rough)) * mag
    ymax = math.ceil(target / step) * step
    # Keep every animal point vertically aligned with its bar.  The paired
    # traces encode identity; horizontal jitter makes those traces needlessly
    # tangled and is not used in the approved manuscript style.
    jitter = np.zeros(6, dtype=float)
    for start in (0, 2):
        for animal in range(6):
            ax.plot(XPOS[start:start + 2] + jitter[animal], [arrays[start][animal], arrays[start + 1][animal]],
                    color=EDGES[start], lw=0.75, alpha=0.42, zorder=2)
    for i, (x, values) in enumerate(zip(XPOS, arrays)):
        ax.bar(x, values.mean(), width=0.38, color=COLORS[i], edgecolor=EDGES[i], lw=1.15, zorder=1)
        ax.errorbar(x, values.mean(), yerr=sem(values), fmt="none", ecolor="black", elinewidth=1.15,
                    capsize=3, capthick=1.15, zorder=5)
        ax.scatter(x + jitter, values, s=22, facecolors="white", edgecolors=EDGES[i], lw=1, zorder=4)
    base = max(top + 0.06 * ymax, 0.72 * ymax)
    for k, (i, j, p) in enumerate(annotations):
        y = base + k * 0.10 * ymax
        h = 0.018 * ymax
        ax.plot([XPOS[i], XPOS[i], XPOS[j], XPOS[j]], [y, y + h, y + h, y], color="black", lw=1.1, clip_on=False)
        ax.text((XPOS[i] + XPOS[j]) / 2, y + h + 0.008 * ymax, star(p), ha="center", va="bottom", fontsize=10)
    ax.set_xlim(0.35, 3.65); ax.set_ylim(0, ymax); ax.set_yticks(np.arange(0, ymax + step * 0.5, step))
    if decimals:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{decimals}f"))
    ax.set_ylabel(ylabel); ax.set_title(title, pad=5); ax.set_xticks(XPOS, LABELS)
    ax.text(1, -0.17, "mCherry", ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(3, -0.17, "hM3Dq", ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(direction="out", pad=2)


def save(fig: plt.Figure, base: Path, dpi: int = 600) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        kwargs = {"dpi": dpi} if suffix in (".png", ".tiff") else {}
        if suffix == ".tiff": kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", facecolor="white", transparent=False, **kwargs)
    for suffix in (".pdf", ".svg"):
        fig.savefig(base.parent / f"{base.name}_AI_clean{suffix}", bbox_inches="tight", facecolor="white", transparent=False)


def archive(root: Path, stamp: str) -> Path:
    dest = root / "archive" / f"before_new_SO_spindle_coupling_{stamp}"
    stems = {s[0] for s in FOUR + SIX}
    targets = []
    for stem in stems:
        targets += list((root / "panels").glob(stem + "*"))
    targets += list(root.glob("Fig8_Spindle_SO_two_coupling_latest*"))
    targets += list(root.glob("Fig8_microstructure_6metrics_latest*"))
    for src in targets:
        if src.is_file():
            dst = dest / src.relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    return dest


def update_manifest(root: Path, rels: list[str]) -> None:
    path = root / "documentation" / "outputs_manifest.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["type", "path", "description", "status"])
    if "path" in old:
        old = old[~old.path.astype(str).isin(rels)]
        old = old[~old.path.astype(str).str.contains("K_SO_Spindle_coupling", regex=False)]
    rows = pd.DataFrame([{"type": "figure_output", "path": r,
                          "description": "Fig8 refreshed Spindle, user-exported SO and temporal coupling", "status": "current"}
                         for r in sorted(set(rels))])
    pd.concat([old, rows], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 1.15,
                         "xtick.major.width": 1.15, "ytick.major.width": 1.15,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    frame, coupled, source_paths, score_audit = load_refresh()
    stats4 = statistics(frame, FOUR)
    stats6 = statistics(frame, SIX)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp = HERE / "_fig8_new_so_temp"
    if temp.exists(): shutil.rmtree(temp)
    (temp / "panels").mkdir(parents=True)
    for spec in {s[0]: s for s in DISPLAY_THREE + SIX}.values():
        fig, ax = plt.subplots(figsize=(3.05, 2.55)); draw(ax, frame, stats6 if spec in SIX else stats4, spec)
        fig.subplots_adjust(left=0.23, right=0.98, top=0.90, bottom=0.26); save(fig, temp / "panels" / spec[0]); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.75))
    for ax, spec in zip(axes.flat, DISPLAY_THREE): draw(ax, frame, stats4, spec)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.27, wspace=0.48)
    save(fig, temp / "Fig8_Spindle_SO_two_coupling_latest"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4))
    for ax, spec in zip(axes.flat, SIX): draw(ax, frame, stats6, spec)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.12, wspace=0.52, hspace=0.62)
    save(fig, temp / "Fig8_microstructure_6metrics_latest"); plt.close(fig)

    report = {"generated": datetime.now().isoformat(timespec="seconds"), "archives": {}}
    for root in (WORK, FORMAL):
        old = archive(root, stamp); report["archives"][str(root)] = str(old)
        # The removed coupled-fraction panel is preserved in the timestamped
        # archive but must not remain among the live Fig8 panel deliverables.
        for obsolete in (root / "panels").glob("K_SO_Spindle_coupling*"):
            if obsolete.is_file():
                obsolete.unlink()
        for sub in ("panels", "data", "scripts", "documentation"): (root / sub).mkdir(parents=True, exist_ok=True)
        rels = []
        for src in (temp / "panels").iterdir():
            dst = root / "panels" / src.name; shutil.copy2(src, dst); rels.append(str(dst.relative_to(root)).replace("/", "\\"))
        for src in temp.glob("Fig8_*latest*"):
            dst = root / src.name; shutil.copy2(src, dst); rels.append(str(dst.relative_to(root)).replace("/", "\\"))
        tables = {
            "Fig8_Spindle_SO_two_coupling_source_data.csv": frame,
            "Fig8_Spindle_SO_two_coupling_statistics.csv": stats4,
            "Fig8_Spindle_SO_temporal_coupled_events.csv": coupled,
            "Fig8_Spindle_SO_source_paths.csv": source_paths,
            "Fig8_Spindle_SO_score_audit.csv": score_audit,
            "Fig8_microstructure_6metrics_source_data.csv": frame,
            "Fig8_microstructure_6metrics_statistics.csv": stats6,
        }
        for name, table in tables.items():
            dst = root / "data" / name; table.to_csv(dst, index=False, encoding="utf-8-sig"); rels.append(str(dst.relative_to(root)).replace("/", "\\"))
        script_dst = root / "scripts" / Path(__file__).name; shutil.copy2(__file__, script_dst); rels.append(str(script_dst.relative_to(root)).replace("/", "\\"))
        provenance = root / "documentation" / "Fig8_Spindle_SO_two_coupling_provenance.md"
        provenance.write_text(
            "# Fig8 refreshed Spindle, SO and temporal coupling\n\n"
            f"- Generated: {report['generated']}\n"
            "- Sample: mCherry+SAL, mCherry+CNO, hM3Dq+SAL and hM3Dq+CNO; n=6 paired animals per virus.\n"
            "- Spindles: freshly redetected from current EDF, scores and artifact masks; persisted manual reviews restored.\n"
            "- SOs: the 24 user-re-exported SO event/summary files generated on 2026-09-26.\n"
            "- SO density denominator: artifact-free NREM minutes recorded in each new SO summary and verified against the refreshed Spindle analysis.\n"
            "- Temporal coupling: an accepted Spindle is coupled when its peak time falls within an accepted SO start-end interval.\n"
            "- Coupled density: temporally coupled Spindles per artifact-free NREM minute. Coupled fraction: temporally coupled Spindles / accepted Spindles x 100.\n"
            "- Display decision: Fig8 mirrors Fig9 and omits the coupled-fraction/temporal-coupling percentage panel; that metric remains in the source and statistics tables only.\n"
            "- The new SO metadata did not assert phase safety; therefore phase angles and phase-locking statistics are not reported. Event-time overlap remains valid.\n"
            "- Statistics: exact paired Wilcoxon for SAL-CNO within virus; exact cross-virus permutation and change-score interaction, Holm-adjusted across the three primary comparisons within each metric.\n"
            "- Display: mean +/- SEM, vertically aligned hollow animal points and confirmed same-animal traces; no horizontal jitter and no animal IDs in formal panels.\n"
            f"- Replaced live outputs archived under `{old}`.\n",
            encoding="utf-8")
        rels.append(str(provenance.relative_to(root)).replace("/", "\\")); update_manifest(root, rels)
    GLOBAL_PROV.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FORMAL / "documentation" / "Fig8_Spindle_SO_two_coupling_provenance.md", GLOBAL_PROV / "Fig8_Spindle_SO_two_coupling_provenance.md")
    shutil.copy2(FORMAL / "data" / "Fig8_Spindle_SO_two_coupling_source_data.csv", GLOBAL_PROV / "Fig8_Spindle_SO_two_coupling_source_data.csv")
    shutil.copy2(FORMAL / "data" / "Fig8_Spindle_SO_two_coupling_statistics.csv", GLOBAL_PROV / "Fig8_Spindle_SO_two_coupling_stats.csv")
    (FORMAL / "documentation" / "Fig8_Spindle_SO_two_coupling_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GROUP MEANS")
    print(frame.groupby("group", observed=True)[[s[1] for s in FOUR]].mean().to_string())
    print("\nSTATISTICS")
    print(stats4[["metric", "comparison", "p_raw", "p_holm_primary_family", "displayed"]].to_string(index=False))
    print(f"\nOUTPUT={FORMAL / 'Fig8_Spindle_SO_two_coupling_latest.png'}")


if __name__ == "__main__":
    main()
