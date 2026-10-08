from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


BACKEND = Path(r"F:\Sleep\EEGsoftware\EEGSoftware\AutoFQ\SleepWorkbench")
sys.path.insert(0, str(BACKEND))

from sleepworkbench.slow_oscillation_detection import (  # noqa: E402
    SlowOscillationConfig,
    detect_multichannel_slow_oscillations,
    phase_safety_from_provenance,
)
from sleepworkbench.workbench_core import ScoreDatabase, read_edf_signals  # noqa: E402


ROOT = Path(r"F:\1.Sleep\eXdata\hm4Di")
OLD_SOURCE = Path(
    r"F:\Sleep\Figure\Fig_hm4Di_FourGroup_Microstructure_FullManualEvents_20260824_v1"
    r"\data\hm4di_four_group_microstructure_source_data.csv"
)
PLOTMOD_PATH = Path(
    r"F:\Sleep\Figure\Fig_hm4Di_FourGroup_Microstructure_FullManualEvents_20260824_v1"
    r"\scripts\analyze_plot_four_group_microstructure_local.py"
)
GROUPS = ("baseline+SAL", "baseline+CNO", "POCD+SAL", "POCD+CNO")
EXPECTED_N = {"baseline+SAL": 8, "baseline+CNO": 6, "POCD+SAL": 8, "POCD+CNO": 7}
METRICS = (
    "spindle_density_per_nrem_min",
    "mean_event_spindle_power_db_vs_nonspindle_nrem_sigma",
    "so_density_per_nrem_min",
)


def load_plot_module():
    spec = importlib.util.spec_from_file_location("hm4di_plot_helpers", PLOTMOD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(PLOTMOD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotmod = load_plot_module()


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def complete_records() -> list[tuple[str, str, Path, Path, Path, Path]]:
    records: list[tuple[str, str, Path, Path, Path, Path]] = []
    for group in GROUPS:
        for animal_dir in sorted((p for p in (ROOT / group).iterdir() if p.is_dir()), key=lambda p: p.name):
            edf = animal_dir / "export_SO200Hz.edf"
            db = animal_dir / "scores.db3"
            events = animal_dir / "export_SO200Hz_spindle" / "export_SO200Hz_spindle_events.tsv"
            summary = animal_dir / "export_SO200Hz_spindle" / "export_SO200Hz_spindle_summary.tsv"
            if all(p.is_file() for p in (edf, db, events, summary)):
                records.append((group, animal_dir.name, edf, db, events, summary))
    counts = pd.Series([r[0] for r in records]).value_counts().to_dict()
    if counts != EXPECTED_N:
        raise RuntimeError(f"Unexpected complete-record counts: {counts}; expected {EXPECTED_N}")
    return records


def analyze_record(group: str, animal: str, edf: Path, db: Path, events: Path, summary: Path) -> dict[str, object]:
    database = ScoreDatabase(db)
    scores = database.scores.copy()
    epoch_seconds = float(database.epoch_seconds)
    artifact_mask = scores == 0
    header, signals = read_edf_signals(edf)
    sample_rate = float(header["sample_rates"][0])
    if not np.isclose(sample_rate, 200.0):
        raise RuntimeError(f"Unexpected sample rate {sample_rate}: {edf}")
    eeg_label = str(header["labels"][0])
    eeg_unit = str(header["physical_dimensions"][0] or "uV")
    eeg = np.asarray(signals[0], dtype=np.float32) * plotmod.unit_factor(eeg_unit)

    manual = pd.read_csv(events, sep="\t")
    manual["accepted"] = manual["accepted"].map(truth)
    accepted = manual.loc[manual["accepted"]].copy()
    manual_summary = pd.read_csv(summary, sep="\t").iloc[0]
    if len(accepted) != int(manual_summary["spindle_count"]):
        raise RuntimeError(f"Manual count mismatch: {group}/{animal}")

    nrem_seconds = float(np.sum(scores == 2) * epoch_seconds)
    preprocess_path = edf.with_suffix(".preprocess.json")
    provenance = json.loads(preprocess_path.read_text(encoding="utf-8"))
    phase_safe, phase_message = phase_safety_from_provenance(edf, provenance)
    if not phase_safe:
        raise RuntimeError(f"Phase safety failed: {group}/{animal}: {phase_message}")

    valid_nrem = plotmod.spindle_module._epoch_mask(
        scores, epoch_seconds, sample_rate, len(eeg), artifact_mask
    )
    sample_mask, epoch_mask = plotmod.event_masks(
        accepted,
        sample_rate=sample_rate,
        sample_count=len(eeg),
        epoch_count=len(scores),
        epoch_seconds=epoch_seconds,
    )
    background = plotmod.filtered_background_power(
        eeg,
        sample_rate=sample_rate,
        valid_nrem_mask=valid_nrem,
        event_sample_mask=sample_mask,
        event_epoch_mask=epoch_mask,
        epoch_seconds=epoch_seconds,
    )
    event_power = np.square(accepted["rms_uv"].to_numpy(dtype=float))
    normalized_db = float(
        10.0
        * np.log10(
            float(event_power.mean())
            / float(background["nonspindle_nrem_sigma_power_uv2"])
        )
    )

    manual_so_candidates = sorted(
        edf.parent.glob("*_so/*_so_summary.tsv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    manual_so_summary_path = manual_so_candidates[0] if manual_so_candidates else None
    manual_so_metadata_path = (
        manual_so_summary_path.with_name(manual_so_summary_path.name.replace("_summary.tsv", "_metadata.json"))
        if manual_so_summary_path is not None else None
    )
    if manual_so_summary_path is not None:
        manual_so_summary = pd.read_csv(manual_so_summary_path, sep="\t").iloc[0]
        if abs(float(manual_so_summary["nrem_minutes"]) - nrem_seconds / 60.0) > epoch_seconds / 60.0 + 1e-9:
            raise RuntimeError(f"Manual SO NREM denominator differs from current DB3: {group}/{animal}")
        so_count = int(manual_so_summary["accepted_count"])
        so_density = float(manual_so_summary["density_per_nrem_min"])
        so_source_kind = "user_exported_so_summary_override"
        so_summary_path = str(manual_so_summary_path.resolve())
        so_summary_sha256 = sha256(manual_so_summary_path)
        so_metadata_path = str(manual_so_metadata_path.resolve()) if manual_so_metadata_path and manual_so_metadata_path.is_file() else ""
        so_metadata_sha256 = sha256(manual_so_metadata_path) if manual_so_metadata_path and manual_so_metadata_path.is_file() else ""
        if manual_so_metadata_path and manual_so_metadata_path.is_file():
            so_metadata = json.loads(manual_so_metadata_path.read_text(encoding="utf-8"))
            so_phase_safe = bool(so_metadata.get("source", {}).get("phase_safe", False))
            so_phase_message = str(so_metadata.get("source", {}).get("phase_message", ""))
        else:
            so_phase_safe = False
            so_phase_message = "Manual SO metadata was not available."
    else:
        so_result = detect_multichannel_slow_oscillations(
            [(0, eeg_label, eeg, "uV")],
            sample_rate,
            scores,
            epoch_seconds,
            config=SlowOscillationConfig.strict_so(),
            artifact_masks={0: artifact_mask},
            metadata={"group": group, "animal": animal, "phase_safety": phase_message},
        )
        so_summary = so_result.summary_frame().iloc[0]
        so_count = int(so_summary["accepted_count"])
        so_density = float(so_summary["density_per_nrem_min"])
        so_source_kind = "fresh_strict_so_detection_from_phase_safe_export_SO200Hz"
        so_summary_path = ""
        so_summary_sha256 = ""
        so_metadata_path = ""
        so_metadata_sha256 = ""
        so_phase_safe = True
        so_phase_message = phase_message
    return {
        "group": group,
        "animal": animal,
        "nrem_minutes": nrem_seconds / 60.0,
        "accepted_spindle_count": int(manual_summary["spindle_count"]),
        "spindle_density_per_nrem_min": float(manual_summary["density_per_nrem_min"]),
        "mean_event_spindle_power_db_vs_nonspindle_nrem_sigma": normalized_db,
        "nonspindle_nrem_sigma_power_uv2": float(background["nonspindle_nrem_sigma_power_uv2"]),
        "so_count": so_count,
        "so_density_per_nrem_min": so_density,
        "so_source_kind": so_source_kind,
        "so_summary_path": so_summary_path,
        "so_summary_sha256": so_summary_sha256,
        "so_metadata_path": so_metadata_path,
        "so_metadata_sha256": so_metadata_sha256,
        "so_phase_safe": so_phase_safe,
        "so_phase_message": so_phase_message,
        "manual_spindle_events_path": str(events.resolve()),
        "manual_spindle_summary_path": str(summary.resolve()),
        "source_edf_path": str(edf.resolve()),
        "scores_db_path": str(db.resolve()),
        "manual_spindle_events_sha256": sha256(events),
        "manual_spindle_events_last_write": events.stat().st_mtime,
        "phase_safe": True,
    }


def statistical_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []
    planned_rows: list[dict[str, object]] = []
    contrasts = (("baseline+SAL", "baseline+CNO"), ("POCD+SAL", "POCD+CNO"))
    for metric in METRICS:
        for group in GROUPS:
            values = frame.loc[frame.group == group, metric].to_numpy(float)
            summary_rows.append({
                "metric": metric,
                "group": group,
                "n": len(values),
                "mean": float(np.mean(values)),
                "sem": float(np.std(values, ddof=1) / math.sqrt(len(values))),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
        arrays = [frame.loc[frame.group == group, metric].to_numpy(float) for group in GROUPS]
        statistic, p = kruskal(*arrays)
        omnibus_rows.append({"metric": metric, "test": "Kruskal-Wallis", "statistic": statistic, "p_raw": p})
        for left, right in contrasts:
            a = frame.loc[frame.group == left, metric].to_numpy(float)
            b = frame.loc[frame.group == right, metric].to_numpy(float)
            result = mannwhitneyu(a, b, alternative="two-sided", method="exact")
            planned_rows.append({
                "metric": metric,
                "contrast": f"{left} vs {right}",
                "group_a": left,
                "group_b": right,
                "n_a": len(a),
                "n_b": len(b),
                "test": "exact two-sided Mann-Whitney U",
                "statistic": float(result.statistic),
                "p_raw": float(result.pvalue),
            })
    planned = pd.DataFrame(planned_rows)
    for contrast in planned["contrast"].unique():
        mask = planned["contrast"] == contrast
        planned.loc[mask, "p_holm_3_endpoints"] = holm(planned.loc[mask, "p_raw"].tolist())
    planned["stars"] = planned["p_holm_3_endpoints"].map(stars)
    return pd.DataFrame(summary_rows), pd.DataFrame(omnibus_rows), planned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    records = complete_records()
    for i, record in enumerate(records, start=1):
        print(f"[{i:02d}/{len(records)}] {record[0]}/{record[1]}", flush=True)
        rows.append(analyze_record(*record))
    frame = pd.DataFrame(rows)
    frame["manual_spindle_events_last_write"] = pd.to_datetime(
        frame["manual_spindle_events_last_write"], unit="s", utc=True
    ).dt.tz_convert("Asia/Shanghai").astype(str)

    old = pd.read_csv(OLD_SOURCE)[["group", "animal", "manual_spindle_events_sha256"]].rename(
        columns={"manual_spindle_events_sha256": "old_sha256"}
    )
    audit = frame[["group", "animal", "manual_spindle_events_sha256", "manual_spindle_events_last_write"]].merge(
        old, on=["group", "animal"], how="left"
    )
    audit["status_vs_20260824"] = np.where(
        audit["old_sha256"].isna(),
        "new_animal",
        np.where(audit["old_sha256"] == audit["manual_spindle_events_sha256"], "unchanged", "changed"),
    )
    summary, omnibus, planned = statistical_tables(frame)
    frame.to_csv(out / "hm4di_latest_manual_spindle_source_data.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "hm4di_latest_manual_spindle_group_summary.csv", index=False, encoding="utf-8-sig")
    omnibus.to_csv(out / "hm4di_latest_manual_spindle_omnibus.csv", index=False, encoding="utf-8-sig")
    planned.to_csv(out / "hm4di_latest_manual_spindle_statistics.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "manual_event_revision_audit.csv", index=False, encoding="utf-8-sig")
    so_audit = frame.loc[frame["so_source_kind"] == "user_exported_so_summary_override", [
        "group", "animal", "so_count", "so_density_per_nrem_min", "so_summary_path",
        "so_summary_sha256", "so_metadata_path", "so_metadata_sha256", "so_phase_safe", "so_phase_message",
    ]]
    so_audit.to_csv(out / "so_override_audit.csv", index=False, encoding="utf-8-sig")
    (out / "analysis_manifest.json").write_text(
        json.dumps({
            "group_order": list(GROUPS),
            "group_n": EXPECTED_N,
            "metrics": list(METRICS),
            "manual_event_status_counts": audit["status_vs_20260824"].value_counts().to_dict(),
            "so_source_counts": frame["so_source_kind"].value_counts().to_dict(),
            "statistics": "Kruskal-Wallis omnibus; exact two-sided Mann-Whitney planned SAL-vs-CNO contrasts; Holm correction across the three redrawn endpoints within each condition",
            "source_root": str(ROOT),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nGROUP SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nPLANNED", flush=True)
    print(planned.to_string(index=False), flush=True)
    print("\nAUDIT", flush=True)
    print(audit["status_vs_20260824"].value_counts().to_string(), flush=True)
    print(f"OUTPUT={out}", flush=True)


if __name__ == "__main__":
    main()
