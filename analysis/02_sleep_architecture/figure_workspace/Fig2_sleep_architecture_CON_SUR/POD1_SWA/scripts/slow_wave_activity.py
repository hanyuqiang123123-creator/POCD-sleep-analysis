from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import windows

from sleepworkbench.analysis_fingerprint import (
    SCORE_FINGERPRINT_SCHEMA,
    score_fingerprint,
)
from sleepworkbench.artifact_store import effective_artifact_mask
from sleepworkbench.spectral_analysis import signal_in_microvolts

ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]
SWA_ANALYSIS_VERSION = "1.1.0"
SWA_LOW_HZ = 0.5
SWA_HIGH_HZ = 4.0


class SlowWaveActivityCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SlowWaveActivityConfig:
    window_seconds: float = 4.0
    step_seconds: float = 1.0
    low_hz: float = SWA_LOW_HZ
    high_hz: float = SWA_HIGH_HZ
    smoothing_windows: int = 5
    time_bin_minutes: float = 60.0
    normalization: str = "recording_nrem_mean"
    reference_start_seconds: float | None = None
    reference_end_seconds: float | None = None
    chunk_windows: int = 2048

    def validate(self, sample_rate: float, epoch_seconds: float) -> None:
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError("Epoch duration must be positive")
        if not (
            math.isfinite(self.window_seconds)
            and math.isfinite(self.step_seconds)
            and self.window_seconds > 0
            and self.step_seconds > 0
        ):
            raise ValueError("SWA window and step must be positive")
        if not (
            0 < self.low_hz < self.high_hz < sample_rate / 2.0
        ):
            raise ValueError("SWA band must be inside the Nyquist range")
        resolution = 1.0 / self.window_seconds
        if not np.isclose(
            self.low_hz / resolution,
            round(self.low_hz / resolution),
            atol=1e-9,
        ):
            raise ValueError(
                "The selected window does not represent the 0.5 Hz "
                "boundary exactly"
            )
        if self.smoothing_windows < 1 or self.smoothing_windows % 2 == 0:
            raise ValueError(
                "SWA smoothing window count must be a positive odd number"
            )
        if self.time_bin_minutes <= 0:
            raise ValueError("Time-bin width must be positive")
        if self.chunk_windows < 1:
            raise ValueError("SWA chunk size must be positive")
        if self.normalization not in {
            "absolute",
            "recording_nrem_mean",
            "reference_interval_mean",
        }:
            raise ValueError(
                f"Unsupported SWA normalization: {self.normalization}"
            )
        if self.normalization == "reference_interval_mean":
            start = self.reference_start_seconds
            end = self.reference_end_seconds
            if (
                start is None
                or end is None
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError(
                    "Reference interval must satisfy 0 <= start < end"
                )
            if end - start < self.window_seconds:
                raise ValueError(
                    "Reference interval must be at least as long as the "
                    f"{self.window_seconds:g}-second SWA window"
                )


@dataclass
class SlowWaveActivityChannelResult:
    channel_index: int
    channel_label: str
    input_physical_unit: str
    window_start_seconds: np.ndarray
    window_end_seconds: np.ndarray
    window_center_seconds: np.ndarray
    window_run_index: np.ndarray
    raw_swa_uv2: np.ndarray
    smoothed_swa_uv2: np.ndarray
    smoothing_window_count: np.ndarray
    full_smoothing_context_mask: np.ndarray
    normalized_swa_percent: np.ndarray
    normalization_reference_uv2: float
    normalization_reference_windows: int
    epoch_raw_swa_uv2: np.ndarray
    epoch_smoothed_swa_uv2: np.ndarray
    epoch_normalized_swa_percent: np.ndarray
    epoch_valid_mask: np.ndarray
    epoch_exclusion_reason: np.ndarray


def _local_datetimes(
    recording_start_timestamp: float,
    offsets: np.ndarray,
) -> list[str]:
    if not recording_start_timestamp:
        return [""] * len(offsets)
    return [
        datetime.fromtimestamp(
            recording_start_timestamp + float(offset)
        )
        .astimezone()
        .isoformat(timespec="milliseconds")
        for offset in offsets
    ]


def _mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else math.nan


def _median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else math.nan


def _sample_sd(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan


def _sem(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (
        _sample_sd(finite) / math.sqrt(len(finite))
        if len(finite) > 1
        else math.nan
    )


def _check_cancel(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise SlowWaveActivityCancelled("SWA analysis was cancelled")


def _hann_psd(
    signal_windows: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = signal_windows.shape[1]
    taper = windows.hann(sample_count, sym=False)
    centered = signal_windows - np.mean(
        signal_windows,
        axis=1,
        keepdims=True,
    )
    transformed = np.fft.rfft(centered * taper, axis=1)
    psd = (
        np.square(np.abs(transformed))
        / (sample_rate * np.sum(np.square(taper)))
    )
    if sample_count % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    frequencies = np.fft.rfftfreq(
        sample_count,
        d=1.0 / sample_rate,
    )
    return frequencies, psd


def _integrate_band_exact(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    selected = frequencies[keep]
    if len(selected) < 2:
        raise ValueError("SWA band contains fewer than two frequency bins")
    if not np.isclose(selected[0], low_hz, atol=1e-9):
        raise ValueError(
            f"SWA lower boundary {low_hz:g} Hz is not represented exactly"
        )
    if not np.isclose(selected[-1], high_hz, atol=1e-9):
        raise ValueError(
            f"SWA upper boundary {high_hz:g} Hz is not represented exactly"
        )
    resolution = float(selected[1] - selected[0])
    if not np.allclose(
        np.diff(selected),
        resolution,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("SWA frequency bins are not uniformly spaced")
    return np.sum(psd[:, keep], axis=1) * resolution


def _rolling_median_within_runs(
    values: np.ndarray,
    run_indices: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(len(values), np.nan, dtype=float)
    counts = np.zeros(len(values), dtype=int)
    radius = width // 2
    for run_index in np.unique(run_indices):
        positions = np.flatnonzero(run_indices == run_index)
        run_values = values[positions]
        for local_index, position in enumerate(positions):
            low = max(0, local_index - radius)
            high = min(len(run_values), local_index + radius + 1)
            counts[position] = high - low
            if counts[position] == width:
                output[position] = np.median(run_values[low:high])
    return output, counts


def _valid_window_starts(
    sample_valid: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sample_valid) < window_samples:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    starts = np.arange(
        0,
        len(sample_valid) - window_samples + 1,
        step_samples,
        dtype=int,
    )
    invalid = (~sample_valid).astype(np.int32)
    cumulative = np.empty(len(invalid) + 1, dtype=np.int32)
    cumulative[0] = 0
    np.cumsum(invalid, dtype=np.int32, out=cumulative[1:])
    invalid_counts = (
        cumulative[starts + window_samples] - cumulative[starts]
    )
    starts = starts[invalid_counts == 0]
    if not len(starts):
        return starts, np.asarray([], dtype=int)
    run_indices = np.zeros(len(starts), dtype=int)
    run_indices[1:] = np.cumsum(np.diff(starts) != step_samples)
    return starts, run_indices


def _epoch_exclusion_reasons(
    scores: np.ndarray,
    artifact_mask: np.ndarray,
    finite_epochs: np.ndarray,
) -> np.ndarray:
    reasons = np.full(len(scores), "included_nrem", dtype=object)
    reasons[scores != 2] = "not_nrem"
    reasons[scores == 255] = "unscored"
    reasons[np.isin(scores, (4, 6, 7))] = "other_state"
    reasons[np.isin(scores, (129, 130, 131))] = "excluded_score"
    reasons[~finite_epochs] = "nonfinite_signal"
    reasons[scores == 0] = "score_artifact"
    reasons[artifact_mask & (scores != 0)] = "verified_artifact"
    return reasons


def _aggregate_windows_to_epochs(
    centers_seconds: np.ndarray,
    values: np.ndarray,
    epoch_count: int,
    epoch_seconds: float,
) -> np.ndarray:
    output = np.full(epoch_count, np.nan, dtype=float)
    epoch_indices = np.floor(centers_seconds / epoch_seconds).astype(int)
    epoch_indices = np.clip(epoch_indices, 0, epoch_count - 1)
    for epoch_index in np.unique(epoch_indices):
        output[epoch_index] = _median(values[epoch_indices == epoch_index])
    return output


def _normalization_reference(
    config: SlowWaveActivityConfig,
    starts_seconds: np.ndarray,
    ends_seconds: np.ndarray,
    smoothed_values: np.ndarray,
) -> tuple[float, int]:
    if config.normalization == "absolute":
        return math.nan, 0
    selected = np.isfinite(smoothed_values)
    if config.normalization == "reference_interval_mean":
        selected &= starts_seconds >= float(
            config.reference_start_seconds
        )
        selected &= ends_seconds <= float(
            config.reference_end_seconds
        )
    reference = _mean(smoothed_values[selected])
    reference_windows = int(np.sum(selected))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError(
            "The selected SWA normalization reference contains no "
            "positive full-context artifact-free NREM windows"
        )
    return reference, reference_windows


@dataclass
class SlowWaveActivityResult:
    config: SlowWaveActivityConfig
    sample_rate: float
    epoch_seconds: float
    scores: np.ndarray
    score_hash: str
    channel_results: list[SlowWaveActivityChannelResult]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def value_column(self) -> str:
        return (
            "smoothed_swa_uv2"
            if self.config.normalization == "absolute"
            else "normalized_swa_percent"
        )

    @property
    def value_unit(self) -> str:
        return (
            "uV^2"
            if self.config.normalization == "absolute"
            else "% reference NREM mean"
        )

    def window_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        start_timestamp = float(
            self.metadata.get("recording_start_timestamp", 0.0)
        )
        for channel in self.channel_results:
            frame = pd.DataFrame(
                {
                    "channel_index_0based": channel.channel_index,
                    "channel": channel.channel_label,
                    "window_index_0based": np.arange(
                        len(channel.window_center_seconds)
                    ),
                    "nrem_run_index_0based": channel.window_run_index,
                    "start_seconds": channel.window_start_seconds,
                    "end_seconds": channel.window_end_seconds,
                    "center_seconds": channel.window_center_seconds,
                    "center_datetime_local": _local_datetimes(
                        start_timestamp,
                        channel.window_center_seconds,
                    ),
                    "raw_swa_uv2": channel.raw_swa_uv2,
                    "smoothed_swa_uv2": channel.smoothed_swa_uv2,
                    "smoothing_window_count": (
                        channel.smoothing_window_count
                    ),
                    "has_full_5_window_context": (
                        channel.full_smoothing_context_mask
                    ),
                    "normalized_swa_percent": (
                        channel.normalized_swa_percent
                    ),
                }
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def epoch_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        starts = np.arange(len(self.scores), dtype=float) * self.epoch_seconds
        for channel in self.channel_results:
            frames.append(
                pd.DataFrame(
                    {
                        "channel_index_0based": channel.channel_index,
                        "channel": channel.channel_label,
                        "epoch_0based": np.arange(len(self.scores)),
                        "start_seconds": starts,
                        "stage_code": self.scores,
                        "is_artifact_free_nrem": (
                            channel.epoch_valid_mask
                        ),
                        "has_valid_swa": np.isfinite(
                            channel.epoch_smoothed_swa_uv2
                        ),
                        "exclusion_reason": (
                            channel.epoch_exclusion_reason
                        ),
                        "raw_swa_uv2": channel.epoch_raw_swa_uv2,
                        "smoothed_swa_uv2": (
                            channel.epoch_smoothed_swa_uv2
                        ),
                        "normalized_swa_percent": (
                            channel.epoch_normalized_swa_percent
                        ),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            values = channel.smoothed_swa_uv2
            normalized = channel.normalized_swa_percent
            rows.append(
                {
                    "channel_index_0based": channel.channel_index,
                    "channel": channel.channel_label,
                    "raw_candidate_windows": len(
                        channel.raw_swa_uv2
                    ),
                    "valid_windows": int(np.sum(np.isfinite(values))),
                    "artifact_free_nrem_epochs": int(
                        np.sum(channel.epoch_valid_mask)
                    ),
                    "valid_nrem_epochs": int(
                        np.sum(
                            np.isfinite(
                                channel.epoch_smoothed_swa_uv2
                            )
                        )
                    ),
                    "mean_swa_uv2": _mean(values),
                    "median_swa_uv2": _median(values),
                    "sd_swa_uv2": _sample_sd(values),
                    "sem_swa_uv2": _sem(values),
                    "normalization_reference_uv2": (
                        channel.normalization_reference_uv2
                    ),
                    "normalization_reference_windows": (
                        channel.normalization_reference_windows
                    ),
                    "mean_normalized_swa_percent": _mean(normalized),
                }
            )
        return pd.DataFrame(rows)

    def timecourse_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        bin_seconds = self.config.time_bin_minutes * 60.0
        recording_seconds = len(self.scores) * self.epoch_seconds
        total_bins = max(
            1,
            math.ceil(recording_seconds / bin_seconds),
        )
        for channel in self.channel_results:
            centers = channel.window_center_seconds
            bin_indices = np.floor(centers / bin_seconds).astype(int)
            for bin_index in range(total_bins):
                selected = (
                    (bin_indices == bin_index)
                    & np.isfinite(channel.smoothed_swa_uv2)
                )
                absolute = channel.smoothed_swa_uv2[selected]
                normalized = channel.normalized_swa_percent[selected]
                rows.append(
                    {
                        "channel_index_0based": channel.channel_index,
                        "channel": channel.channel_label,
                        "time_bin_index_0based": int(bin_index),
                        "time_bin_start_seconds": (
                            float(bin_index) * bin_seconds
                        ),
                        "time_bin_end_seconds": (
                            min(
                                float(bin_index + 1) * bin_seconds,
                                recording_seconds,
                            )
                        ),
                        "valid_windows": int(np.sum(selected)),
                        "mean_swa_uv2": _mean(absolute),
                        "median_swa_uv2": _median(absolute),
                        "sem_swa_uv2": _sem(absolute),
                        "mean_normalized_swa_percent": _mean(normalized),
                        "sem_normalized_swa_percent": _sem(normalized),
                    }
                )
        return pd.DataFrame(rows)

    def bout_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            valid = channel.epoch_valid_mask
            padded = np.concatenate(([False], valid, [False]))
            transitions = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(transitions == 1)
            ends = np.flatnonzero(transitions == -1)
            for bout_index, (start, end) in enumerate(zip(starts, ends)):
                start_seconds = float(start) * self.epoch_seconds
                end_seconds = float(end) * self.epoch_seconds
                selected = (
                    (channel.window_center_seconds >= start_seconds)
                    & (channel.window_center_seconds < end_seconds)
                    & np.isfinite(channel.smoothed_swa_uv2)
                )
                rows.append(
                    {
                        "channel_index_0based": channel.channel_index,
                        "channel": channel.channel_label,
                        "bout_index_0based": bout_index,
                        "start_epoch_0based": int(start),
                        "end_epoch_0based_inclusive": int(end - 1),
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "duration_seconds": end_seconds - start_seconds,
                        "valid_windows": int(np.sum(selected)),
                        "mean_swa_uv2": _mean(
                            channel.smoothed_swa_uv2[selected]
                        ),
                        "mean_normalized_swa_percent": _mean(
                            channel.normalized_swa_percent[selected]
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def qc_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            reasons, counts = np.unique(
                channel.epoch_exclusion_reason,
                return_counts=True,
            )
            for reason, count in zip(reasons, counts):
                rows.append(
                    {
                        "channel_index_0based": channel.channel_index,
                        "channel": channel.channel_label,
                        "category": "epoch_exclusion",
                        "reason": str(reason),
                        "count": int(count),
                    }
                )
            rows.append(
                {
                    "channel_index_0based": channel.channel_index,
                    "channel": channel.channel_label,
                    "category": "raw_candidate_windows",
                    "reason": "artifact_free_nrem_4s_windows",
                    "count": len(channel.window_center_seconds),
                }
            )
            rows.append(
                {
                    "channel_index_0based": channel.channel_index,
                    "channel": channel.channel_label,
                    "category": "valid_smoothed_windows",
                    "reason": "full_5_window_context",
                    "count": int(
                        np.sum(channel.full_smoothing_context_mask)
                    ),
                }
            )
        return pd.DataFrame(rows)


def analyze_slow_wave_activity(
    channels: Sequence[tuple[int, str, np.ndarray, str]],
    sample_rate: float,
    scores: np.ndarray,
    epoch_seconds: float,
    *,
    config: SlowWaveActivityConfig | None = None,
    artifact_masks: Mapping[int, np.ndarray] | None = None,
    metadata: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> SlowWaveActivityResult:
    selected = config or SlowWaveActivityConfig()
    selected.validate(sample_rate, epoch_seconds)
    if not channels:
        raise ValueError("Select at least one EEG channel")
    score_values = np.asarray(scores, dtype=int).reshape(-1)
    if not len(score_values):
        raise ValueError("No sleep scores are available")
    samples_per_epoch = round(sample_rate * epoch_seconds)
    window_samples = round(sample_rate * selected.window_seconds)
    step_samples = round(sample_rate * selected.step_seconds)
    if window_samples < 4 or step_samples < 1:
        raise ValueError("SWA window contains too few samples")
    expected_samples = len(score_values) * samples_per_epoch
    masks = artifact_masks or {}
    results: list[SlowWaveActivityChannelResult] = []

    for position, (index, label, values, unit) in enumerate(channels):
        _check_cancel(cancelled)
        if progress is not None:
            progress(
                int(position * 90 / len(channels)),
                f"Calculating SWA: {label}",
            )
        signal = signal_in_microvolts(values, unit)
        if len(signal) < expected_samples:
            raise ValueError(
                f"{label}: EEG contains {len(signal)} samples, but "
                f"{expected_samples} are required by the scores"
            )
        signal = signal[:expected_samples]
        epochs = signal.reshape(len(score_values), samples_per_epoch)
        finite_epochs = np.all(np.isfinite(epochs), axis=1)
        channel_artifact = np.asarray(
            masks.get(int(index), np.zeros(len(score_values), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if len(channel_artifact) != len(score_values):
            raise ValueError(
                f"{label}: Artifact mask length does not match scores"
            )
        artifacts = effective_artifact_mask(
            score_values,
            (channel_artifact,),
        )
        reasons = _epoch_exclusion_reasons(
            score_values,
            artifacts,
            finite_epochs,
        )
        valid_epochs = reasons == "included_nrem"
        sample_valid = np.repeat(valid_epochs, samples_per_epoch)
        sample_valid &= np.isfinite(signal)
        starts, run_indices = _valid_window_starts(
            sample_valid,
            window_samples,
            step_samples,
        )
        if not len(starts):
            raise ValueError(
                f"{label}: not enough artifact-free NREM data for "
                "a 4-second SWA window"
            )
        window_offsets = np.arange(window_samples, dtype=int)[None, :]
        raw_chunks: list[np.ndarray] = []
        for chunk_start in range(0, len(starts), selected.chunk_windows):
            _check_cancel(cancelled)
            chunk_end = min(
                len(starts),
                chunk_start + selected.chunk_windows,
            )
            chunk_indices = (
                starts[chunk_start:chunk_end, None] + window_offsets
            )
            signal_windows = signal[chunk_indices]
            chunk_frequencies, psd = _hann_psd(
                signal_windows,
                sample_rate,
            )
            raw_chunks.append(
                _integrate_band_exact(
                    chunk_frequencies,
                    psd,
                    selected.low_hz,
                    selected.high_hz,
                )
            )
            if progress is not None:
                within_channel = chunk_end / len(starts)
                progress(
                    int(
                        (
                            position
                            + 0.9 * within_channel
                        )
                        * 100
                        / len(channels)
                    ),
                    f"Calculating SWA: {label}",
                )
        raw = np.concatenate(raw_chunks)
        smoothed, smoothing_counts = _rolling_median_within_runs(
            raw,
            run_indices,
            selected.smoothing_windows,
        )
        full_smoothing_context = (
            smoothing_counts == selected.smoothing_windows
        )
        starts_seconds = starts / sample_rate
        ends_seconds = (starts + window_samples) / sample_rate
        centers = (starts + window_samples / 2.0) / sample_rate
        reference, reference_windows = _normalization_reference(
            selected,
            starts_seconds,
            ends_seconds,
            smoothed,
        )
        normalized = (
            np.full(len(smoothed), np.nan, dtype=float)
            if selected.normalization == "absolute"
            else 100.0 * smoothed / reference
        )
        epoch_raw = _aggregate_windows_to_epochs(
            centers,
            raw,
            len(score_values),
            epoch_seconds,
        )
        epoch_smoothed = _aggregate_windows_to_epochs(
            centers,
            smoothed,
            len(score_values),
            epoch_seconds,
        )
        epoch_normalized = _aggregate_windows_to_epochs(
            centers,
            normalized,
            len(score_values),
            epoch_seconds,
        )
        results.append(
            SlowWaveActivityChannelResult(
                channel_index=int(index),
                channel_label=str(label),
                input_physical_unit=str(unit),
                window_start_seconds=starts_seconds,
                window_end_seconds=ends_seconds,
                window_center_seconds=centers,
                window_run_index=run_indices,
                raw_swa_uv2=raw,
                smoothed_swa_uv2=smoothed,
                smoothing_window_count=smoothing_counts,
                full_smoothing_context_mask=full_smoothing_context,
                normalized_swa_percent=normalized,
                normalization_reference_uv2=reference,
                normalization_reference_windows=reference_windows,
                epoch_raw_swa_uv2=epoch_raw,
                epoch_smoothed_swa_uv2=epoch_smoothed,
                epoch_normalized_swa_percent=epoch_normalized,
                epoch_valid_mask=valid_epochs,
                epoch_exclusion_reason=reasons,
            )
        )

    if progress is not None:
        progress(100, "SWA analysis complete")
    return SlowWaveActivityResult(
        config=selected,
        sample_rate=float(sample_rate),
        epoch_seconds=float(epoch_seconds),
        scores=score_values.copy(),
        score_hash=score_fingerprint(score_values, epoch_seconds),
        channel_results=results,
        metadata=dict(metadata or {}),
    )


def _atomic_text(destination: Path, payload: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _tsv_payload(frame: pd.DataFrame) -> str:
    return frame.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
        na_rep="",
    )


def export_swa_bundle(
    directory: Path,
    result: SlowWaveActivityResult,
    *,
    prefix: str = "swa",
    overwrite: bool = False,
    plot_figure: object | None = None,
) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        prefix,
    ).strip("._") or "swa"
    paths = {
        "summary": root / f"{safe_prefix}_summary.tsv",
        "timecourse": root / f"{safe_prefix}_timecourse.tsv",
        "windows": root / f"{safe_prefix}_windows.tsv",
        "epochs": root / f"{safe_prefix}_epochs.tsv",
        "nrem_bouts": root / f"{safe_prefix}_nrem_bouts.tsv",
        "qc": root / f"{safe_prefix}_qc.tsv",
        "metadata": root / f"{safe_prefix}_metadata.json",
    }
    if plot_figure is not None:
        paths["plot_png"] = root / f"{safe_prefix}_plot.png"
        paths["plot_svg"] = root / f"{safe_prefix}_plot.svg"
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "SWA export already exists: "
            + ", ".join(path.name for path in existing)
        )
    metadata = {
        "schema_version": 2,
        "analysis": "Sleep Workbench Slow-Wave Activity",
        "analysis_version": SWA_ANALYSIS_VERSION,
        "definition": "NREM EEG power integrated from 0.5 to 4.0 Hz",
        "config": asdict(result.config),
        "sample_rate": result.sample_rate,
        "epoch_seconds": result.epoch_seconds,
        "score_sha256": result.score_hash,
        "score_fingerprint_schema": SCORE_FINGERPRINT_SCHEMA,
        "power_unit": "uV^2",
        "normalized_unit": "% reference NREM mean",
        "spectral_estimator": {
            "window": "periodic_hann",
            "window_seconds": result.config.window_seconds,
            "step_seconds": result.config.step_seconds,
            "overlap_seconds": (
                result.config.window_seconds
                - result.config.step_seconds
            ),
            "detrend": "remove_window_mean",
            "one_sided": True,
            "frequency_resolution_hz": (
                1.0 / result.config.window_seconds
            ),
            "band_integration": (
                "psd_bin_sum_times_frequency_resolution_"
                "inclusive_exact_boundaries"
            ),
            "smoothing": (
                "strict_centered_5_window_median_within_"
                "uninterrupted_valid_nrem_run"
            ),
            "smoothing_boundary_policy": (
                "requires_full_context; first_and_last_two_windows_"
                "per_run_are_not_smoothed"
            ),
            "normalization_reference_policy": (
                "entire_4s_window_must_be_contained_in_reference_interval"
            ),
            "time_bin_policy": (
                "continuous_bins_across_complete_recording; "
                "empty_bins_are_exported_with_NA"
            ),
        },
        "source": result.metadata,
        "channels": [
            {
                "index_0based": channel.channel_index,
                "label": channel.channel_label,
                "input_physical_unit": channel.input_physical_unit,
                "raw_candidate_windows": len(
                    channel.window_center_seconds
                ),
                "valid_smoothed_windows": int(
                    np.sum(channel.full_smoothing_context_mask)
                ),
                "valid_nrem_epochs": int(
                    np.sum(
                        np.isfinite(
                            channel.epoch_smoothed_swa_uv2
                        )
                    )
                ),
                "normalization_reference_uv2": (
                    channel.normalization_reference_uv2
                ),
                "normalization_reference_windows": (
                    channel.normalization_reference_windows
                ),
            }
            for channel in result.channel_results
        ],
    }
    payloads = {
        "summary": _tsv_payload(result.summary_frame()),
        "timecourse": _tsv_payload(result.timecourse_frame()),
        "windows": _tsv_payload(result.window_frame()),
        "epochs": _tsv_payload(result.epoch_frame()),
        "nrem_bouts": _tsv_payload(result.bout_frame()),
        "qc": _tsv_payload(result.qc_frame()),
        "metadata": json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{safe_prefix}.", dir=root)
    )
    staged: dict[str, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for key, payload in payloads.items():
            staged[key] = staging / paths[key].name
            _atomic_text(staged[key], payload)
        if plot_figure is not None:
            for key, file_format in (
                ("plot_png", "png"),
                ("plot_svg", "svg"),
            ):
                staged[key] = staging / paths[key].name
                plot_figure.savefig(
                    staged[key],
                    format=file_format,
                    dpi=300 if file_format == "png" else None,
                    bbox_inches="tight",
                )
        for key, destination in paths.items():
            if destination.exists():
                backup = staging / f"{destination.name}.backup"
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(staged[key], destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return paths
