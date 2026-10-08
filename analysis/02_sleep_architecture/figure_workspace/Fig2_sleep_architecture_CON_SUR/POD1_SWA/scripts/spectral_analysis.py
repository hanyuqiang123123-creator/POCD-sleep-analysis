from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import windows

from sleepworkbench.analysis_fingerprint import (
    SCORE_FINGERPRINT_SCHEMA,
    score_fingerprint,
)
from sleepworkbench.artifact_store import effective_artifact_mask


ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]
SPECTRAL_ANALYSIS_VERSION = "1.0.0"

STAGE_NAMES = {
    1: "Wake",
    2: "NREM",
    3: "REM",
}


class SpectralAnalysisCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    low_hz: float
    high_hz: float

    def validate(self, nyquist: float) -> None:
        if not self.name.strip():
            raise ValueError("Frequency-band names cannot be blank")
        if (
            not math.isfinite(self.low_hz)
            or not math.isfinite(self.high_hz)
            or not 0 <= self.low_hz < self.high_hz <= nyquist
        ):
            raise ValueError(
                f"Invalid band {self.name}: "
                f"{self.low_hz:g}-{self.high_hz:g} Hz"
            )


def nature_frequency_bands() -> tuple[FrequencyBand, ...]:
    return (
        FrequencyBand("Delta", 1.0, 4.4),
        FrequencyBand("Delta1", 1.0, 1.6),
        FrequencyBand("Delta2", 2.4, 3.6),
        FrequencyBand("Theta", 6.0, 9.0),
        FrequencyBand("Alpha", 9.0, 13.0),
        FrequencyBand("Beta", 13.0, 25.0),
    )


def autofq_frequency_bands() -> tuple[FrequencyBand, ...]:
    return (
        FrequencyBand("Delta", 0.5, 4.0),
        FrequencyBand("Theta", 6.0, 8.0),
        FrequencyBand("Sigma", 8.0, 20.0),
        FrequencyBand("SigmaNarrow", 10.0, 15.0),
        FrequencyBand("Beta", 15.0, 30.0),
        FrequencyBand("Gamma", 30.0, 80.0),
    )


@dataclass(frozen=True)
class SpectralAnalysisConfig:
    preset: str = "nature_corrected"
    method: str = "multitaper"
    frequency_low_hz: float = 1.0
    frequency_high_hz: float = 25.0
    reference_low_hz: float = 1.0
    reference_high_hz: float = 25.0
    bands: tuple[FrequencyBand, ...] = field(
        default_factory=nature_frequency_bands
    )
    states: tuple[int, ...] = (1, 2, 3)
    time_bin_minutes: float = 60.0
    taper_count: int = 5
    time_bandwidth: float = 3.0
    chunk_epochs: int = 256
    export_epoch_spectra: bool = False

    @classmethod
    def nature_corrected(cls) -> "SpectralAnalysisConfig":
        return cls()

    @classmethod
    def nature_legacy(cls) -> "SpectralAnalysisConfig":
        return cls(
            preset="nature_matlab_legacy",
            method="legacy_welch",
            taper_count=1,
        )

    @classmethod
    def autofq(cls, *, sample_rate: float = 200.0) -> "SpectralAnalysisConfig":
        nyquist = float(sample_rate) / 2.0
        high = min(80.0, math.nextafter(nyquist, 0.0))
        bands = tuple(
            FrequencyBand(item.name, item.low_hz, min(item.high_hz, high))
            for item in autofq_frequency_bands()
            if item.low_hz < high
        )
        return cls(
            preset="autofq_features",
            frequency_low_hz=0.5,
            frequency_high_hz=high,
            reference_low_hz=0.5,
            reference_high_hz=high,
            bands=bands,
        )

    @property
    def is_legacy(self) -> bool:
        return self.method == "legacy_welch"

    @property
    def absolute_power_unit(self) -> str:
        return "PSD-bin sum (legacy)" if self.is_legacy else "uV^2"

    @property
    def relative_spectrum_unit(self) -> str:
        return "%/bin" if self.is_legacy else "%/Hz"

    def validate(
        self,
        sample_rate: float,
        epoch_seconds: float,
    ) -> None:
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError("Epoch duration must be positive")
        nyquist = sample_rate / 2.0
        if self.method not in {"multitaper", "legacy_welch"}:
            raise ValueError(f"Unsupported spectral method: {self.method}")
        if not (
            0 <= self.frequency_low_hz
            < self.frequency_high_hz
            <= nyquist
        ):
            raise ValueError(
                "Spectrum range must satisfy "
                f"0 <= low < high <= {nyquist:g} Hz"
            )
        if not (
            self.frequency_low_hz
            <= self.reference_low_hz
            < self.reference_high_hz
            <= self.frequency_high_hz
        ):
            raise ValueError(
                "Relative-power reference must be inside the spectrum range"
            )
        if not self.bands:
            raise ValueError("At least one frequency band is required")
        names: set[str] = set()
        column_keys: set[str] = set()
        resolution = sample_rate / int(round(sample_rate * epoch_seconds))
        for band in self.bands:
            band.validate(nyquist)
            if (
                band.low_hz < self.frequency_low_hz
                or band.high_hz > self.frequency_high_hz
            ):
                raise ValueError(
                    f"Band {band.name} must be inside the selected "
                    "spectrum range"
                )
            key = band.name.strip().casefold()
            if key in names:
                raise ValueError(
                    f"Duplicate frequency-band name: {band.name}"
                )
            names.add(key)
            column_key = _column_key(band.name)
            if column_key in column_keys:
                raise ValueError(
                    "Frequency-band names produce duplicate export "
                    f"columns: {band.name}"
                )
            column_keys.add(column_key)
            if band.high_hz - band.low_hz < resolution:
                raise ValueError(
                    f"Band {band.name} is narrower than the "
                    f"{resolution:g} Hz epoch frequency interval"
                )
        invalid_states = sorted(set(self.states) - set(STAGE_NAMES))
        if invalid_states:
            raise ValueError(
                "Unsupported sleep states: "
                + ", ".join(str(value) for value in invalid_states)
            )
        if not self.states:
            raise ValueError("Select at least one sleep state")
        if self.time_bin_minutes <= 0:
            raise ValueError("Time-bin width must be positive")
        if self.taper_count < 1:
            raise ValueError("Taper count must be positive")
        if self.chunk_epochs < 1:
            raise ValueError("Chunk size must be positive")
        samples_per_epoch = int(round(sample_rate * epoch_seconds))
        if samples_per_epoch < 4:
            raise ValueError("Epochs contain too few samples")
        if self.reference_high_hz - self.reference_low_hz < resolution:
            raise ValueError(
                "Relative-power reference is narrower than one "
                "frequency interval"
            )


@dataclass
class SpectralChannelResult:
    channel_index: int
    channel_label: str
    physical_unit: str
    frequencies_hz: np.ndarray
    epoch_psd_uv2_per_hz: np.ndarray
    epoch_relative_spectrum: np.ndarray
    absolute_band_power: np.ndarray
    relative_band_power_percent: np.ndarray
    valid_mask: np.ndarray
    exclusion_reasons: np.ndarray


@dataclass
class SpectralAnalysisResult:
    config: SpectralAnalysisConfig
    sample_rate: float
    epoch_seconds: float
    scores: np.ndarray
    score_hash: str
    channel_results: list[SpectralChannelResult]
    metadata: dict[str, object] = field(default_factory=dict)

    def spectrum_summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            for stage in self.config.states:
                selected = channel.valid_mask & (self.scores == stage)
                values = channel.epoch_psd_uv2_per_hz[selected]
                relative = channel.epoch_relative_spectrum[selected]
                count = int(len(values))
                if count:
                    mean = np.mean(values, axis=0)
                    relative_mean = np.mean(relative, axis=0)
                    if count > 1:
                        sd = np.std(values, axis=0, ddof=1)
                        relative_sd = np.std(relative, axis=0, ddof=1)
                    else:
                        sd = np.full(values.shape[1], np.nan)
                        relative_sd = np.full(values.shape[1], np.nan)
                    sem = sd / math.sqrt(count)
                    relative_sem = relative_sd / math.sqrt(count)
                else:
                    length = len(channel.frequencies_hz)
                    mean = np.full(length, np.nan)
                    sd = np.full(length, np.nan)
                    sem = np.full(length, np.nan)
                    relative_mean = np.full(length, np.nan)
                    relative_sd = np.full(length, np.nan)
                    relative_sem = np.full(length, np.nan)
                for index, frequency in enumerate(channel.frequencies_hz):
                    rows.append(
                        {
                            "channel_index_0based": channel.channel_index,
                            "channel": channel.channel_label,
                            "stage_code": stage,
                            "stage": STAGE_NAMES[stage],
                            "valid_epochs": count,
                            "frequency_hz": float(frequency),
                            "mean_psd_uv2_per_hz": float(mean[index]),
                            "sd_psd_uv2_per_hz": float(sd[index]),
                            "sem_psd_uv2_per_hz": float(sem[index]),
                            "mean_relative_spectrum": float(
                                relative_mean[index]
                            ),
                            "sd_relative_spectrum": float(
                                relative_sd[index]
                            ),
                            "sem_relative_spectrum": float(
                                relative_sem[index]
                            ),
                            "relative_spectrum_unit": (
                                self.config.relative_spectrum_unit
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def band_summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            for stage in self.config.states:
                selected = channel.valid_mask & (self.scores == stage)
                count = int(np.sum(selected))
                for band_index, band in enumerate(self.config.bands):
                    absolute = channel.absolute_band_power[
                        selected, band_index
                    ]
                    relative = channel.relative_band_power_percent[
                        selected, band_index
                    ]
                    rows.append(
                        {
                            "channel_index_0based": channel.channel_index,
                            "channel": channel.channel_label,
                            "stage_code": stage,
                            "stage": STAGE_NAMES[stage],
                            "band": band.name,
                            "low_hz": band.low_hz,
                            "high_hz": band.high_hz,
                            "valid_epochs": count,
                            "valid_minutes": (
                                count * self.epoch_seconds / 60.0
                            ),
                            "absolute_power_unit": (
                                self.config.absolute_power_unit
                            ),
                            "absolute_mean": _mean(absolute),
                            "absolute_sd": _sample_sd(absolute),
                            "absolute_sem": _sem(absolute),
                            "relative_mean_percent": _mean(relative),
                            "relative_sd_percent": _sample_sd(relative),
                            "relative_sem_percent": _sem(relative),
                        }
                    )
        return pd.DataFrame(rows)

    def timecourse_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        bin_seconds = self.config.time_bin_minutes * 60.0
        epoch_bins = np.floor(
            np.arange(len(self.scores), dtype=float)
            * self.epoch_seconds
            / bin_seconds
        ).astype(int)
        if not len(epoch_bins):
            return pd.DataFrame()
        for channel in self.channel_results:
            for bin_index in range(int(epoch_bins[-1]) + 1):
                in_bin = epoch_bins == bin_index
                start_seconds = bin_index * bin_seconds
                end_seconds = min(
                    (bin_index + 1) * bin_seconds,
                    len(self.scores) * self.epoch_seconds,
                )
                for stage in self.config.states:
                    selected = (
                        channel.valid_mask
                        & in_bin
                        & (self.scores == stage)
                    )
                    count = int(np.sum(selected))
                    for band_index, band in enumerate(self.config.bands):
                        absolute = channel.absolute_band_power[
                            selected, band_index
                        ]
                        relative = channel.relative_band_power_percent[
                            selected, band_index
                        ]
                        rows.append(
                            {
                                "channel_index_0based": (
                                    channel.channel_index
                                ),
                                "channel": channel.channel_label,
                                "time_bin_index_0based": bin_index,
                                "time_bin_start_seconds": start_seconds,
                                "time_bin_end_seconds": end_seconds,
                                "time_bin_start_minutes": (
                                    start_seconds / 60.0
                                ),
                                "time_bin_end_minutes": (
                                    end_seconds / 60.0
                                ),
                                "stage_code": stage,
                                "stage": STAGE_NAMES[stage],
                                "band": band.name,
                                "low_hz": band.low_hz,
                                "high_hz": band.high_hz,
                                "valid_epochs": count,
                                "absolute_power_unit": (
                                    self.config.absolute_power_unit
                                ),
                                "absolute_mean": _mean(absolute),
                                "absolute_sd": _sample_sd(absolute),
                                "absolute_sem": _sem(absolute),
                                "relative_mean_percent": _mean(relative),
                                "relative_sd_percent": _sample_sd(relative),
                                "relative_sem_percent": _sem(relative),
                            }
                        )
        return pd.DataFrame(rows)

    def epoch_band_power_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        recording_start = float(
            self.metadata.get("recording_start_timestamp") or 0.0
        )
        for channel in self.channel_results:
            epoch_index = np.arange(len(self.scores), dtype=int)
            frame = pd.DataFrame(
                {
                    "channel_index_0based": channel.channel_index,
                    "channel": channel.channel_label,
                    "epoch_0based": epoch_index,
                    "start_seconds": epoch_index * self.epoch_seconds,
                    "start_datetime_local": _local_datetimes(
                        recording_start,
                        epoch_index * self.epoch_seconds,
                    ),
                    "stage_code": self.scores,
                    "stage": [
                        STAGE_NAMES.get(int(value), f"Code {int(value)}")
                        for value in self.scores
                    ],
                    "included": channel.valid_mask,
                    "exclusion_reason": channel.exclusion_reasons,
                }
            )
            for band_index, band in enumerate(self.config.bands):
                key = _column_key(band.name)
                frame[f"{key}_absolute"] = channel.absolute_band_power[
                    :, band_index
                ]
                frame[f"{key}_relative_percent"] = (
                    channel.relative_band_power_percent[:, band_index]
                )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def epoch_qc_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for channel in self.channel_results:
            reasons, counts = np.unique(
                channel.exclusion_reasons,
                return_counts=True,
            )
            for reason, count in zip(reasons, counts, strict=True):
                rows.append(
                    {
                        "channel_index_0based": channel.channel_index,
                        "channel": channel.channel_label,
                        "status": (
                            "included"
                            if str(reason) == "included"
                            else "excluded"
                        ),
                        "reason": str(reason),
                        "epochs": int(count),
                        "seconds": float(count * self.epoch_seconds),
                        "minutes": float(
                            count * self.epoch_seconds / 60.0
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def epoch_spectra_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for channel in self.channel_results:
            selected = np.flatnonzero(channel.valid_mask)
            if not len(selected):
                continue
            values = channel.epoch_psd_uv2_per_hz[selected]
            frequency_columns = [
                f"{frequency:.6f}"
                for frequency in channel.frequencies_hz
            ]
            frame = pd.DataFrame(values, columns=frequency_columns)
            frame.insert(0, "stage_code", self.scores[selected])
            frame.insert(0, "epoch_0based", selected)
            frame.insert(0, "channel", channel.channel_label)
            frame.insert(
                0,
                "channel_index_0based",
                channel.channel_index,
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def _sem(values: np.ndarray) -> float:
    return (
        _sample_sd(values) / math.sqrt(len(values))
        if len(values) > 1
        else math.nan
    )


def _column_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return key or "band"


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


def signal_in_microvolts(
    values: np.ndarray,
    physical_unit: str,
) -> np.ndarray:
    normalized = str(physical_unit).strip().lower()
    factors = {
        "uv": 1.0,
        "µv": 1.0,
        "μv": 1.0,
        "microvolt": 1.0,
        "microvolts": 1.0,
        "mv": 1_000.0,
        "millivolt": 1_000.0,
        "millivolts": 1_000.0,
        "v": 1_000_000.0,
        "volt": 1_000_000.0,
        "volts": 1_000_000.0,
        "nv": 0.001,
        "nanovolt": 0.001,
        "nanovolts": 0.001,
    }
    if normalized not in factors:
        raise ValueError(
            "Unsupported EEG physical unit for spectral analysis: "
            f"{physical_unit or '(blank)'}"
        )
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    factor = factors[normalized]
    return signal if factor == 1.0 else signal * factor


def _check_cancel(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise SpectralAnalysisCancelled("Spectral analysis was cancelled")


def _sine_tapers(
    sample_count: int,
    taper_count: int,
) -> np.ndarray:
    sample_index = np.arange(1, sample_count + 1, dtype=float)
    return np.asarray(
        [
            np.sqrt(2.0 / (sample_count + 1.0))
            * np.sin(
                np.pi
                * taper
                * sample_index
                / (sample_count + 1.0)
            )
            for taper in range(1, taper_count + 1)
        ]
    )


def _one_sided_psd(
    epochs: np.ndarray,
    sample_rate: float,
    config: SpectralAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = epochs.shape[1]
    frequencies = np.fft.rfftfreq(
        sample_count,
        d=1.0 / sample_rate,
    )
    if config.method == "legacy_welch":
        tapers = windows.hann(sample_count, sym=True)[None, :]
    else:
        tapers = _sine_tapers(sample_count, config.taper_count)
    normalization = sample_rate * np.sum(tapers[0] ** 2)
    psd = np.zeros(
        (len(epochs), len(frequencies)),
        dtype=np.float64,
    )
    for taper in tapers:
        transformed = np.fft.rfft(epochs * taper, axis=1)
        psd += np.square(np.abs(transformed)) / normalization
    psd /= len(tapers)
    if sample_count % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0
    return frequencies, psd


def _frequency_mask(
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    return (frequencies >= low_hz) & (frequencies <= high_hz)


def _reduce_frequency_range(
    psd: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    legacy: bool,
) -> np.ndarray:
    keep = _frequency_mask(frequencies, low_hz, high_hz)
    if int(np.sum(keep)) < 2:
        raise ValueError(
            f"Frequency range {low_hz:g}-{high_hz:g} Hz contains "
            "fewer than two bins"
        )
    if legacy:
        return np.sum(psd[:, keep], axis=1)
    return np.trapezoid(
        psd[:, keep],
        frequencies[keep],
        axis=1,
    )


def _exclusion_reasons(
    scores: np.ndarray,
    artifact_mask: np.ndarray,
    finite_epochs: np.ndarray,
    selected_states: Sequence[int],
) -> np.ndarray:
    output = np.full(len(scores), "included", dtype=object)
    output[~np.isin(scores, selected_states)] = "state_not_selected"
    output[scores == 255] = "unscored"
    output[np.isin(scores, (4, 6, 7))] = "other_state"
    output[np.isin(scores, (129, 130, 131))] = "excluded_score"
    output[~finite_epochs] = "nonfinite_signal"
    output[scores == 0] = "score_artifact"
    output[artifact_mask & (scores != 0)] = "verified_artifact"
    return output


def analyze_multichannel_spectra(
    channels: Sequence[tuple[int, str, np.ndarray, str]],
    sample_rate: float,
    scores: np.ndarray,
    epoch_seconds: float,
    *,
    config: SpectralAnalysisConfig | None = None,
    artifact_masks: Mapping[int, np.ndarray] | None = None,
    metadata: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> SpectralAnalysisResult:
    selected = config or SpectralAnalysisConfig.nature_corrected()
    selected.validate(sample_rate, epoch_seconds)
    if not channels:
        raise ValueError("Select at least one EEG channel")
    score_values = np.asarray(scores, dtype=int).reshape(-1)
    if not len(score_values):
        raise ValueError("No sleep scores are available")
    samples_per_epoch = int(round(sample_rate * epoch_seconds))
    expected_samples = len(score_values) * samples_per_epoch
    masks = artifact_masks or {}
    results: list[SpectralChannelResult] = []
    total_channels = len(channels)

    for channel_position, (index, label, values, unit) in enumerate(channels):
        _check_cancel(cancelled)
        signal = signal_in_microvolts(values, unit)
        if len(signal) < expected_samples:
            raise ValueError(
                f"{label}: EEG contains {len(signal)} samples, but "
                f"{expected_samples} are required by the scores"
            )
        epochs = signal[:expected_samples].reshape(
            len(score_values),
            samples_per_epoch,
        )
        finite_epochs = np.all(np.isfinite(epochs), axis=1)
        channel_artifact = np.asarray(
            masks.get(int(index), np.zeros(len(score_values), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if len(channel_artifact) != len(score_values):
            raise ValueError(
                f"{label}: Artifact mask length does not match scores"
            )
        effective_artifacts = effective_artifact_mask(
            score_values,
            (channel_artifact,),
        )
        reasons = _exclusion_reasons(
            score_values,
            effective_artifacts,
            finite_epochs,
            selected.states,
        )
        valid = reasons == "included"
        if not np.any(valid):
            raise ValueError(
                f"{label}: no artifact-free selected-state epochs"
            )

        result_frequencies: np.ndarray | None = None
        epoch_psd: np.ndarray | None = None
        epoch_relative: np.ndarray | None = None
        absolute_bands = np.full(
            (len(score_values), len(selected.bands)),
            np.nan,
            dtype=np.float64,
        )
        relative_bands = np.full_like(absolute_bands, np.nan)
        valid_indices = np.flatnonzero(valid)
        for chunk_start in range(0, len(valid_indices), selected.chunk_epochs):
            _check_cancel(cancelled)
            chunk_indices = valid_indices[
                chunk_start : chunk_start + selected.chunk_epochs
            ]
            frequencies, full_psd = _one_sided_psd(
                epochs[chunk_indices],
                sample_rate,
                selected,
            )
            spectrum_keep = _frequency_mask(
                frequencies,
                selected.frequency_low_hz,
                selected.frequency_high_hz,
            )
            if result_frequencies is None:
                result_frequencies = frequencies[spectrum_keep]
                epoch_psd = np.full(
                    (len(score_values), int(np.sum(spectrum_keep))),
                    np.nan,
                    dtype=np.float32,
                )
                epoch_relative = np.full_like(epoch_psd, np.nan)
            reference_power = _reduce_frequency_range(
                full_psd,
                frequencies,
                selected.reference_low_hz,
                selected.reference_high_hz,
                legacy=selected.is_legacy,
            )
            if np.any(
                ~np.isfinite(reference_power) | (reference_power <= 0)
            ):
                raise ValueError(
                    f"{label}: relative-power reference contains "
                    "non-positive values"
                )
            selected_psd = full_psd[:, spectrum_keep]
            if selected.is_legacy:
                relative_spectrum = (
                    selected_psd / reference_power[:, None] * 100.0
                )
            else:
                relative_spectrum = (
                    selected_psd / reference_power[:, None] * 100.0
                )
            assert epoch_psd is not None
            assert epoch_relative is not None
            epoch_psd[chunk_indices] = selected_psd.astype(np.float32)
            epoch_relative[chunk_indices] = relative_spectrum.astype(
                np.float32
            )
            for band_index, band in enumerate(selected.bands):
                absolute = _reduce_frequency_range(
                    full_psd,
                    frequencies,
                    band.low_hz,
                    band.high_hz,
                    legacy=selected.is_legacy,
                )
                absolute_bands[chunk_indices, band_index] = absolute
                relative_bands[chunk_indices, band_index] = (
                    absolute / reference_power * 100.0
                )
            if progress is not None:
                within = (
                    chunk_start + len(chunk_indices)
                ) / len(valid_indices)
                percent = int(
                    round(
                        100.0
                        * (channel_position + within)
                        / total_channels
                    )
                )
                progress(
                    percent,
                    f"{label}: spectra "
                    f"{min(chunk_start + len(chunk_indices), len(valid_indices))}"
                    f"/{len(valid_indices)} epochs",
                )

        assert result_frequencies is not None
        assert epoch_psd is not None
        assert epoch_relative is not None
        results.append(
            SpectralChannelResult(
                channel_index=int(index),
                channel_label=str(label),
                physical_unit="uV",
                frequencies_hz=result_frequencies,
                epoch_psd_uv2_per_hz=epoch_psd,
                epoch_relative_spectrum=epoch_relative,
                absolute_band_power=absolute_bands,
                relative_band_power_percent=relative_bands,
                valid_mask=valid,
                exclusion_reasons=np.asarray(reasons, dtype=str),
            )
        )
    if progress is not None:
        progress(100, "Spectral analysis complete")
    return SpectralAnalysisResult(
        config=selected,
        sample_rate=float(sample_rate),
        epoch_seconds=float(epoch_seconds),
        scores=score_values.copy(),
        score_hash=score_fingerprint(score_values, epoch_seconds),
        channel_results=results,
        metadata=dict(metadata or {}),
    )


def _atomic_text(path: Path, payload: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def export_spectral_bundle(
    directory: Path,
    result: SpectralAnalysisResult,
    *,
    prefix: str = "spectral",
    overwrite: bool = False,
) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        prefix,
    ).strip("._")
    safe_prefix = safe_prefix or "spectral"
    paths = {
        "spectrum_summary": root / f"{safe_prefix}_spectrum_summary.tsv",
        "band_summary": root / f"{safe_prefix}_band_summary.tsv",
        "band_timecourse": root / f"{safe_prefix}_band_timecourse.tsv",
        "epoch_band_power": root / f"{safe_prefix}_epoch_band_power.tsv",
        "epoch_qc": root / f"{safe_prefix}_epoch_qc.tsv",
        "metadata": root / f"{safe_prefix}_metadata.json",
    }
    if result.config.export_epoch_spectra:
        paths["epoch_spectra"] = (
            root / f"{safe_prefix}_epoch_spectra.tsv"
        )
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Spectral export already exists: "
            + ", ".join(path.name for path in existing)
        )
    metadata = {
        "schema_version": 1,
        "analysis": "Sleep Workbench Spectral & Band Power Analysis",
        "analysis_version": SPECTRAL_ANALYSIS_VERSION,
        "method": result.config.method,
        "preset": result.config.preset,
        "config": asdict(result.config),
        "sample_rate": result.sample_rate,
        "epoch_seconds": result.epoch_seconds,
        "score_sha256": result.score_hash,
        "score_fingerprint_schema": SCORE_FINGERPRINT_SCHEMA,
        "absolute_power_unit": result.config.absolute_power_unit,
        "psd_unit": "uV^2/Hz",
        "relative_band_power_unit": "%",
        "relative_spectrum_unit": (
            result.config.relative_spectrum_unit
        ),
        "spectral_estimator": {
            "one_sided": True,
            "detrend": "none",
            "frequency_resolution_hz": (
                result.sample_rate
                / int(
                    round(
                        result.sample_rate * result.epoch_seconds
                    )
                )
            ),
            "frequency_boundary_policy": (
                "sampled_frequency_bins_inclusive"
            ),
            "taper_family": (
                "symmetric_hann"
                if result.config.is_legacy
                else "orthogonal_sine"
            ),
            "taper_count": (
                1
                if result.config.is_legacy
                else result.config.taper_count
            ),
            "overlap_samples": 0,
            "nfft_samples": int(
                round(result.sample_rate * result.epoch_seconds)
            ),
            "time_bandwidth_note": (
                "Retained for AutoFQ configuration compatibility; "
                "orthogonal sine tapers do not use a DPSS "
                "time-bandwidth product."
            ),
        },
        "source": result.metadata,
        "channels": [
            {
                "index_0based": channel.channel_index,
                "label": channel.channel_label,
                "physical_unit_after_conversion": channel.physical_unit,
                "included_epochs": int(np.sum(channel.valid_mask)),
                "excluded_epochs": int(np.sum(~channel.valid_mask)),
            }
            for channel in result.channel_results
        ],
    }
    payloads = {
        "spectrum_summary": _tsv_payload(
            result.spectrum_summary_frame()
        ),
        "band_summary": _tsv_payload(result.band_summary_frame()),
        "band_timecourse": _tsv_payload(result.timecourse_frame()),
        "epoch_band_power": _tsv_payload(
            result.epoch_band_power_frame()
        ),
        "epoch_qc": _tsv_payload(result.epoch_qc_frame()),
        "metadata": json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    }
    if result.config.export_epoch_spectra:
        payloads["epoch_spectra"] = _tsv_payload(
            result.epoch_spectra_frame()
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{safe_prefix}.",
            dir=root,
        )
    )
    staged: dict[str, Path] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    retain_staging = False
    try:
        for key, payload in payloads.items():
            staged[key] = staging / paths[key].name
            _atomic_text(staged[key], payload)
        for key, destination in paths.items():
            if destination.exists():
                backup = staging / f"{destination.name}.backup"
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(staged[key], destination)
            published.append(destination)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{destination.name}: {rollback_exc}"
                )
        for destination, backup in backups.items():
            if not backup.exists():
                continue
            try:
                os.replace(backup, destination)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"{destination.name}: {rollback_exc}"
                )
        if rollback_errors:
            retain_staging = True
            raise RuntimeError(
                "Spectral export failed and rollback was incomplete. "
                f"Recovery files remain in {staging}: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    finally:
        if not retain_staging:
            shutil.rmtree(staging, ignore_errors=True)
    return paths
