from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1


class ArtifactAuditStatus(str, Enum):
    VERIFIED = "verified"
    MISSING = "missing"
    UNREVIEWED = "unreviewed"
    STALE = "stale"
    CHANNEL_MISMATCH = "channel_mismatch"
    INVALID = "invalid"
    SOURCE_MISSING = "source_missing"


class ArtifactDecision(str, Enum):
    VERIFIED = "verified"
    EXPLORATORY = "exploratory"
    OPEN_TOOL = "open_tool"
    CANCEL = "cancel"


def artifact_sidecar_path(edf_path: Path) -> Path:
    return Path(edf_path).with_suffix(".artifacts.json")


def artifact_table_path(edf_path: Path) -> Path:
    return Path(edf_path).with_suffix(".artifacts.csv")


@lru_cache(maxsize=32)
def _sha256_for_stat(
    resolved_path: str,
    size_bytes: int,
    mtime_ns: int,
) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _sha256_for_stat(
        str(source),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def mask_sha256(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=np.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(values.tobytes())
    digest.update(f"|epochs={len(values)}".encode("ascii"))
    return digest.hexdigest()


def mask_to_ranges(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    return [
        [int(part[0]), int(part[-1])]
        for part in np.split(indices, split_points)
    ]


def ranges_to_mask(
    ranges: Sequence[Sequence[int]],
    epoch_count: int,
) -> np.ndarray:
    mask = np.zeros(int(epoch_count), dtype=bool)
    for value in ranges:
        if len(value) != 2:
            raise ValueError("Artifact range must contain start and end")
        start, end = int(value[0]), int(value[1])
        if start < 0 or end < start or end >= int(epoch_count):
            raise ValueError(
                f"Artifact range is outside 0..{int(epoch_count) - 1}: "
                f"{start}-{end}"
            )
        mask[start : end + 1] = True
    return mask


def _record_key(eeg_channel_index: int, emg_channel_index: int) -> str:
    return f"eeg{int(eeg_channel_index)}_emg{int(emg_channel_index)}"


@dataclass
class ArtifactRecord:
    source_edf: str
    source_fingerprint: dict[str, Any]
    epoch_count: int
    epoch_seconds: float
    sample_rate: float
    eeg_channel_index: int
    emg_channel_index: int
    eeg_channel_label: str
    emg_channel_label: str
    thresholds_log: dict[str, float]
    use_auto_candidates: bool
    auto_candidate_mask: np.ndarray
    manual_add_mask: np.ndarray
    manual_remove_mask: np.ndarray
    final_artifact_mask: np.ndarray
    reason_masks: dict[str, np.ndarray] = field(default_factory=dict)
    review_status: str = "confirmed"
    detector_version: str = ""
    created_at: str = ""
    imported_from: str = ""

    @property
    def key(self) -> str:
        return _record_key(
            self.eeg_channel_index,
            self.emg_channel_index,
        )

    @property
    def reviewed(self) -> bool:
        return self.review_status == "confirmed"

    def validate(self) -> None:
        if int(self.epoch_count) <= 0:
            raise ValueError("Artifact epoch count must be positive")
        if float(self.epoch_seconds) <= 0:
            raise ValueError("Artifact epoch duration must be positive")
        if float(self.sample_rate) <= 0:
            raise ValueError("Artifact sample rate must be positive")
        for name, mask in (
            ("auto_candidate_mask", self.auto_candidate_mask),
            ("manual_add_mask", self.manual_add_mask),
            ("manual_remove_mask", self.manual_remove_mask),
            ("final_artifact_mask", self.final_artifact_mask),
            *self.reason_masks.items(),
        ):
            if len(np.asarray(mask).reshape(-1)) != int(self.epoch_count):
                raise ValueError(
                    f"{name} length does not match artifact epoch count"
                )

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "eeg_channel_index_0based": int(self.eeg_channel_index),
            "emg_channel_index_0based": int(self.emg_channel_index),
            "eeg_channel_label": str(self.eeg_channel_label),
            "emg_channel_label": str(self.emg_channel_label),
            "thresholds_log": {
                str(key): float(value)
                for key, value in self.thresholds_log.items()
            },
            "use_auto_candidates": bool(self.use_auto_candidates),
            "auto_candidate_ranges": mask_to_ranges(
                self.auto_candidate_mask
            ),
            "manual_add_ranges": mask_to_ranges(self.manual_add_mask),
            "manual_remove_ranges": mask_to_ranges(
                self.manual_remove_mask
            ),
            "final_artifact_ranges": mask_to_ranges(
                self.final_artifact_mask
            ),
            "reason_ranges": {
                str(name): mask_to_ranges(mask)
                for name, mask in self.reason_masks.items()
            },
            "mask_sha256": mask_sha256(self.final_artifact_mask),
            "counts": {
                "auto_candidates": int(
                    np.sum(self.auto_candidate_mask)
                ),
                "manual_added": int(np.sum(self.manual_add_mask)),
                "manual_removed": int(np.sum(self.manual_remove_mask)),
                "final": int(np.sum(self.final_artifact_mask)),
            },
            "review_status": str(self.review_status),
            "detector_version": str(self.detector_version),
            "created_at": str(self.created_at),
            "imported_from": str(self.imported_from),
        }

    @classmethod
    def from_payload(
        cls,
        root: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> "ArtifactRecord":
        epoch_count = int(root["epoch_count"])
        reason_payload = payload.get("reason_ranges", {})
        if not isinstance(reason_payload, Mapping):
            raise ValueError("Artifact reason_ranges must be an object")
        record = cls(
            source_edf=str(root.get("source_edf", "")),
            source_fingerprint=dict(root.get("source_fingerprint", {})),
            epoch_count=epoch_count,
            epoch_seconds=float(root["epoch_seconds"]),
            sample_rate=float(root["sample_rate"]),
            eeg_channel_index=int(
                payload["eeg_channel_index_0based"]
            ),
            emg_channel_index=int(
                payload["emg_channel_index_0based"]
            ),
            eeg_channel_label=str(
                payload.get("eeg_channel_label", "")
            ),
            emg_channel_label=str(
                payload.get("emg_channel_label", "")
            ),
            thresholds_log={
                str(key): float(value)
                for key, value in dict(
                    payload.get("thresholds_log", {})
                ).items()
            },
            use_auto_candidates=bool(
                payload.get("use_auto_candidates", True)
            ),
            auto_candidate_mask=ranges_to_mask(
                payload.get("auto_candidate_ranges", []),
                epoch_count,
            ),
            manual_add_mask=ranges_to_mask(
                payload.get("manual_add_ranges", []),
                epoch_count,
            ),
            manual_remove_mask=ranges_to_mask(
                payload.get("manual_remove_ranges", []),
                epoch_count,
            ),
            final_artifact_mask=ranges_to_mask(
                payload.get("final_artifact_ranges", []),
                epoch_count,
            ),
            reason_masks={
                str(name): ranges_to_mask(ranges, epoch_count)
                for name, ranges in reason_payload.items()
            },
            review_status=str(
                payload.get("review_status", "unreviewed")
            ),
            detector_version=str(
                payload.get("detector_version", "")
            ),
            created_at=str(payload.get("created_at", "")),
            imported_from=str(payload.get("imported_from", "")),
        )
        record.validate()
        expected_mask_hash = str(payload.get("mask_sha256", ""))
        if (
            expected_mask_hash
            and expected_mask_hash
            != mask_sha256(record.final_artifact_mask)
        ):
            raise ValueError("Artifact mask SHA-256 does not match")
        return record


@dataclass(frozen=True)
class ArtifactAudit:
    status: ArtifactAuditStatus
    severity: str
    title: str
    message: str
    source_edf: str
    sidecar_path: str
    fingerprint: str
    eeg_channel_index: int
    emg_channel_index: int
    record: ArtifactRecord | None = None
    reasons: tuple[str, ...] = ()
    checksum_verified: bool = False

    @property
    def can_use_mask(self) -> bool:
        return (
            self.record is not None
            and self.status
            in {
                ArtifactAuditStatus.VERIFIED,
                ArtifactAuditStatus.UNREVIEWED,
            }
        )

    @property
    def mask(self) -> np.ndarray | None:
        if not self.can_use_mask or self.record is None:
            return None
        return self.record.final_artifact_mask.copy()

    def metadata(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "source_edf": self.source_edf,
            "sidecar_path": self.sidecar_path,
            "fingerprint": self.fingerprint,
            "eeg_channel_index_0based": self.eeg_channel_index,
            "emg_channel_index_0based": self.emg_channel_index,
            "reasons": list(self.reasons),
            "checksum_verified": self.checksum_verified,
            "mask_sha256": (
                mask_sha256(self.record.final_artifact_mask)
                if self.record is not None
                else ""
            ),
            "artifact_epochs": (
                int(np.sum(self.record.final_artifact_mask))
                if self.record is not None
                else 0
            ),
            "review_status": (
                self.record.review_status
                if self.record is not None
                else ""
            ),
        }


def _quick_fingerprint(source: Path, sidecar: Path) -> str:
    parts = [str(source.resolve())]
    for path in (source, sidecar):
        try:
            stat = path.stat()
        except OSError:
            parts.extend(["missing", "missing"])
        else:
            parts.extend([str(stat.st_size), str(stat.st_mtime_ns)])
    return "|".join(parts)


def source_fingerprint(
    source_edf: Path,
    *,
    sha256_value: str | None = None,
) -> dict[str, Any]:
    source = Path(source_edf).resolve()
    stat = source.stat()
    return {
        "resolved_path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": str(sha256_value or _sha256(source)),
    }


def _load_payload(source_edf: Path) -> dict[str, Any]:
    sidecar = artifact_sidecar_path(source_edf)
    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Artifact sidecar root must be an object")
    if int(payload.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported Artifact sidecar schema")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError("Artifact sidecar records must be an object")
    return payload


def load_all_artifact_records(
    source_edf: Path,
) -> dict[str, ArtifactRecord]:
    payload = _load_payload(source_edf)
    records: dict[str, ArtifactRecord] = {}
    for key, value in dict(payload["records"]).items():
        if not isinstance(value, Mapping):
            raise ValueError(f"Artifact record {key} is invalid")
        records[str(key)] = ArtifactRecord.from_payload(payload, value)
    return records


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _table_payload(
    records: Mapping[str, ArtifactRecord],
) -> str:
    frames: list[pd.DataFrame] = []
    for record in records.values():
        count = int(record.epoch_count)
        frame = pd.DataFrame(
            {
                "eeg_channel_index_0based": np.full(
                    count, record.eeg_channel_index, dtype=int
                ),
                "emg_channel_index_0based": np.full(
                    count, record.emg_channel_index, dtype=int
                ),
                "epoch_0based": np.arange(count, dtype=int),
                "start_seconds": (
                    np.arange(count, dtype=float)
                    * float(record.epoch_seconds)
                ),
                "auto_candidate": np.asarray(
                    record.auto_candidate_mask, dtype=bool
                ),
                "manual_added": np.asarray(
                    record.manual_add_mask, dtype=bool
                ),
                "manual_removed": np.asarray(
                    record.manual_remove_mask, dtype=bool
                ),
                "final_artifact": np.asarray(
                    record.final_artifact_mask, dtype=bool
                ),
                "review_status": np.full(
                    count, record.review_status, dtype=object
                ),
            }
        )
        for name, mask in sorted(record.reason_masks.items()):
            frame[f"reason_{name}"] = np.asarray(mask, dtype=bool)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return output.to_csv(index=False, lineterminator="\n")


def save_artifact_record(record: ArtifactRecord) -> dict[str, Path]:
    record.validate()
    source = Path(record.source_edf).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Artifact source EDF not found: {source}")
    current_fingerprint = source_fingerprint(source)
    record.source_edf = str(source)
    record.source_fingerprint = current_fingerprint
    if not record.created_at:
        record.created_at = datetime.now().astimezone().isoformat()

    sidecar = artifact_sidecar_path(source)
    table = artifact_table_path(source)
    records: dict[str, ArtifactRecord] = {}
    if sidecar.is_file():
        try:
            existing_payload = _load_payload(source)
            existing_fingerprint = existing_payload.get(
                "source_fingerprint",
                {},
            )
            if not isinstance(existing_fingerprint, Mapping):
                raise ValueError("Artifact source fingerprint is invalid")
            for field in (
                "resolved_path",
                "size_bytes",
                "mtime_ns",
                "sha256",
            ):
                if str(existing_fingerprint.get(field, "")) != str(
                    current_fingerprint[field]
                ):
                    raise ValueError(
                        "Artifact source fingerprint changed"
                    )
            if (
                int(existing_payload.get("epoch_count", -1))
                != int(record.epoch_count)
                or not np.isclose(
                    float(existing_payload.get("epoch_seconds", -1)),
                    float(record.epoch_seconds),
                )
                or not np.isclose(
                    float(existing_payload.get("sample_rate", -1)),
                    float(record.sample_rate),
                )
            ):
                raise ValueError("Artifact epoch geometry changed")
            records = {
                str(key): ArtifactRecord.from_payload(
                    existing_payload,
                    value,
                )
                for key, value in dict(
                    existing_payload["records"]
                ).items()
            }
        except Exception:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            shutil.copy2(
                sidecar,
                sidecar.with_name(
                    f"{sidecar.stem}.backup_{timestamp}{sidecar.suffix}"
                ),
            )
            if table.is_file():
                shutil.copy2(
                    table,
                    table.with_name(
                        f"{table.stem}.backup_{timestamp}{table.suffix}"
                    ),
                )
            records = {}
    records[record.key] = record
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_edf": str(source),
        "source_fingerprint": current_fingerprint,
        "epoch_count": int(record.epoch_count),
        "epoch_seconds": float(record.epoch_seconds),
        "sample_rate": float(record.sample_rate),
        "updated_at": datetime.now().astimezone().isoformat(),
        "records": {
            key: value.payload() for key, value in sorted(records.items())
        },
    }
    _atomic_write_text(
        sidecar,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _atomic_write_text(
        table,
        _table_payload(records),
        encoding="utf-8-sig",
    )
    return {"json": sidecar, "csv": table}


def audit_artifact_record(
    source_edf: Path,
    *,
    epoch_count: int,
    epoch_seconds: float,
    sample_rate: float,
    eeg_channel_index: int,
    emg_channel_index: int,
    channel_labels: Sequence[str] | None = None,
    verify_checksum: bool = False,
) -> ArtifactAudit:
    source = Path(source_edf).resolve()
    sidecar = artifact_sidecar_path(source)
    quick = _quick_fingerprint(source, sidecar)
    common = {
        "source_edf": str(source),
        "sidecar_path": str(sidecar),
        "fingerprint": quick,
        "eeg_channel_index": int(eeg_channel_index),
        "emg_channel_index": int(emg_channel_index),
    }
    if not source.is_file():
        return ArtifactAudit(
            status=ArtifactAuditStatus.SOURCE_MISSING,
            severity="critical",
            title="Artifact source EDF is missing",
            message="The selected EDF no longer exists.",
            reasons=("source_missing",),
            **common,
        )
    if not sidecar.is_file():
        return ArtifactAudit(
            status=ArtifactAuditStatus.MISSING,
            severity="warning",
            title="Artifact QC has not been completed",
            message=(
                "No reusable Artifact review exists for this EDF and "
                "channel pair."
            ),
            reasons=("sidecar_missing",),
            **common,
        )
    try:
        payload = _load_payload(source)
    except Exception as exc:
        return ArtifactAudit(
            status=ArtifactAuditStatus.INVALID,
            severity="critical",
            title="Artifact QC record is invalid",
            message=f"The Artifact sidecar cannot be read: {exc}",
            reasons=(f"sidecar_invalid:{type(exc).__name__}",),
            **common,
        )

    reasons: list[str] = []
    stored_fingerprint = payload.get("source_fingerprint")
    if not isinstance(stored_fingerprint, Mapping):
        reasons.append("source_fingerprint_missing")
        stored_fingerprint = {}
    stat = source.stat()
    stored_path = str(stored_fingerprint.get("resolved_path", ""))
    if stored_path and Path(stored_path) != source:
        reasons.append("source_path_mismatch")
    if int(stored_fingerprint.get("size_bytes", -1)) != int(
        stat.st_size
    ):
        reasons.append("source_size_mismatch")
    if int(stored_fingerprint.get("mtime_ns", -1)) != int(
        stat.st_mtime_ns
    ):
        reasons.append("source_mtime_mismatch")
    checksum_verified = False
    if verify_checksum:
        expected_hash = str(stored_fingerprint.get("sha256", ""))
        if len(expected_hash) != 64:
            reasons.append("source_sha256_missing")
        else:
            current_hash = _sha256(source)
            checksum_verified = True
            if current_hash.lower() != expected_hash.lower():
                reasons.append("source_sha256_mismatch")

    if reasons:
        return ArtifactAudit(
            status=ArtifactAuditStatus.STALE,
            severity="critical",
            title="Artifact QC does not match the current EDF",
            message=(
                "The saved Artifact mask is stale and will not be used. "
                "Run Artifact Detection & Review again."
            ),
            reasons=tuple(reasons),
            checksum_verified=checksum_verified,
            **common,
        )

    if int(payload.get("epoch_count", -1)) != int(epoch_count):
        reasons.append("epoch_count_mismatch")
    if not np.isclose(
        float(payload.get("epoch_seconds", -1)),
        float(epoch_seconds),
    ):
        reasons.append("epoch_seconds_mismatch")
    if not np.isclose(
        float(payload.get("sample_rate", -1)),
        float(sample_rate),
    ):
        reasons.append("sample_rate_mismatch")
    if reasons:
        return ArtifactAudit(
            status=ArtifactAuditStatus.STALE,
            severity="critical",
            title="Artifact QC geometry does not match",
            message=(
                "The saved Artifact epoch geometry does not match the "
                "current EDF/scoring session."
            ),
            reasons=tuple(reasons),
            checksum_verified=checksum_verified,
            **common,
        )

    key = _record_key(eeg_channel_index, emg_channel_index)
    records = payload.get("records", {})
    if key not in records:
        return ArtifactAudit(
            status=ArtifactAuditStatus.CHANNEL_MISMATCH,
            severity="warning",
            title="Artifact QC channel mismatch",
            message=(
                "Artifact QC exists for another EEG/EMG channel pair. "
                "Review the currently selected channels before analysis."
            ),
            reasons=("channel_pair_missing",),
            checksum_verified=checksum_verified,
            **common,
        )
    try:
        record = ArtifactRecord.from_payload(payload, records[key])
    except Exception as exc:
        return ArtifactAudit(
            status=ArtifactAuditStatus.INVALID,
            severity="critical",
            title="Artifact QC record is invalid",
            message=f"The saved Artifact mask is invalid: {exc}",
            reasons=(f"record_invalid:{type(exc).__name__}",),
            checksum_verified=checksum_verified,
            **common,
        )

    if channel_labels is not None:
        if (
            0 <= int(eeg_channel_index) < len(channel_labels)
            and str(channel_labels[int(eeg_channel_index)])
            != record.eeg_channel_label
        ):
            reasons.append("eeg_channel_label_mismatch")
        if (
            0 <= int(emg_channel_index) < len(channel_labels)
            and str(channel_labels[int(emg_channel_index)])
            != record.emg_channel_label
        ):
            reasons.append("emg_channel_label_mismatch")
    if reasons:
        return ArtifactAudit(
            status=ArtifactAuditStatus.CHANNEL_MISMATCH,
            severity="warning",
            title="Artifact QC channel labels changed",
            message=(
                "The saved Artifact channel labels do not match the "
                "currently loaded EDF."
            ),
            record=record,
            reasons=tuple(reasons),
            checksum_verified=checksum_verified,
            **common,
        )
    if not record.reviewed:
        return ArtifactAudit(
            status=ArtifactAuditStatus.UNREVIEWED,
            severity="warning",
            title="Artifact candidates have not been confirmed",
            message=(
                "An Artifact mask exists, but it has not been marked as "
                "human reviewed."
            ),
            record=record,
            reasons=("review_not_confirmed",),
            checksum_verified=checksum_verified,
            **common,
        )
    return ArtifactAudit(
        status=ArtifactAuditStatus.VERIFIED,
        severity="verified",
        title="Artifact QC verified",
        message=(
            f"Artifact QC verified: "
            f"{int(np.sum(record.final_artifact_mask))}/"
            f"{record.epoch_count} epochs excluded."
        ),
        record=record,
        checksum_verified=checksum_verified,
        **common,
    )


def load_verified_artifact_masks(
    source_edf: Path,
    *,
    epoch_count: int,
    epoch_seconds: float,
    sample_rate: float,
    emg_channel_index: int,
    eeg_channel_indices: Sequence[int],
    channel_labels: Sequence[str] | None = None,
    verify_checksum: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, ArtifactAudit]]:
    masks: dict[int, np.ndarray] = {}
    audits: dict[int, ArtifactAudit] = {}
    for eeg_index in eeg_channel_indices:
        audit = audit_artifact_record(
            source_edf,
            epoch_count=epoch_count,
            epoch_seconds=epoch_seconds,
            sample_rate=sample_rate,
            eeg_channel_index=int(eeg_index),
            emg_channel_index=int(emg_channel_index),
            channel_labels=channel_labels,
            verify_checksum=verify_checksum,
        )
        audits[int(eeg_index)] = audit
        if audit.status == ArtifactAuditStatus.VERIFIED:
            mask = audit.mask
            if mask is not None:
                masks[int(eeg_index)] = mask
    return masks, audits


def effective_artifact_mask(
    scores: np.ndarray | Sequence[int],
    artifact_masks: Sequence[np.ndarray] = (),
) -> np.ndarray:
    values = np.asarray(scores, dtype=int).reshape(-1)
    output = values == 0
    for mask in artifact_masks:
        candidate = np.asarray(mask, dtype=bool).reshape(-1)
        if len(candidate) != len(output):
            raise ValueError("Artifact mask length does not match scores")
        output |= candidate
    return output
