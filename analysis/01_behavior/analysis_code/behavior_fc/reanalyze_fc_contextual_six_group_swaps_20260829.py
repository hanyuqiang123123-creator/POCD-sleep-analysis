from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(SCRIPT_DIR))
import reanalyze_fc_contextual_20260317_20260829 as analysis  # noqa: E402


FIGURE_ID = "fc_contextual_20260317_six_group_swaps_20260829"
OUTDIR = Path(r"F:\Sleep\Figure\FC_contextual_4groups_20260317_six_group_swaps_20260829")
PROVENANCE_DIR = Path(r"F:\Sleep\outputs\figure_provenance")
DATA_DIR = Path(r"F:\1.Sleep\eXdata\BehaviorTest\hm4Di\FC\20260317-TEST\CONTEXTUAL")
CORRECTIONS = [
    {
        "pair": "S8/S23",
        "description": "S8: POCD+SAL → POCD+CNO; S23: POCD+CNO → POCD+SAL",
        "audit": DATA_DIR / "packwin_group_swap_S8_S23_audit_20260829.json",
        "script": SCRIPT_DIR / "complete_packwin_group_swap_s8_s23_20260317.py",
        "backup": DATA_DIR / "_backup_before_group_swap_S8_S23_20260829",
    },
    {
        "pair": "S9/S26",
        "description": "S9: POCD+SAL → POCD+CNO; S26: POCD+CNO → POCD+SAL",
        "audit": DATA_DIR / "packwin_group_swap_S9_S26_audit_20260829.json",
        "script": SCRIPT_DIR / "complete_packwin_group_swap_s9_s26_20260317.py",
        "backup": DATA_DIR / "_backup_before_group_swap_S9_S26_20260829",
    },
    {
        "pair": "S10/S20",
        "description": "S10: POCD+SAL → baseline+CNO; S20: baseline+CNO → POCD+SAL",
        "audit": DATA_DIR / "packwin_group_swap_S10_S20_audit_20260829.json",
        "script": SCRIPT_DIR / "complete_packwin_group_swap_s10_s20_20260317.py",
        "backup": DATA_DIR / "_backup_before_group_swap_S10_S20_20260829",
    },
]


def upsert_manifest(manifest: pd.DataFrame, path: Path, note: str) -> pd.DataFrame:
    manifest = manifest.loc[manifest["path"] != str(path)]
    row = {"path": str(path), "format_notes": note, "sha256": analysis.sha256_file(path)}
    return pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)


def augment_records() -> None:
    source_path = OUTDIR / "source_paths.csv"
    sources = pd.read_csv(source_path)
    additions: list[dict[str, str]] = []
    for item in CORRECTIONS:
        candidates = [
            (item["audit"], "group-assignment-audit", f"Audited {item['pair']} randomization group correction"),
            (item["script"], "script", f"PACKWIN {item['pair']} group-correction script"),
            (item["backup"] / "CONTEXTUAL.exp", "raw-backup", f"EXP backup before {item['pair']} correction"),
            (item["backup"] / "CONTEXTUAL.ssn", "raw-backup", f"SSN backup before {item['pair']} correction"),
        ]
        for path, role, description in candidates:
            if path.exists():
                additions.append(
                    {
                        "role": role,
                        "path": str(path),
                        "description": description,
                        "sha256": analysis.sha256_file(path),
                    }
                )
    new_paths = {row["path"] for row in additions}
    sources = sources.loc[~sources["path"].isin(new_paths)]
    pd.concat([sources, pd.DataFrame(additions)], ignore_index=True).to_csv(source_path, index=False)

    correction_lines = "\n".join(f"- {item['description']}." for item in CORRECTIONS)
    readme_path = OUTDIR / "README_analysis.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme += f"""

## User-confirmed randomization group corrections

{correction_lines}
- Every animal retained its own raw session, timestamp, animal code, and freezing measurement.
- PACKWIN `.exp` subject slots and `.ssn` session references were synchronized at each step.
- `.ini` and `.raw` remained unchanged; all 26 measurement records were numerically preserved.
- Separate audits and exact stepwise backups are listed in `source_paths.csv`.
- Generated revision: {datetime.now().astimezone().isoformat(timespec='seconds')}.
"""
    readme_path.write_text(readme, encoding="utf-8")

    provenance_path = PROVENANCE_DIR / f"{FIGURE_ID}_provenance.md"
    provenance = provenance_path.read_text(encoding="utf-8")
    provenance += f"""

## User-confirmed randomization corrections

{correction_lines}
- Raw sessions and measurements remain paired with the same animal identities.
- Each correction has an exact pre-step EXP/SSN backup and JSON audit in the figure source table.
"""
    provenance_path.write_text(provenance, encoding="utf-8")

    manifest_path = OUTDIR / "outputs_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    for item in CORRECTIONS:
        for path, note in [
            (item["audit"], f"JSON {item['pair']} group-correction audit"),
            (item["script"], f"Reproducible {item['pair']} correction script"),
            (item["backup"] / "CONTEXTUAL.exp", f"EXP backup before {item['pair']} correction"),
            (item["backup"] / "CONTEXTUAL.ssn", f"SSN backup before {item['pair']} correction"),
        ]:
            if path.exists():
                manifest = upsert_manifest(manifest, path, note)
    for path, note in [
        (source_path, "Updated source and audit path table"),
        (readme_path, "Updated analysis record"),
        (provenance_path, "Updated figure provenance record"),
    ]:
        manifest = upsert_manifest(manifest, path, note)
    manifest.to_csv(manifest_path, index=False)


def main() -> int:
    analysis.FIGURE_ID = FIGURE_ID
    analysis.OUTDIR = OUTDIR
    analysis.PROVENANCE_DIR = PROVENANCE_DIR
    analysis.SCRIPT_PATH = SCRIPT_PATH
    status = analysis.main()
    augment_records()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
