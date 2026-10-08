from __future__ import annotations

import csv
import hashlib
import math
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"F:\Sleep\PHD稿件\Figures\Fig5\Anal\Gcamp_FQ")
OUT = ROOT / "state_level_reanalysis"
DATA_OUT = OUT / "data"
DOC_OUT = OUT / "documentation"

STATE_MAP = {1: "Wake", 2: "NREM", 3: "REM", 4: "MA"}
STATE_ORDER = ["Wake", "NREM", "REM", "MA"]

# Project1.pzfx has six rows but no recording identifiers. This order is inferred
# from exact matching to five rows plus the distinctive fifth-row discrepancy.
PRISM6_INFERRED_ORDER = [
    "20250602Trail3",
    "20260313",
    "20250605Trail1",
    "202506062",
    "202506125",
    "gcamp20250511",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted.tolist()


def rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff = diff[diff != 0]
    ranks = stats.rankdata(np.abs(diff))
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    return (positive - negative) / (positive + negative)


def read_result_csv(path: Path) -> pd.DataFrame:
    # The numeric fields are ASCII. latin-1 safely preserves the one historical
    # CSV whose Chinese stage labels contain mixed/corrupted bytes.
    return pd.read_csv(path, encoding="latin1")


def collect_records() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    means: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    sources = sorted(ROOT.rglob("*_results.csv"))
    for path in sources:
        table = read_result_csv(path)
        record = path.parent.name
        mean_row: dict[str, object] = {"record_id": record}
        for _, row in table.iterrows():
            state = STATE_MAP[int(row["stage"])]
            mean_row[state] = float(row["mean_intensity"])
            details.append(
                {
                    "record_id": record,
                    "state": state,
                    "mean_intensity": float(row["mean_intensity"]),
                    "within_record_epoch_sd": float(row["std_intensity"]),
                    "epoch_count": int(row["count"]),
                    "duration_seconds": int(row["count"]) * 5,
                    "source_csv": str(path),
                }
            )
        means.append(mean_row)
    wide = pd.DataFrame(means).set_index("record_id")[STATE_ORDER].sort_index()
    long = pd.DataFrame(details).sort_values(["record_id", "state"])
    if wide.isna().any().any() or len(wide) != 8:
        raise ValueError("Expected eight complete recording-level rows across four states.")
    return wide, long, sources


def summary_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        x = wide[state].to_numpy(float)
        rows.append(
            {
                "state": state,
                "record_n": len(x),
                "mean": np.mean(x),
                "sd": np.std(x, ddof=1),
                "sem": stats.sem(x),
                "median": np.median(x),
                "q1": np.quantile(x, 0.25),
                "q3": np.quantile(x, 0.75),
                "min": np.min(x),
                "max": np.max(x),
            }
        )
    return pd.DataFrame(rows)


def repeated_measures_anova(wide: pd.DataFrame) -> tuple[float, int, int, float]:
    x = wide[STATE_ORDER].to_numpy(float)
    n, k = x.shape
    grand = x.mean()
    ss_state = n * np.square(x.mean(axis=0) - grand).sum()
    ss_subject = k * np.square(x.mean(axis=1) - grand).sum()
    ss_total = np.square(x - grand).sum()
    ss_error = ss_total - ss_state - ss_subject
    df_state = k - 1
    df_error = (n - 1) * (k - 1)
    f_value = (ss_state / df_state) / (ss_error / df_error)
    return f_value, df_state, df_error, stats.f.sf(f_value, df_state, df_error)


def statistical_tables(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    friedman = stats.friedmanchisquare(*[wide[s] for s in STATE_ORDER])
    kendall_w = float(friedman.statistic) / (len(wide) * (len(STATE_ORDER) - 1))
    f_value, df1, df2, rm_p = repeated_measures_anova(wide)
    omnibus = pd.DataFrame(
        [
            {
                "analysis": "primary",
                "test": "Friedman repeated-measures test",
                "statistic": friedman.statistic,
                "df1": 3,
                "df2": "",
                "p_value": friedman.pvalue,
                "effect_name": "Kendall_W",
                "effect_value": kendall_w,
                "record_n": len(wide),
            },
            {
                "analysis": "parametric_sensitivity",
                "test": "one-way repeated-measures ANOVA",
                "statistic": f_value,
                "df1": df1,
                "df2": df2,
                "p_value": rm_p,
                "effect_name": "",
                "effect_value": "",
                "record_n": len(wide),
            },
        ]
    )

    pair_rows: list[dict[str, object]] = []
    for a, b in combinations(STATE_ORDER, 2):
        xa = wide[a].to_numpy(float)
        xb = wide[b].to_numpy(float)
        wilcoxon = stats.wilcoxon(xa, xb, method="exact")
        paired_t = stats.ttest_rel(xa, xb)
        pair_rows.append(
            {
                "state_1": a,
                "state_2": b,
                "mean_difference_state1_minus_state2": np.mean(xa - xb),
                "median_difference_state1_minus_state2": np.median(xa - xb),
                "wilcoxon_W": wilcoxon.statistic,
                "wilcoxon_p_raw": wilcoxon.pvalue,
                "paired_rank_biserial": rank_biserial_paired(xa, xb),
                "paired_t": paired_t.statistic,
                "paired_t_p_raw": paired_t.pvalue,
                "difference_shapiro_p": stats.shapiro(xa - xb).pvalue,
                "record_n": len(wide),
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["wilcoxon_p_holm_6"] = holm_adjust(pairwise["wilcoxon_p_raw"].tolist())
    pairwise["paired_t_p_holm_6"] = holm_adjust(pairwise["paired_t_p_raw"].tolist())
    return omnibus, pairwise


def parse_prism_data2(path: Path) -> pd.DataFrame:
    root = ET.parse(path).getroot()
    ns = {"p": "http://graphpad.com/prism/Prism.htm"}
    table = root.find("p:Table[@ID='Table4']", ns)
    if table is None:
        raise ValueError("Project1.pzfx Table4 not found")
    columns: dict[str, list[float]] = {}
    for col in table.findall("p:YColumn", ns):
        title_node = col.find("p:Title", ns)
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        values = [float(x.text) for x in col.findall("p:Subcolumn/p:d", ns)]
        columns[title] = values
    prism = pd.DataFrame(columns)
    prism.insert(0, "inferred_record_id", PRISM6_INFERRED_ORDER)
    return prism


def write_report(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    summary: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    prism_audit: pd.DataFrame,
) -> None:
    s = summary.set_index("state")
    frow = omnibus.iloc[0]
    rmrow = omnibus.iloc[1]

    def pair(a: str, b: str) -> pd.Series:
        return pairwise[(pairwise.state_1 == a) & (pairwise.state_2 == b)].iloc[0]

    ma_nrem = pair("NREM", "MA")
    wake_nrem = pair("Wake", "NREM")
    nrem_rem = pair("NREM", "REM")
    min_counts = long.groupby("state")["epoch_count"].min().to_dict()

    report = f"""# GCaMP 睡眠状态钙信号强度复核

## 结论

按文件夹中的全部 8 条记录作为配对重复测量单位，处理后钙信号在 NREM 期最低：Wake {s.loc['Wake','mean']:.3f} ± {s.loc['Wake','sem']:.3f}、NREM {s.loc['NREM','mean']:.3f} ± {s.loc['NREM','sem']:.3f}、REM {s.loc['REM','mean']:.3f} ± {s.loc['REM','sem']:.3f}、MA {s.loc['MA','mean']:.3f} ± {s.loc['MA','sem']:.3f}（均为均值 ± SEM，n = 8 条记录）。现有 PDF 将纵轴标为 ΔF/F (%)，但当前文件夹没有从原始荧光 F、F0 到 ΔF/F 的计算代码，因此正式写作前应确认单位。

四状态总体差异显著（Friedman χ²(3) = {frow.statistic:.3f}, P = {frow.p_value:.6f}, Kendall's W = {frow.effect_value:.3f}）。Holm 校正后的精确配对 Wilcoxon 检验显示：MA 高于 NREM（P = {ma_nrem.wilcoxon_p_holm_6:.6f}），Wake 高于 NREM（P = {wake_nrem.wilcoxon_p_holm_6:.6f}）；REM 与 NREM 为边界趋势（P = {nrem_rem.wilcoxon_p_holm_6:.6f}）。MA 与 Wake、MA 与 REM 均无显著差异。参数敏感性分析得到重复测量 ANOVA F({int(rmrow.df1)},{int(rmrow.df2)}) = {rmrow.statistic:.3f}, P = {rmrow.p_value:.3g}，方向一致。

因此，当前数据最稳妥的表述是：**钙信号在稳定 NREM 睡眠期间受到抑制，而在微觉醒期间升高到接近持续觉醒和 REM 睡眠的水平。** 这支持 MA 伴随神经元群体活动重新增强，但仅凭状态均值不能判断钙信号升高发生在 MA 前、MA 起始时还是 MA 后。

## 分析单位与方法

- 状态编码：1 = Wake，2 = NREM，3 = REM，4 = MA。
- 每个分期 epoch 为 5 秒；每条记录先计算各状态的 epoch 平均值，再以记录为配对单位比较四种状态。
- 主检验为 Friedman 重复测量检验；6 个事后两两比较使用精确配对 Wilcoxon 检验并作 Holm 校正。
- 重复测量 ANOVA 和配对 t 检验仅作为参数敏感性分析。
- `count` 是 5 秒 epoch 数，不是动物数；本次探索性分析的 n = 8 条记录。记录文件夹能否一一对应 8 只独立小鼠，仍需实验台账确认。

## 重要质控发现

1. 所有分析 `.mat` 的 `reshaped_data` 每个 5 秒 epoch 均为 200 点，对应本分析有效采样率 40 Hz。两个文件名中的 `200Hz` 更可能描述原始来源或命名历史，不能据此把当前分析率写成 200 Hz。
2. `Code/analyzeGcampFQ.m` 写明 40 Hz、5 秒 epoch，但其保存文件名和字段与现有输出不一致；它不是生成这些输出的精确代码版本。
3. 人工时间校准以完整 5 秒 epoch 为步长。校准图中存在 +5 秒、+25 秒、-60 秒以及输入 -77 秒但实际四舍五入为 -15 epoch（即 -75 秒）的记录；`20260313` 未保存校准图。正式分析需保留每条记录最终使用的精确 stage shift。
4. 个别状态样本很少：最少 REM 为 {min_counts.get('REM')} 个 epoch，最少 MA 为 {min_counts.get('MA')} 个 epoch。即某些状态均值只基于约 10–15 秒信号，方差较大，应做最小有效时长/epoch 数敏感性分析。
5. `Project1.pzfx` 和 `gcamp.pdf` 仅显示 6 条记录，未纳入 `20250603Trail5` 与 `GCaMP4`。此外，Prism 第 5 行推定对应 `202506125`，其 Wake、REM、MA 比原 CSV 分别低 3.0，而 NREM 未变。除非这是有记录的基线校正，否则不应使用该手工修改表作为最终源数据。
6. `20250603Trail5` 整体响应较低，但没有预先规定的排除理由。当前复核保留全部 8 条记录，不根据结果高低进行事后排除。

## 建议的下一步

- 从实验台账补齐：每条记录的 mouse ID、病毒/探针、组别、记录日期、原始 F 与参考通道、ΔF/F 或 z-score 算法、排除标准。
- 确认 8 条记录是否均为独立小鼠；若同一小鼠有多个 session，应使用小鼠为随机效应的混合模型。
- 对 MA 做事件触发分析：以 MA 起点为 0 秒，比较起始前 NREM 基线、MA 窗口及恢复期；先在每只小鼠内平均事件，再做动物级统计。
- 增加对最少状态时长的敏感性分析，例如每状态至少 3 或 5 个 epoch；阈值须在看结果之前固定。
- 在上述信息确认前，本报告应标记为 exploratory/candidate，不宜直接作为最终方法学或因果结论。

## 输出文件

- `data/Gcamp_state_record_means_all8.csv`：8 条记录的四状态均值。
- `data/Gcamp_state_epoch_details_all8.csv`：每状态均值、epoch 内 SD、epoch 数和时长。
- `data/Gcamp_state_summary_all8.csv`：动物/记录级描述统计。
- `data/Gcamp_state_omnibus_stats_all8.csv`：总体检验。
- `data/Gcamp_state_pairwise_stats_all8.csv`：两两检验及 Holm 校正。
- `data/Gcamp_state_stats_all8_long.csv`：用于溯源归档的总体与两两检验长表。
- `data/Gcamp_Prism6_value_audit.csv`：Prism 6 条记录与原 CSV 的逐值审计。
- `documentation/source_manifest.csv`：输入文件路径、大小、时间与 SHA-256。
"""
    (DOC_OUT / "README_analysis.md").write_text(report, encoding="utf-8")


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    DOC_OUT.mkdir(parents=True, exist_ok=True)

    wide, long, result_sources = collect_records()
    summary = summary_table(wide)
    omnibus, pairwise = statistical_tables(wide)

    prism = parse_prism_data2(ROOT / "Project1.pzfx")
    audit_rows = []
    for _, row in prism.iterrows():
        record = row["inferred_record_id"]
        for state in STATE_ORDER:
            source_value = float(wide.loc[record, state])
            prism_value = float(row[state])
            audit_rows.append(
                {
                    "inferred_record_id": record,
                    "state": state,
                    "source_csv_value": source_value,
                    "prism_value": prism_value,
                    "prism_minus_source": prism_value - source_value,
                    "mapping_note": "Prism rows lack IDs; row-to-record mapping is inferred",
                }
            )
    prism_audit = pd.DataFrame(audit_rows)

    wide.reset_index().to_csv(DATA_OUT / "Gcamp_state_record_means_all8.csv", index=False)
    long.to_csv(DATA_OUT / "Gcamp_state_epoch_details_all8.csv", index=False)
    summary.to_csv(DATA_OUT / "Gcamp_state_summary_all8.csv", index=False)
    omnibus.to_csv(DATA_OUT / "Gcamp_state_omnibus_stats_all8.csv", index=False)
    pairwise.to_csv(DATA_OUT / "Gcamp_state_pairwise_stats_all8.csv", index=False)
    omnibus_long = omnibus.assign(
        comparison="all_four_states",
        p_value_raw=omnibus["p_value"],
        p_value_adjusted=omnibus["p_value"],
    )[["analysis", "test", "comparison", "statistic", "df1", "df2", "p_value_raw", "p_value_adjusted", "record_n"]]
    pairwise_long = pd.DataFrame(
        {
            "analysis": "primary_pairwise",
            "test": "exact paired Wilcoxon with Holm correction",
            "comparison": pairwise["state_1"] + "_vs_" + pairwise["state_2"],
            "statistic": pairwise["wilcoxon_W"],
            "df1": "",
            "df2": "",
            "p_value_raw": pairwise["wilcoxon_p_raw"],
            "p_value_adjusted": pairwise["wilcoxon_p_holm_6"],
            "record_n": pairwise["record_n"],
        }
    )
    pd.concat([omnibus_long, pairwise_long], ignore_index=True).to_csv(
        DATA_OUT / "Gcamp_state_stats_all8_long.csv", index=False
    )
    prism_audit.to_csv(DATA_OUT / "Gcamp_Prism6_value_audit.csv", index=False)

    manifest_paths = result_sources + sorted(ROOT.rglob("*_analysis*.mat")) + [
        ROOT / "Code" / "analyzeGcampFQ.m",
        ROOT / "Project1.pzfx",
        ROOT / "gcamp.pdf",
    ]
    with (DOC_OUT / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["role", "path", "size_bytes", "last_write_time", "sha256"]
        )
        writer.writeheader()
        for path in manifest_paths:
            writer.writerow(
                {
                    "role": "source",
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "last_write_time": path.stat().st_mtime,
                    "sha256": sha256(path),
                }
            )

    write_report(wide, long, summary, omnibus, pairwise, prism_audit)


if __name__ == "__main__":
    main()
