# POCD sleep analysis

Analysis scripts associated with a mouse study of postoperative sleep microstructure, Edinger–Westphal neuronal activity, and cognitive outcomes.

## Status

This is an initial, functionally organized research-code archive, **not a fully validated, one-command reproduction release**. Script selection is linked to existing manuscript methods and figure-workspace records; individual versions still require numerical matching to final manuscript outputs. No raw experimental recordings, unpublished manuscript, personal documents, or local provenance archives are included.

Python files passed static syntax checks. Analyses were not rerun during packaging. Personal workstation usernames were replaced with `USER`; algorithms were not changed. Legacy absolute paths and output-writing behavior remain: inspect and configure all input/output paths before running. Do not run against original data without backups.

## Functional organization

| Directory | Function | Manuscript |
|---|---|---|
| 01_behavior | Open field, NOR, Y-maze, fear conditioning | Fig.1, Fig.8, S8 |
| 02_sleep_architecture | Sleep duration, bouts, spectral summaries | Fig.1, S1–S3 |
| 03_spindle_SO_MA_coupling | Microarousals, spindles, SOs, coupling | Fig.2, S7 |
| 04_cfos_histology | Regional c-Fos density and co-labeling | Fig.3, S4 |
| 05_photometry_state_phase | State/event-aligned calcium and sigma phase | Fig.4 |
| 06_photometry_postoperative | Postoperative calcium and sigma associations | Fig.5 |
| 07_optogenetics | ChR2 and NpHR stimulation analyses | Fig.6, S9 |
| 08_chemogenetics_hM3Dq | Activation: sleep and oscillatory events | Fig.7, S5 |
| 09_chemogenetics_hM4Di | Inhibition: paired SAL/CNO sleep analyses | Fig.7, S6 |
| 10_golgi | Mouse-level morphology and spine analysis | Fig.8 |
| 11_human_genetics | Missing original analysis pipeline, documented only | Fig.3, S4 |

Legacy filenames retain historical figure numbering. `SCRIPT_MANIFEST.csv` provides the current functional mapping and file hashes. Module READMEs list scripts. `DEPENDENCY_IMPORTS.json` is a static import inventory, **not** a tested environment specification. It contains standard-library and project-local imports as well as third-party dependencies.

## Reproduction requirements

Python scripts use combinations of NumPy, pandas, SciPy, statsmodels and plotting tools; selected workflows additionally require MATLAB or Windows software such as Origin. Versions and complete dependency chains have not yet been frozen. Data are not bundled; a data-access statement and a tested execution order will be added when established.

The complete upstream 470/405-nm-to-delta-F/F preprocessing entry point and original BrainXcan/MR execution scripts have not been located. Figure assembly must not be mistaken for genetic-analysis reproduction. Behavioral rescue comparisons use independent groups, sleep SAL/CNO comparisons are paired, and Golgi inference uses the mouse as the biological unit.

## 中文说明

本仓库为论文代码的首次公开整理版，按功能收录70个脚本，包含本地模块依赖。原文件未改动；公开副本仅清理了个人工作站标识。尚未逐项复跑，也不能据此声称全部分析已完整复现。使用前请检查输入、输出路径和软件依赖；不要直接运行含固定路径的脚本。原始数据、论文全文和历史审计资料未上传。

## License and citation

No open-source license has been assigned in this initial archive. Public visibility does not grant additional reuse rights; third-party rights remain with their respective owners. A license and manuscript citation can be added after authorship and permissions are confirmed.
