# Assignment 4 · Cleveland 分类与回归

课程代码、运行结果及报告。数据与固定五折来自课程提供的 `assignment4-data.zip`，解压后将 `assignment4-data/` 放在 `assignment4/` 旁边，或通过 `--data-dir` 指定其路径。

## 复现

在 `assignment4/` 中使用 Python 3.13 运行。主要依赖版本在 `requirements.txt`，本次验证使用 scikit-learn 1.7.2。

```text
python -m pip install -r requirements.txt
python main.py
python main.py --data-dir "/path/to/assignment4-data"
```

`main.py` 会运行参考基线、独立复现实验和四项消融，重写 `figures/` 与 `results/`。参照 `results/tables.md` 核对报告数据。若要重新生成 Markdown 报告，给上述命令加 `--write-report`，该选项会覆盖现有 `report.md` 和 `analysis.md`。

## 文件

- `classify.py` 扫描线性核与 RBF 核 SVM，计算多数类参考线。
- `regress.py` 运行回归树深度扫描和代价复杂度剪枝，以及岭回归正则化扫描。
- `ablations.py` 运行特征移除、标准化、单次划分与二次特征四项对照。
- `experiment.py` 提供固定五折读取和评估函数。
- `main.py` 统一入口，`report_build.py` 生成表格、图和报告。
- `report.md` 包含实验报告与 AI 使用声明，`analysis.md` 包含讨论题。
- `results/` 包含基线 JSON、独立结果 JSON 和全部实验表。
- `results/fold_auc.md` 单独列出调参后两个 SVM 在同一测试折上的 AUC 与差值（线性核减 RBF 核），由默认运行命令重生。

## 亲自核查 AI 辅助结果

1. 运行上述命令，看 `results/tables.md` 的复现核对表。若使用课程自带核查命令，从数据目录执行 `python baselines.py --verify <assignment4/results/mine.json 的绝对路径>`。
2. 确认 SVM 的 AUC 使用 decision_function 连续分数，并核对 results/fold_auc.md 中逐折 AUC 的等权平均与分类总表一致。
3. 同时核对逐折回归均值基线和讲义中的零值。可逐折手算训练均值，再比较其在测试折上的 R²。
4. 报告末尾的 AI 使用声明记录了工具用途、验证方法和本人核查确认；更新实验后应同步核对该声明。

