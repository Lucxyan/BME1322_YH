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

## 已完成的核查

本人已完成对 AI 辅助结果的核查，具体方法见报告末尾的 AI 使用声明。

1. 已核对固定超参数基线。独立实现与课程脚本的 45 项指标完全一致，最大绝对差值为 0.000000，满足 0.01 的容差要求，详见 `results/tables.md`。
2. 已确认 SVM 的 AUC 使用 `decision_function()` 连续分数，并核对 `results/fold_auc.md` 中逐折 AUC 的等权平均与分类总表一致。
3. 已确认两条任务线使用相同的课程固定五折，填补、标准化和多项式变换在训练折内拟合；十次单次划分对照预先固定课程基线深度 3。
4. 已核对逐折训练均值参考线，五折平均 R² 为 −0.011653；其负值来自各训练折与测试折均值的差异。
5. 已检查报告中的结果、图表、讨论题与 AI 使用声明，并确认结论限定于本次评价协议和扫描范围。

### 核查结果的复现命令

当 `assignment4-data/` 与 `assignment4/` 相邻时，可在课程数据目录中重新运行以下命令。

```text
python baselines.py --verify ../assignment4/results/mine.json
```
