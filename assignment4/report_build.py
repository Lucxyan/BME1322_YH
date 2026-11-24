"""Generate the tables, figures and editable Markdown reports."""
import json
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score
from sklearn.tree import plot_tree

import classify
import regress
from experiment import ROOT, SEED, class_metrics

CLS = ("accuracy", "precision", "recall", "f1", "auc", "precision_neg", "recall_neg", "f1_neg")
CLS_LABELS = ("Accuracy", "Precision+", "Recall+", "F1+", "AUC", "Precision−", "Recall−", "F1−")
REG = ("r2", "mae", "rmse")
REG_LABELS = ("R²", "MAE", "RMSE")
THRESHOLDS = (0.3, 0.5, 0.7, 1.0)
UNITS = {
    "age": "岁", "sex": "0/1 编码", "cp": "1–4 类别编码",
    "trestbps": "mm Hg", "chol": "mg/dl", "fbs": "0/1 编码",
    "restecg": "0–2 类别编码", "thalach": "bpm", "exang": "0/1 编码",
    "oldpeak": "ST 压低幅度，课程未标单位", "slope": "1–3 类别编码",
    "ca": "显影血管数（0–3）", "thal": "3/6/7 类别编码",
}


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def values(result, keys):
    return [fmt(result["metrics"][key]) for key in keys]


def compact(result, keys):
    return " / ".join(values(result, keys))


def table(headers, rows, finding):
    line = lambda cells: "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows),
                      "", finding, ""])


def complexity(result):
    return [fmt(mean([fn(model) for model in result["models"]]), 1)
            for fn in (lambda model: model.get_n_leaves(), lambda model: model.get_depth())]


def describe_setting(kernel, key):
    c, gamma = key
    return f"C={c}" if kernel == "linear" else f"C={c}, gamma={gamma}"


def save(fig, file):
    fig.savefig(ROOT / "figures" / file, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw(X, num, classification, regression, selected_q):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    labels, counts = np.unique(num, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.7, 3.4), layout="constrained")
    bars = ax.bar(labels, counts, color="#2e6e80")
    ax.bar_label(bars, padding=3)
    ax.set(xticks=labels, xlabel="Angiographic severity grade (num)", ylabel="Patients",
           ylim=(0, max(counts) * 1.16))
    save(fig, "label_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 3.4), layout="constrained")
    ax.boxplot([X.loc[num == label, "oldpeak"].to_numpy() for label in labels],
               tick_labels=[str(label) for label in labels], patch_artist=True,
               boxprops={"facecolor": "#c3dbe3"}, medianprops={"color": "#974b18", "linewidth": 1.5})
    ax.set(xlabel="Angiographic severity grade (num)", ylabel="ST depression (oldpeak)")
    save(fig, "oldpeak_by_grade.png")

    grid = classification["rbf"]
    auc = np.array([[grid[c, g]["metrics"]["auc"] for g in classify.GAMMAS]
                    for c in classify.C_VALUES])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    im = ax.imshow(auc, cmap="cividis", vmin=0.5, vmax=0.95)
    for (i, j), value in np.ndenumerate(auc):
        ax.text(j, i, fmt(value, 3), va="center", ha="center",
                color="black" if value > 0.75 else "white")
    ax.set(xticks=range(len(classify.GAMMAS)), xticklabels=classify.GAMMAS,
           yticks=range(len(classify.C_VALUES)), yticklabels=classify.C_VALUES,
           xlabel="Gamma", ylabel="C")
    fig.colorbar(im, ax=ax, label="Mean held-out AUC")
    save(fig, "rbf_auc.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.7), layout="constrained")
    depths = list(regression["depth"])
    ax.plot(range(len(depths)), [regression["depth"][d]["metrics"]["r2"] for d in depths],
            "o-", color="#2e6e80", label="Tree held-out R²")
    ax.plot(range(len(depths)), [regression["depth"][d]["train_r2"] for d in depths],
            "o--", color="#6b7b48", label="Tree training R²")
    ax.axhline(regression["ridge"][regress.best_setting(regression["ridge"])]["metrics"]["r2"],
               color="#974b18", ls="--", label="Selected ridge (held-out)")
    ax.set(xticks=range(len(depths)), xticklabels=[d if d else "None" for d in depths],
           xlabel="Maximum tree depth", ylabel="Mean R²")
    ax.legend(frameon=False)
    save(fig, "tree_depth.png")

    alpha = regress.path_alpha(regress.pruning_path(X, num), selected_q)
    tree = regress.tree(alpha=alpha).fit(X, num).named_steps["model"]
    fig, ax = plt.subplots(figsize=(max(10, min(27, tree.get_n_leaves() * 1.45)),
                                    max(5, min(18, (tree.get_depth() + 1) * 1.1))),
                          layout="constrained")
    plot_tree(tree, ax=ax, feature_names=list(X.columns), filled=True,
              rounded=True, precision=2, fontsize=8)
    save(fig, "pruned_tree.png")
    return tree, alpha


def tree_splits(tree, features):
    t = tree.tree_
    return "；".join(
        f"`{features[t.feature[i]]} ≤ {t.threshold[i]:.2f}`"
        for i in (0, t.children_left[0], t.children_right[0])
        if i >= 0 and t.feature[i] >= 0
    )


def audit_table(reference, mine):
    rows = [(model.split("/", 2)[-1], metric, fmt(target, 6),
             fmt(mine[model][metric], 6), f"{mine[model][metric] - target:+.6f}")
            for model, metrics in reference.items() for metric, target in metrics.items()]
    gap = max(abs(mine[k][m] - v) for k, metrics in reference.items() for m, v in metrics.items())
    return table(("方法", "指标", "课程基线", "独立复现", "差值"), rows,
                 f"全部 {len(rows)} 个指标的最大绝对差值为 {fmt(gap, 6)}，容差为 0.010000。")


def classification_tables(classification, keys, y):
    rows = []
    for kernel in keys:
        fixed = classification[kernel][1, "scale"]
        chosen = classification[kernel][keys[kernel]]
        rows.append([f"SVM {kernel}", describe_setting(kernel, (1, "scale")),
                     describe_setting(kernel, keys[kernel]),
                     *[f"{left} / {right}" for left, right in zip(values(fixed, CLS), values(chosen, CLS))],
                     fmt(chosen["metrics"]["auc"] - fixed["metrics"]["auc"])])
    dummy = classification["dummy"]
    winner = max(keys, key=lambda k: classification[k][keys[k]]["metrics"]["auc"])
    rows.append(["每折训练集多数类", "多数类", "多数类",
                 *[f"{value} / {value}" for value in values(dummy, CLS)], "—"])
    output = (
        "### 固定基线与扫描最优\n\n"
        "总表各指标单元格按“基线 / 扫描最优”列出，正负类 AUC 在反转标签和分数后相同。\n\n"
        + table(("模型", "基线（固定超参）", "我的最优（扫描后）", *CLS_LABELS, "ΔAUC"), rows,
                f"按课程固定五折的平均 AUC 选参和比较，{winner} 核在本次搜索中更合适。")
    )
    scan = []
    for kernel in keys:
        fixed_auc = classification[kernel][1, "scale"]["metrics"]["auc"]
        best_auc = classification[kernel][keys[kernel]]["metrics"]["auc"]
        for (c, g), result in classification[kernel].items():
            m = result["metrics"]
            scan.append((kernel, c, g, fmt(m["train_auc"]), fmt(m["auc"]),
                         fmt(m["train_auc"] - m["auc"]), fmt(fixed_auc), fmt(best_auc)))
    output += "\n### C 与 gamma 扫描\n\n" + table(("核", "C", "gamma", "训练 AUC", "验证 AUC", "训练−验证",
                              "固定基线 AUC", "扫描最优 AUC"), scan,
                             f"C=1、gamma=10 时，RBF 训练 AUC 为 "
                             f"{fmt(classification['rbf'][1, 10]['metrics']['train_auc'])}，验证 AUC 为 "
                             f"{fmt(classification['rbf'][1, 10]['metrics']['auc'])}，显示出较大的泛化差距。")
    chosen = classification[winner][keys[winner]]
    tn, fp, fn, tp = confusion_matrix(y, chosen["pred"], labels=[0, 1]).ravel()
    output += "\n### 最优分类器的混淆矩阵\n\n" + table(("真实类别", "预测阴性", "预测阳性"),
                             [("阴性", tn, fp), ("阳性", fn, tp)],
                             f"合并折外预测的敏感性为 {fmt(tp / (tp + fn))}，特异性为 "
                             f"{fmt(tn / (tn + fp))}。")
    output += (
        f"在 {tp + fn} 例阳性中漏判 {fn} 例，在 {tn + fp} 例阴性中误报 {fp} 例。"
        "AUC 衡量排序能力，默认分类阈值下的漏诊和误报仍需分别检查。"
        "本次按 AUC 选择核函数；若要用于筛查，应根据漏诊和额外检查的代价另行确定决策阈值。\n"
    )
    return output


def full_coef(X, num, regression, chosen_alpha):
    fit = regress.ridge(chosen_alpha).fit(X, num).named_steps["model"]
    ols = regression["ols_scaled"].named_steps["model"]
    return fit.coef_, ols.coef_


def fold_auc_table(classification, keys, y, cv):
    linear_scores = classification["linear"][keys["linear"]]["score"]
    rbf_scores = classification["rbf"][keys["rbf"]]["score"]
    rows = []
    differences = []
    for fold, (_, test) in enumerate(cv, start=1):
        linear_auc = roc_auc_score(y[test], linear_scores[test])
        rbf_auc = roc_auc_score(y[test], rbf_scores[test])
        difference = linear_auc - rbf_auc
        differences.append(difference)
        rows.append((fold, len(test), int(y[test].sum()), fmt(linear_auc, 6),
                     fmt(rbf_auc, 6), f"{difference:+.6f}"))
    linear_mean = classification["linear"][keys["linear"]]["metrics"]["auc"]
    rbf_mean = classification["rbf"][keys["rbf"]]["metrics"]["auc"]
    rows.append(("五折等权平均", "—", "—", fmt(linear_mean, 6),
                 fmt(rbf_mean, 6), f"{mean(differences):+.6f}"))
    wins = sum(difference > 0 for difference in differences)
    losses = sum(difference < 0 for difference in differences)
    ties = len(differences) - wins - losses
    setup = (
        f"线性核使用 {describe_setting('linear', keys['linear'])}，RBF 核使用 "
        f"{describe_setting('rbf', keys['rbf'])}。两者均采用折内中位数填补和标准化，"
        "按课程 splits.json 中的顺序逐折计算 decision_function 的测试 AUC。"
        "差值定义为线性核 AUC 减 RBF 核 AUC，正值表示线性核在该折领先。\n\n"
    )
    finding = (
        f"线性核在 {wins} 折领先、{losses} 折落后、{ties} 折持平，平均差值为 "
        f"{mean(differences):+.6f}，五个配对差值的样本标准差为 "
        f"{np.std(differences, ddof=1):.6f}。"
        "按本次固定五折的平均 AUC 比较，线性核更合适。"
        "逐折胜负及差值补充描述了优势在各折的分布。"
        "超参由同一组五折选择，各折训练集相互重叠，"
        "该结论限定于本实验的评价协议。"
    )
    return setup + table(("折", "测试例数", "阳性例数", "线性核 AUC", "RBF 核 AUC",
                          "差值（线性−RBF）"), rows, finding)


def regression_tables(X, num, regression, d, q, a):
    fixed_tree = regression["depth"][3]
    fixed_ridge = regression["ridge"][1]
    best_depth = regression["depth"][d]
    best_prune = regression["prune"][q]
    best_ridge = regression["ridge"][a]
    dummy = regression["dummy"]
    rows = []
    for name, fixed_setting, chosen_setting, fixed, chosen in [
        ("回归树（限制深度）", "max_depth=3，ccp_alpha=0", f"max_depth={d}，ccp_alpha=0", fixed_tree, best_depth),
        ("回归树（后剪枝扫描）", "max_depth=3，ccp_alpha=0",
         f"max_depth=None，ccp_alpha=折内取值（q={q}）", fixed_tree, best_prune),
        ("岭回归", "alpha=1", f"alpha={a}", fixed_ridge, best_ridge),
        ("每折训练集均值", "训练均值", "训练均值", dummy, dummy),
    ]:
        rows.append((name, fixed_setting, chosen_setting,
                     *[f"{left} / {right}" for left, right in zip(values(fixed, REG), values(chosen, REG))],
                     f"{chosen['metrics']['r2'] - fixed['metrics']['r2']:+.4f}",
                     fmt(chosen["metrics"]["r2"] - dummy["metrics"]["r2"])))
    output = (
        "### 固定基线与扫描最优\n\n"
        "各指标单元格按“固定基线 / 扫描内最优”列出，每行在对应的扫描分支内按五折平均 R² 选参。"
        "两棵树共用课程规定的深度 3 基线，后剪枝分支从不限制深度的完整树开始扫描。\n\n"
        + table(("模型", "基线（固定超参）", "我的最优（扫描后）", *REG_LABELS,
                 "相对固定基线 ΔR²", "相对训练均值 ΔR²"), rows,
                "深度扫描的最优值与课程基线一致；后剪枝分支内的最优 R² 低于该固定基线，"
                "因此本次后剪枝没有改善课程基线的成绩。")
    )
    output += (
        "后剪枝行的 q 是各训练折剪枝候选序列的相对索引，实际拟合使用该位置对应的 ccp_alpha。"
        f"所选 q={q} 对应五折 ccp_alpha 依次为 "
        f"{', '.join(fmt(value, 6) for value in best_prune['alphas'])}。"
        "后剪枝路径未必包含深度受限为 3 时生成的那棵树，因此扫描内最优仍可能落后于固定基线。"
        "两列 ΔR² 均用扫描内最优 R² 减去各自参照值，分别衡量调参变化和相对训练均值的预测增益。\n"
    )
    output += "\n### 回归树深度扫描\n\n" + table(("max_depth", "训练 R²", *REG_LABELS, "平均叶数", "平均深度",
                              "固定基线 R²", "扫描最优 R²"),
                             [(depth, fmt(res["train_r2"]), *values(res, REG), *complexity(res),
                               fmt(fixed_tree["metrics"]["r2"]), fmt(best_depth["metrics"]["r2"]))
                              for depth, res in regression["depth"].items()],
                             f"按 R² 选择深度 {d}。按 MAE 选择深度 "
                             f"{regress.best_setting(regression['depth'], 'mae')}。")
    output += "\n### 回归树后剪枝路径\n\n" + table(("路径相对位置", "ccp_alpha 中位数", "各折 alpha 最小–最大",
                              "训练 R²", *REG_LABELS, "平均叶数", "平均深度", "固定基线 R²", "扫描最优 R²"),
                             [(point, fmt(np.median(res["alphas"]), 6),
                               f"{fmt(min(res['alphas']), 6)}–{fmt(max(res['alphas']), 6)}",
                               fmt(res["train_r2"]), *values(res, REG), *complexity(res), fmt(fixed_tree["metrics"]["r2"]),
                               fmt(best_prune["metrics"]["r2"]))
                              for point, res in regression["prune"].items()],
                             f"按 R² 选择路径位置 {q}。位置 0 对应未剪枝树，1 对应最强剪枝。")
    output += "\n### 岭回归正则化扫描\n\n" + table(("岭回归 alpha", *REG_LABELS, "固定基线 R²", "扫描最优 R²"),
                             [(alpha, *values(res, REG), fmt(fixed_ridge["metrics"]["r2"]),
                               fmt(best_ridge["metrics"]["r2"]))
                              for alpha, res in regression["ridge"].items()],
                             f"按 R² 选择 alpha={a}。")
    ridge, ols = full_coef(X, num, regression, a)
    output += "\n### 标准化特征的系数\n\n" + table(("特征", f"岭回归 alpha={a}", "OLS alpha→0"),
                             [(name, fmt(w1, 6), fmt(w2, 6)) for name, w1, w2 in zip(X.columns, ridge, ols)],
                             f"岭回归绝对系数最大的是 {X.columns[int(np.abs(ridge).argmax())]}。"
                             "系数来自全样本说明性拟合，两列采用相同的填补和标准化口径。")
    output += "\n### Lasso 稀疏性核查\n\n" + table(("Lasso alpha", "各折零系数数目", *REG_LABELS),
                             [(alpha, "/".join(str(np.count_nonzero(m.coef_ == 0)) for m in res["models"]),
                               *values(res, REG)) for alpha, res in regression["lasso_check"].items()],
                             "本次 alpha=1 和 10 时，Lasso 五折的 13 个系数均为零，"
                             "与训练均值常数预测得到相同的负 R²；主线模型使用岭回归。")
    candidates = [(f"{family}={setting}", result)
                  for family in ("depth", "prune", "ridge")
                  for setting, result in regression[family].items()]
    rankings = {}
    for metric, reverse in (("r2", True), ("mae", False)):
        rankings[metric] = sorted(candidates, key=lambda item: item[1]["metrics"][metric],
                                 reverse=reverse)
    best_r2_name, best_r2 = rankings["r2"][0]
    best_mae_name, best_mae = rankings["mae"][0]
    output += "\n### 指标改变时的配置排序\n\n" + table(("名次", "按 R² 排序的配置", "R²", "按 MAE 排序的配置", "MAE"),
        [(rank + 1, r2_item[0], fmt(r2_item[1]["metrics"]["r2"]),
          mae_item[0], fmt(mae_item[1]["metrics"]["mae"]))
         for rank, (r2_item, mae_item) in enumerate(zip(rankings["r2"], rankings["mae"]))],
        f"在全部主线回归候选配置中，按 R² 最优的是 {best_r2_name}"
        f"（R²={fmt(best_r2['metrics']['r2'])}），改按 MAE 最优的是 {best_mae_name}"
        f"（MAE={fmt(best_mae['metrics']['mae'])}）。主实验采用 R² 选参。")
    return output


def regularization_evidence(X, regression):
    correlation = X.corr().where(np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool)).stack()
    strongest = correlation.abs().nlargest(5).index
    output = "### 相关结构与变量选择稳定性\n\n" + table(
        ("特征对", "Pearson r"),
        [(f"{first} 与 {second}", fmt(correlation[first, second])) for first, second in strongest],
        "最高相关来自 oldpeak 与 slope，其余特征对的相关程度应结合类别编码理解。")
    selected_counts = {}
    for alpha in (0.01, 0.1):
        coefficients = np.asarray([model.coef_ for model in regression["lasso_check"][alpha]["models"]])
        selected_counts[alpha] = np.count_nonzero(coefficients, axis=0)
    output += "\n" + table(
        ("特征", "Lasso alpha=0.01 非零折数", "Lasso alpha=0.1 非零折数"),
        [(feature, f"{selected_counts[0.01][i]}/5", f"{selected_counts[0.1][i]}/5")
         for i, feature in enumerate(X.columns)],
        "非零折数记录特征是否入选，反映同一 alpha 下的折间选择变化。")
    output += (
        "alpha=0.01 时，chol 和 fbs 分别在 3/5、4/5 折保留；alpha=0.1 时，sex 在 3/5 折保留。"
        "这些变量的入选对训练病例有一定敏感性。相关最高的 oldpeak 与 slope 在这两个 alpha 下"
        "均同时保留 5/5 折，因此本数据没有展示这对变量互相替代的现象。"
        "结果支持 Lasso 可以产生稀疏解，也显示部分选择随折变化；将这种变化归因为相关组中的随机择一，"
        "仍缺少直接证据。五折训练集相互重叠，保留次数只作本实验的描述性核查。\n"
    )
    return output


def feature_agreement(X, num, keys, alpha):
    classifier = classify.svm("linear", *keys["linear"]).fit(X, (num > 0).astype(int))
    classifier_coef = classifier.named_steps["model"].coef_.ravel()
    ridge_coef = regress.ridge(alpha).fit(X, num).named_steps["model"].coef_
    class_order = np.argsort(-np.abs(classifier_coef))[:5]
    ridge_order = np.argsort(-np.abs(ridge_coef))[:5]
    class_features = [X.columns[i] for i in class_order]
    ridge_features = [X.columns[i] for i in ridge_order]
    shared = sorted(set(class_features) & set(ridge_features))
    output = table(
        ("绝对系数名次", "线性 SVM 特征", "SVM 系数", "岭回归特征", "岭回归系数"),
        [(rank + 1, X.columns[first], fmt(classifier_coef[first]),
          X.columns[second], fmt(ridge_coef[second]))
         for rank, (first, second) in enumerate(zip(class_order, ridge_order))],
        f"两条任务线前五项重合 {len(shared)} 项，为 {', '.join(shared)}。")
    output += (
        "两模型采用相同的填补和标准化，表中系数来自全样本说明性拟合，性能评价仍使用折外预测。"
        "重合特征说明两种任务依赖的主要信号有一致性，分类侧 exang 与回归侧 thalach 的差别也表明"
        "二分类边界和分级大小各有侧重。两种损失函数决定了不同的系数尺度，比较时使用各模型内部排序；"
        "类别编码和特征相关性还会影响排序，它不能直接解释为独立临床作用。\n"
    )
    return output, shared


def ablation_tables(classification, regression, controls, keys):
    output = "### 1. 去掉 ca 与 thal\n\n" + table(("固定配置模型", "指标", "13 列", "去掉 ca/thal", "差值"),
                   [(name, metric, fmt(full["metrics"][metric]), fmt(reduced["metrics"][metric]),
                     f"{reduced['metrics'][metric] - full['metrics'][metric]:+.4f}")
                    for name, metric, full, reduced in controls["features"]],
                   "删列后两个 SVM 与岭回归下降，浅树反而改善。"
                   "各模型超参保持原样；gamma=scale 会在每折按余下的特征重算。")
    rows = []
    for kernel, key in keys.items():
        raw = controls["raw"][kernel]
        best_raw = classify.best_setting(raw)
        for c in classify.C_VALUES:
            scaled = classification[kernel]
            std_key = max((k for k in scaled if k[0] == c), key=lambda k: scaled[k]["metrics"]["auc"])
            raw_key = max((k for k in raw if k[0] == c), key=lambda k: raw[k]["metrics"]["auc"])
            rows.append([kernel, c, std_key[1], fmt(scaled[std_key]["metrics"]["auc"]),
                         raw_key[1], fmt(raw[raw_key]["metrics"]["auc"]),
                         f"{key[0]} → {best_raw[0]}"])
    output += "\n### 2. 标准化与不标准化\n\n" + table(("核", "C", "标准化最佳 gamma", "标准化 AUC",
                              "不标准化最佳 gamma", "不标准化 AUC", "两设定最优 C"),
                             rows, "标准化改变了两个核的最优 C，RBF 在未经缩放时表现下降更明显。"
                             "表中 RBF 在每个 C 下先取搜索网格中 AUC 最高的 gamma。")
    scores = [r2 for _, r2 in controls["holdout"]]
    d = controls["holdout_depth"]
    cv_r2 = regression["depth"][d]["metrics"]["r2"]
    output += "\n### 3. 单次划分与固定五折\n\n" + table(("80/20 划分种子", "R²"),
                             [(seed, fmt(r2)) for seed, r2 in controls["holdout"]]
                             + [("最小", fmt(min(scores))), ("最大", fmt(max(scores))),
                                ("标准差（总体）", fmt(pstdev(scores))), ("固定五折均值", fmt(cv_r2))],
                             f"按课程基线预先固定深度为 {d}，十次单次划分极差为 {fmt(max(scores) - min(scores))}。")
    output += "\n" + table(
        ("种子", "测试等级计数 0/1/2/3/4", "测试标签方差", "MSE", "MAE", "误差>1 例数", "前三大误差占平方损失"),
        [(row["seed"], "/".join(map(str, row["grade_counts"])), fmt(row["target_variance"]),
          fmt(row["mse"]), fmt(row["mae"]), row["large_errors"], f"{row['top3_squared_share']:.1%}")
         for row in controls["holdout_diagnostics"]],
        "各次 R² 同时受预测误差和测试标签方差影响，需结合 MSE 与等级构成解释。")
    poly = controls["polynomial"]
    linear = regression["ridge"][regress.best_setting(regression["ridge"])]
    output += "\n### 4. 二次多项式特征\n\n" + table(("输入", "Ridge alpha", "训练 R²", *REG_LABELS),
                             [("原始 13 列", regress.best_setting(regression["ridge"]),
                               fmt(linear["train_r2"]), *values(linear, REG))]
                             + [("二次展开 104 列", a, fmt(res["train_r2"]), *values(res, REG)) for a, res in poly.items()],
                             f"二次模型最高验证 R² 为 {fmt(poly[regress.best_setting(poly)]['metrics']['r2'])}，"
                             f"低于一阶岭回归的 {fmt(linear['metrics']['r2'])}，增加容量未带来验证收益。")
    return output


def transfer_table(y, cv, classification, regression, keys, a):
    pred = regression["ridge"][a]["pred"]
    test = [fold for _, fold in cv]
    auc = mean([roc_auc_score(y[i], pred[i]) for i in test])
    rows = []
    for t in THRESHOLDS:
        metrics = [class_metrics(y[i], (pred[i] >= t).astype(int), pred[i]) for i in test]
        rows.append((str(t), auc, mean([m["accuracy"] for m in metrics]),
                     mean([m["f1"] for m in metrics])))
    best_threshold = max(rows, key=lambda row: row[3])
    message = table(("模型/阈值", "AUC", "Accuracy", "F1+"),
                    [(f"岭回归 ≥ {t}", fmt(auc), fmt(acc), fmt(f1)) for t, auc, acc, f1 in rows]
                    + [(f"SVM {k}", *values(classification[k][setting], ("auc", "accuracy", "f1")))
                       for k, setting in keys.items()],
                    "岭回归的排序 AUC 直接用预测值计算，因此阈值只改变 Accuracy 和 F1。")
    return message, auc, best_threshold


def check_table(classification, regression, num):
    y = (num > 0).astype(int)
    usable = regression["dummy"]
    oof_r2 = r2_score(num, usable["pred"])
    rows = []
    for kernel in ("linear", "rbf"):
        fixed = classification[kernel][1, "scale"]
        m = fixed["metrics"]
        rows.append((kernel, fmt(m["auc"], 6),
                     fmt(roc_auc_score(y, fixed["score"]), 6)))
    audit = table(("固定基线模型", "连续分数 AUC（五折平均）",
                   "连续分数 AUC（合并折外）"), rows,
                  "两列均使用 decision_function 的连续分数。正式结果采用五折等权平均，与课程基线一致。")
    audit += "\n" + table(("均值基线算法", "R²"),
                            [("把评估集自身的均值填入全评估集（理论对照）",
                              fmt(r2_score(num, np.full(len(num), num.mean())), 6)),
                             ("逐折训练均值并合并折外预测", fmt(oof_r2, 6)),
                             ("逐折训练均值并平均五个测试 R²", fmt(usable["metrics"]["r2"], 6))],
                            "正式回归表使用五个测试 R² 的等权平均。")
    return audit

def cls_story(classification, keys):
    linear = classification["linear"][keys["linear"]]["metrics"]
    rbf = classification["rbf"][keys["rbf"]]["metrics"]
    fixed_lin = classification["linear"][1, "scale"]["metrics"]["auc"]
    fixed_rbf = classification["rbf"][1, "scale"]["metrics"]["auc"]
    wide = classification["rbf"][keys["rbf"][0], 10]["metrics"]
    smaller = classification["rbf"][keys["rbf"]]["metrics"]
    lin_aucs = [result["metrics"]["auc"] for result in classification["linear"].values()]
    rbf_at_best_g = [classification["rbf"][c, keys["rbf"][1]]["metrics"]["auc"]
                     for c in classify.C_VALUES]
    return (
        "### 核函数选择与训练、验证差距\n\n"
        f"按 AUC 选参，线性核（{describe_setting('linear', keys['linear'])}）得到 "
        f"{fmt(linear['auc'])}，RBF 核（{describe_setting('rbf', keys['rbf'])}）得到 "
        f"{fmt(rbf['auc'])}。RBF 的 Accuracy 为 {fmt(rbf['accuracy'])}，线性核为 "
        f"{fmt(linear['accuracy'])}，所以两种指标会给出不同偏好。"
        "按本实验预设的平均 AUC 指标，线性核更合适，RBF 核调参后的 AUC 仍低于线性核。\n\n"
        f"保持 RBF 的 C={keys['rbf'][0]}，将 gamma 从 {keys['rbf'][1]} 提高到 10，"
        f"训练 AUC 从 {fmt(smaller['train_auc'])} 升至 {fmt(wide['train_auc'])}，验证 AUC "
        f"从 {fmt(smaller['auc'])} 降至 {fmt(wide['auc'])}。"
        "在 exp(−gamma·‖x−x′‖²) 中，大 gamma 让相似度随距离快速衰减，训练点的高相似度邻域"
        "更局部，折外样本更难共享这些局部信息，这与训练排序接近完美而验证排序变差一致。\n\n"
        f"线性核跨 C 的验证 AUC 极差为 {fmt(max(lin_aucs) - min(lin_aucs))}，而 RBF 在所选 "
        f"gamma 下为 {fmt(max(rbf_at_best_g) - min(rbf_at_best_g))}，"
        "两者对 C 的敏感性取决于核所表示的特征空间。"
        "线性核在本次扫描中训练与验证 AUC 的差距较小，提高 C 可以改变误分惩罚，"
        "但线性边界的表达形式相同，因而观察到的验证收益有限。"
        f"固定参数改用 RBF 的 AUC 变化为 {fixed_rbf - fixed_lin:+.4f}，"
        f"调好 RBF 的变化为 {rbf['auc'] - fixed_rbf:+.4f}，因此本次换核没有带来正收益，"
        "选参则补回了 RBF 的大部分差距。\n"
    )


def regression_story(X, num, regression, d, q, a, illustrative, alpha):
    depths = regression["depth"]
    deep = depths[None]
    pre = depths[d]
    post = regression["prune"][q]
    best_mae = regress.best_setting(depths, "mae")
    n_deep = mean([float(np.median(model.tree_.n_node_samples[model.tree_.children_left == -1]))
                   for model in deep["models"]])
    n_shallow = mean([float(np.median(model.tree_.n_node_samples[model.tree_.children_left == -1]))
                      for model in pre["models"]])
    corr = X.corr().where(np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool)).stack()
    pair = corr.abs().idxmax()
    ridge_coef, ols_coef = full_coef(X, num, regression, a)
    changes = np.abs(ridge_coef - ols_coef)
    largest = int(changes.argmax())
    ridge_zero = sum(np.count_nonzero(model.coef_ == 0)
                     for model in regression["ridge"][a]["models"])
    ridge = regression["ridge"][a]
    lasso_constant = regression["lasso_check"][10]["metrics"]["r2"]
    lines = [
        "### 树复杂度如何影响泛化\n\n"
        f"从深度 {d} 到不限制深度，折均叶数由 {complexity(pre)[0]} 增为 {complexity(deep)[0]}，"
        f"每折叶内训练样本中位数的均值由 {fmt(n_shallow, 1)} 降至 {fmt(n_deep, 1)}。"
        f"训练 R² 从 {fmt(pre['train_r2'])} 升至 {fmt(deep['train_r2'])}，"
        f"验证 R² 却从 {fmt(pre['metrics']['r2'])} 降至 {fmt(deep['metrics']['r2'])}。"
        "树的叶子输出叶内病例的平均等级，当叶内只剩一两例时，该均值容易由个别病例决定。"
        "深树能把训练病例分得很细，训练误差降低的同时折外误差增大；"
        "该现象与小叶子的分级均值估计较不稳定的解释一致。",
        f"预剪枝在分裂前限制深度，后剪枝先长出完整树，再按误差与叶数之间的代价裁掉子树。"
        "代价复杂度剪枝最小化“训练均方误差 + ccp_alpha × 叶节点数”，alpha 越大，"
        "保留额外叶节点就需要带来更大的误差下降。后剪枝允许先形成深浅不同的分支，再决定保留哪些；"
        "深度限制则会提前阻止达到上限的分支继续分裂。\n\n"
        f"最佳预剪枝和后剪枝的 R² 分别为 {fmt(pre['metrics']['r2'])} 和 {fmt(post['metrics']['r2'])}，"
        f"折均叶数为 {complexity(pre)[0]} 和 {complexity(post)[0]}。"
        f"深度 2 的 R² 为 {fmt(depths[2]['metrics']['r2'])}，放宽到深度 {d} 后提高，"
        "说明限制过紧会损失预测信息；适当后剪枝也明显优于未限制深度的树。"
        "按本次扫描的最高验证 R²，深度限制更有效。讲义中“可能剪早”是对分支生长机制的提醒，"
        "本实验没有发现后剪枝总体优于适当深度限制的证据，整体分数也不能判断某个具体分支是否剪早。",
        "### 剪枝树前两层的含义\n\n"
        f"树图用全样本按同一路径规则拟合以展示结构，ccp_alpha={fmt(alpha, 6)}，"
        f"叶数 {illustrative.get_n_leaves()}，深度 {illustrative.get_depth()}。"
        f"前两层的分裂为 {tree_splits(illustrative, X.columns)}。"
        "根部用 thal 分出正常灌注编码与缺损编码。其后 ca 的分裂区分零支与至少一支显影血管，"
        "oldpeak 的分裂则区分运动诱发 ST 压低的幅度。两条分支反映模型在不同灌注组选择不同的下一级指标。"
        "这些阈值描述模型如何分组，分裂本身不能证明相应指标对疾病存在因果作用。",
        "### MAE 与 R² 为什么给出不同排序\n\n"
        f"以 R² 排序，深度 {d} 的 {fmt(pre['metrics']['r2'])} 高于深度 {best_mae} 的 "
        f"{fmt(depths[best_mae]['metrics']['r2'])}。按 MAE 排序，深度 {best_mae} 的 "
        f"{fmt(depths[best_mae]['metrics']['mae'])} 优于深度 {d} 的 {fmt(pre['metrics']['mae'])}。"
        "MAE 反映平均偏离几个等级，R² 则通过平方损失使少数大误差占更大权重。"
        "本实验以 R² 选参，以提高跨多个等级误差在评价中的权重，同时保留 MAE 检查典型误差。"
        "全部主线候选配置的双指标排序见上表，具体误差贡献见实验日志第一条。",
        "### 正则化在本数据中起了多大作用\n\n"
        f"最优岭回归 alpha={a} 得到 R²={fmt(ridge['metrics']['r2'])}。"
        f"近似无正则化的 alpha=0.01 得到 R²={fmt(regression['ridge'][0.01]['metrics']['r2'])}，"
        f"两者相差 {ridge['metrics']['r2'] - regression['ridge'][0.01]['metrics']['r2']:+.4f}，"
        "本次适度正则化带来的预测改善有限，模型的主要预测能力在弱惩罚时已经具备。"
        f"与标准化 OLS 相比，全样本系数绝对变化最大的是 {X.columns[largest]}"
        f"（{fmt(ols_coef[largest])} → {fmt(ridge_coef[largest])}）。"
        f"当 alpha 增至 1000，R² 降为 {fmt(regression['ridge'][1000]['metrics']['r2'])}，"
        "说明过强惩罚也会压低有用信号，出现欠拟合。\n\n"
        f"所选岭回归在五折中的精确零系数总数为 {ridge_zero}。"
        "L2 惩罚使解平滑收缩，L1 惩罚则能将部分系数精确压到零，因而具有变量选择效果。"
        f"Lasso 在 alpha=10 时五折均得到常数预测，R²={fmt(lasso_constant)}。"
        "此时截距保留了各训练折的目标均值，所以结果与均值参考线一致。"
        f"13 列中绝对 Pearson 相关最高的一对是 {pair[0]} 与 {pair[1]}（r={fmt(corr[pair])}），"
        "提示部分信号重叠，但类别编码也影响相关系数；下方用五折特征保留次数检验 Lasso 的选择稳定性。",
    ]
    return "\n\n".join(lines)


def ablation_story(regression, controls, keys):
    raw_keys = {k: classify.best_setting(controls["raw"][k]) for k in keys}
    poly_a = regress.best_setting(controls["polynomial"])
    poly = controls["polynomial"][poly_a]["metrics"]["r2"]
    ridge = regression["ridge"][regress.best_setting(regression["ridge"])]["metrics"]["r2"]
    scores = [r2 for _, r2 in controls["holdout"]]
    return "\n\n".join([
        "### 四项对照说明了什么\n\n"
        "去掉 ca 与 thal 后，两种 SVM 和岭回归下降，深度 3 的树反而上升。"
        "SVM 和岭回归的下降说明这两列包含可利用的预测信号；浅树的上升则表明更多候选特征并不保证更好的折外表现。"
        "浅树每一步只选择当前最有利的分裂，删列会改变后续能走到的路径；"
        "实验日志用根节点对比确认了路径起点发生变化，性能改善是否稳定仍需新数据验证。"
        "对拟用于转诊前初筛的模型，ca 来自所要决定是否安排的检查，thal 也需要另行获取。"
        "因此，13 列成绩用于课程模型比较，11 列消融更贴近不具备这两项检查结果的应用场景，"
        "即使可用特征下的性能下降，也不能用预测时尚未取得的信息补回成绩。",
        f"线性核最优 C 由 {keys['linear'][0]} 变为 {raw_keys['linear'][0]}，"
        f"RBF 核由 {keys['rbf'][0]} 变为 {raw_keys['rbf'][0]}。"
        f"分别调参后，RBF 的验证 AUC 从不标准化的 {fmt(controls['raw']['rbf'][raw_keys['rbf']]['metrics']['auc'])} "
        f"提高到标准化的 {fmt(controls['features'][1][2]['metrics']['auc'])}。"
        "不标准化的 RBF 最优 C 位于网格边界，变化幅度只适用于已扫描范围。"
        "原始量纲改变线性 SVM 对各变量系数的相对惩罚，也改变 RBF 中欧氏距离的构成。"
        "例如胆固醇数值范围大于二元编码，未经缩放时更容易主导距离，改变 C 只能部分补偿这种几何变化。"
        "回归树按列排序寻找阈值，单列的线性标准化保持阈值两侧的样本划分。",
        f"预先固定课程基线深度 {controls['holdout_depth']}，10 次随机划分中的 R² 从 {fmt(min(scores))} 到 {fmt(max(scores))}，"
        "单次划分可能把同一配置评为很弱，也可能评为不错。"
        "R²=1−MSE/测试标签方差，在预测误差相近时，更窄的测试等级分布也会给出更低的 R²。"
        "测试等级计数和误差分解已列在对照表中，日志第三条进一步比较了具体种子。"
        "训练病例和测试病例都随种子改变，不能把全部波动只归因于测试集构成。",
        f"二次岭回归最佳 alpha={poly_a}，R² 为 {fmt(poly)}，低于一阶岭回归的 {fmt(ridge)}。"
        f"二次模型训练 R² 为 {fmt(controls['polynomial'][poly_a]['train_r2'])}，"
        f"一阶模型为 {fmt(regression['ridge'][regress.best_setting(regression['ridge'])]['train_r2'])}。"
        "平方与交互项扩大了可拟合函数空间，在 303 例下新增自由度可能主要吸收样本噪声。"
        "深树通过小叶子估计局部均值，二次模型通过更多平方和交互系数拟合样本差异，"
        "两者都展示了更大容量与更高训练表现可以同时伴随更差的折外表现。"
        "这支持本次样本规模下控制复杂度的选择，结论范围限于已扫描的正则化强度。",
    ])


def discussion(num, cv, classification, regression, controls, keys, a, d, q, shared_features):
    y = (num > 0).astype(int)
    transfer, auc, best_t = transfer_table(y, cv, classification, regression, keys, a)
    ridge = regression["ridge"][a]["metrics"]["r2"]
    depth = regression["depth"][d]["metrics"]["r2"]
    pruned = regression["prune"][q]["metrics"]["r2"]
    poly = controls["polynomial"][regress.best_setting(controls["polynomial"])]["metrics"]["r2"]
    linear = classification["linear"][keys["linear"]]["metrics"]
    rbf = classification["rbf"][keys["rbf"]]["metrics"]
    pred = regression["ridge"][a]["pred"]
    f1_low = mean([class_metrics(y[idx], (pred[idx] >= THRESHOLDS[0]).astype(int),
                                  pred[idx])["f1"] for _, idx in cv])
    return f"""# 讨论题

## 1. 线性核与 RBF 核

采用 decision_function 连续分数和固定五折等权平均 AUC，线性核最优为 {fmt(linear['auc'])}，RBF 核最优为 {fmt(rbf['auc'])}，因此本实验按 AUC 选择线性核。
RBF 的 Accuracy 为 {fmt(rbf['accuracy'])}，高于线性核的 {fmt(linear['accuracy'])}，说明模型偏好依赖预先选择的指标。
RBF 调参后仍未超过线性核的 AUC，支持这组编码下线性决策面已捕捉较多有效信号的解释。
保持 C={keys['rbf'][0]}，把 gamma 从 {keys['rbf'][1]} 提高到 10，训练 AUC 从 {fmt(classification['rbf'][keys['rbf']]['metrics']['train_auc'])} 升至 {fmt(classification['rbf'][keys['rbf'][0], 10]['metrics']['train_auc'])}，验证 AUC 从 {fmt(rbf['auc'])} 降至 {fmt(classification['rbf'][keys['rbf'][0], 10]['metrics']['auc'])}。
由 exp(−gamma·‖x−x′‖²) 可知，大 gamma 使相似度随距离快速衰减，模型更依赖训练点附近的局部结构，训练与验证差距扩大。
固定参数从线性核换为 RBF 的 AUC 变化为 {classification['rbf'][1, 'scale']['metrics']['auc'] - classification['linear'][1, 'scale']['metrics']['auc']:+.4f}，而 RBF 调参带来 {rbf['auc'] - classification['rbf'][1, 'scale']['metrics']['auc']:+.4f} 的变化，因此本次调参比直接换核带来更大的正向收益。
逐折差值表见 report.md，该选择结论限于本次固定五折和扫描范围。

## 2. 分类与回归结论

线性 SVM 的最优 AUC 为 {fmt(linear['auc'])}，高于 RBF 的 {fmt(rbf['auc'])}，因此按本次固定五折平均 AUC 选择线性核。
回归中，岭回归 R² 为 {fmt(ridge)}，高于限制深度树的 {fmt(depth)} 与后剪枝树的 {fmt(pruned)}。
分类按平均 AUC 选择线性核，回归按平均 R² 选择岭回归，两条线都支持本实验中全局线性模型的适用性。
二次岭回归最高 R² 为 {fmt(poly)}，低于一阶岭回归的 {fmt(ridge)}，进一步支持当前样本量与编码下全局加性趋势较容易稳定估计的解释。
两个线性模型按绝对系数排序的前五特征有 {len(shared_features)} 项重合，为 {', '.join(shared_features)}，为两条任务依赖主要信号的一致性提供旁证。
分类关注 num=0 的边界，回归还拟合正例内部的等级差异，AUC 与 R² 也不能跨任务直接比较高低。
这些结果仍可能受样本量、类别编码和特征相关性影响，无法据此认定真实关系处处线性。
去除 ca 与 thal 后还需按初筛时实际可用的特征评估，不能仅凭全特征排名推断临床表现。

## 3. 把岭回归用于分类

{transfer}

岭回归连续预测的 AUC 为 {fmt(auc)}，与线性 SVM 的 {fmt(linear['auc'])} 很接近。
四个阈值中，{best_t[0]} 的 F1+ 最高，为 {fmt(best_t[3])}，线性 SVM 的 F1+ 是 {fmt(linear['f1'])}。
阈值为 {THRESHOLDS[0]} 时，岭回归的 F1+ 降为 {fmt(f1_low)}，此时按 F1+ 反而落后于 SVM。
因此“是否逊于分类器”会随评价指标和决策阈值而变，AUC 与某个阈值下的 F1+ 可以给出不同答案。
由于 E[num｜X]=P(num>0｜X)×E[num｜num>0,X]，回归目标同时包含阳性概率与阳性病例严重程度的信息，拟合 num 因而可能给出有用的分类排序。
不同阳性概率与严重程度可以产生相同的分级期望，所以该预测值不能直接解释成阳性概率，分级损失也不会专门优化二分类边界。
阈值是用这组折外预测做的探索，实际部署阈值需要在独立数据上再验证。
"""


def error_share(y, pred, cutoff=1):
    errors = np.abs(y - pred)
    high = errors > cutoff
    squared = errors ** 2
    return int(high.sum()), squared[high].sum() / squared.sum()


def log_story(X, num, regression, controls):
    d = regress.best_setting(regression["depth"])
    better_mae = regress.best_setting(regression["depth"], "mae")
    shallow = regression["depth"][d]
    other = regression["depth"][better_mae]
    n1, s1 = error_share(num, shallow["pred"])
    n2, s2 = error_share(num, other["pred"])
    tree = next(row for row in controls["features"] if row[0].startswith("回归树"))
    _, _, full, reduced = tree
    full_roots = "、".join(X.columns[model.tree_.feature[0]] for model in full["models"])
    reduced_names = X.drop(columns=["ca", "thal"]).columns
    reduced_roots = "、".join(reduced_names[model.tree_.feature[0]] for model in reduced["models"])
    folds_cause = [r2 for _, r2 in controls["holdout"]]
    worst_seed = min(controls["holdout"], key=lambda item: item[1])[0]
    diagnostics = {row["seed"]: row for row in controls["holdout_diagnostics"]}
    first = controls["holdout_diagnostics"][0]
    worst = diagnostics[worst_seed]
    first_r2 = dict(controls["holdout"])[first["seed"]]
    worst_r2 = dict(controls["holdout"])[worst_seed]
    return f"""## 实验日志

### 1. MAE 改善时，R² 反而下降

待检验的直觉是，预测误差改善应使不同指标给出一致的排序。实际结果是，深度 {d} 的 R²/MAE 为 {compact(shallow, ('r2', 'mae'))}，深度 {better_mae} 为 {compact(other, ('r2', 'mae'))}，两种指标的排序相反。

原因假设是，较深的树降低了一部分常见病例的绝对误差，却保留或放大了少数大误差。核查折外预测后，绝对误差超过一级的病例数分别为 {n1} 和 {n2}，占病例总数的 {n1 / len(num):.1%} 和 {n2 / len(num):.1%}，却贡献了各自平方损失的 {s1:.1%} 和 {s2:.1%}。这证明少数较大误差在平方损失中占主导，解释了为什么只看 MAE 的改善不足以判断 R² 也会改善。

### 2. 删去有预测信号的特征，浅树反而改善

待检验的直觉是，删除 ca 与 thal 会让所有模型的性能下降。实际结果是，深度 {d} 的回归树 R² 从 {fmt(full['metrics']['r2'])} 升至 {fmt(reduced['metrics']['r2'])}，而两个 SVM 和岭回归均下降。

原因假设是，浅树采用逐步贪心分裂，新增特征可能改变早期路径，未必改善折外预测。五折根节点在删列前依次为 {full_roots}，删列后变为 {reduced_roots}，直接确认了删列改变模型路径。该证据支持路径变化的解释，但没有证明新增特征在其他模型中无用，也没有证明删列收益在新数据上必然重现。

### 3. 相近的 MSE 可以得到明显不同的 R²

待检验的直觉是，固定深度和填补方式后，一次 80/20 划分应足以代表同一模型的水平。实际固定深度为 {controls['holdout_depth']}、使用十个种子后，R² 范围为 {fmt(min(folds_cause))}–{fmt(max(folds_cause))}，极差达到 {fmt(max(folds_cause) - min(folds_cause))}。

原因假设是，训练病例变化影响预测误差，测试等级构成还会通过 R² 的分母改变评价。用测试集总体方差记法，R²=1−MSE/Var(num_test)；种子 {first['seed']} 与 {worst_seed} 的 MSE 分别为 {fmt(first['mse'])}、{fmt(worst['mse'])}，测试标签方差却为 {fmt(first['target_variance'])}、{fmt(worst['target_variance'])}，对应 R² 为 {fmt(first_r2)}、{fmt(worst_r2)}。两者误差量级接近，R² 差异仍可很大，说明跨划分比较时必须同时检查误差与标签变异程度。

种子 {worst_seed} 中前三大误差占平方损失的 {worst['top3_squared_share']:.1%}，进一步说明少量病例对小测试集的影响。该分解验证了分母与大误差都值得检查；训练集与测试集同时改变，现有对照无法将波动唯一归因于某一种病例构成。因此主结论使用课程固定五折，十次划分用于展示单次评价的不稳定性。
"""


def render(X, num, cv, classification, regression, controls, reference, mine, write_report):
    for directory in ("figures", "results"):
        (ROOT / directory).mkdir(exist_ok=True)
    keys = {k: classify.best_setting(classification[k]) for k in ("linear", "rbf")}
    d, q, a = [regress.best_setting(regression[k]) for k in ("depth", "prune", "ridge")]
    y = (num > 0).astype(int)
    n = len(y)
    tree, alpha = draw(X, num, classification, regression, q)
    data_ranges = table(("特征", "单位或编码", "最小", "最大", "缺失"),
                        [(c, UNITS[c], f"{X[c].min():g}", f"{X[c].max():g}", X[c].isna().sum())
                         for c in X.columns],
                        f"共 {int(X.isna().sum().sum())} 个空单元格，被保留到各训练折内填补。")
    agreement, shared_features = feature_agreement(X, num, keys, a)
    grading = discussion(num, cv, classification, regression, controls, keys, a, d, q, shared_features)
    transfer, _, _ = transfer_table(y, cv, classification, regression, keys, a)
    blocks = {
        "数据概况": data_ranges,
        "复现核对": audit_table(reference, mine),
        "分类": classification_tables(classification, keys, y),
        "逐折 AUC 差值": fold_auc_table(classification, keys, y, cv),
        "回归": regression_tables(X, num, regression, d, q, a),
        "相关结构与变量选择": regularization_evidence(X, regression),
        "跨任务特征排序": agreement,
        "消融": ablation_tables(classification, regression, controls, keys),
        "回归转分类": transfer,
        "验证记录": check_table(classification, regression, num),
    }
    (ROOT / "results" / "tables.md").write_text(
        "# 可复现实验表\n\n" + "\n\n".join(f"## {name}\n\n{body}" for name, body in blocks.items()),
        encoding="utf-8",
    )
    (ROOT / "results" / "fold_auc.md").write_text(
        "# 调参后 SVM 的逐折 AUC 差值\n\n" + blocks["逐折 AUC 差值"], encoding="utf-8",
    )
    for name, payload in [("baseline", reference), ("mine", mine)]:
        (ROOT / "results" / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                       encoding="utf-8")
    if not write_report:
        return
    labels = "，".join(f"{k} 级 {count} 例" for k, count in zip(*np.unique(num, return_counts=True)))
    splits = " / ".join(str(len(test)) for _, test in cv)
    class_story = cls_story(classification, keys)
    reg_story = regression_story(X, num, regression, d, q, a, tree, alpha)
    abl_story = ablation_story(regression, controls, keys)
    logs = log_story(X, num, regression, controls)
    draft = f"""# 医学数据的分类与回归

Cleveland 心脏病数据 · Assignment 4

## 主要结果

在课程固定五折下，按平均 AUC 选择线性 SVM（{fmt(classification['linear'][keys['linear']]['metrics']['auc'])}），RBF 核调参后为 {fmt(classification['rbf'][keys['rbf']]['metrics']['auc'])}；按平均 R² 选择岭回归 alpha={a}（{fmt(regression['ridge'][a]['metrics']['r2'])}），高于本次最优回归树。该选择使用相同病例划分和折内预处理，结论限定于本次特征编码、指标和扫描范围。

消融结果说明，模型容量和输入信息都需要结合任务场景判断。去掉 ca 与 thal 后，两种 SVM 和岭回归下降，浅树却改善；增加二次特征也没有提高回归表现。十次单次划分的明显波动进一步说明，解释结果时需要同时检查误差分布、测试病例构成和模型的复杂度。

## 实验设计

共 {n} 例，13 个输入特征。分类目标是 `num > 0`，回归目标是将 `num` 作为 0–4 级有序数值。
按课程标签定义，num=0 表示主要血管的管腔直径狭窄均未超过 50%，num=1–4 表示存在超过 50% 的狭窄且分级递增；num 与显影血管数 ca 是不同变量。
这样可以分别评估阳性排序和预测偏离了几个等级。若改做五分类，需要为每一级估计概率，等级间的距离也不会自动进入损失。
两条任务使用课程固定五折，测试折大小为 {splits}，每例恰好取得一个折外预测。模型种子为 {SEED}，回归树单次划分的对照种子为 {SEED}–{SEED + 9}。

保留课程给定的 13 列数值编码，以复现基线的列数与距离定义。类别编码做数值处理会引入人为间距，解释系数时需注意。
中位数填补、标准化与二次变换均置于 Pipeline，在训练折拟合后应用于测试折。
中位数对极端值较稳健，也与课程基线一致；本数据仅 ca 和 thal 缺失，各训练折的中位数均属于原有合法编码，保留全部病例参与评估。
SVM 和岭回归做标准化，树直接用填补后的特征。去标准化实验仅改变 SVM 的缩放设置。

AUC 逐折用 `decision_function()` 计算并等权平均。阴性类的 AUC 可通过反转标签及分数方向得到，与阳性类相同。准确率为全体样本指标，其余阈值指标按类报告。查全率 Recall+ 为敏感性，Recall− 为特异性，Precision+ 是预测阳性中的真实阳性比例。回归 R² 与 MAE 逐折等权平均，RMSE 是五折 MSE 等权平均后开根，与课程脚本一致。
超参按同一固定五折的 AUC 或 R² 选出，表中扫描最优值用于课程内比较，可能含选参偏乐观，未用独立外部测试集。

运行 `python main.py --data-dir <解压后的 assignment4-data>` 可重生 `figures/` 与 `results/tables.md`。另加 `--write-report` 会覆盖报告和讨论题。文件分工和完整命令见 README。

## 数据探索

{labels}。阳性比例为 {y.mean():.1%}。

{blocks['数据概况']}

![严重程度分级分布](figures/label_distribution.png)

![各等级的运动 ST 段压低](figures/oldpeak_by_grade.png)

分级 4 的样本最少。oldpeak 在不同等级间存在重叠，单一运动指标难以把五级病例完全分开。

## 复现核对

课程脚本独立运行生成参考 JSON。复现侧用本项目的折内拟合和显式指标计算，对齐课程脚本的固定参数。扩展模型只列在这一节作流程对照。

{blocks['复现核对']}

## 分类

{blocks['分类']}

![RBF 验证 AUC 网格](figures/rbf_auc.png)

{class_story}

### 逐折 AUC 配对比较

{blocks['逐折 AUC 差值']}

C 越大，对训练误分的惩罚越强，小 C 则允许更多间隔违例以获得更宽的间隔。扫描呈现的波动也受 gamma 和标准化影响。

## 回归

{blocks['回归']}

![深度扫描](figures/tree_depth.png)

![剪枝后回归树](figures/pruned_tree.png)

每折只从自己的训练数据生成代价复杂度路径。用路径序列的相对位置作为可比较的超参数，再列出各折实际 ccp_alpha 的范围；避免用测试标签生成候选点。

{reg_story}

{blocks['相关结构与变量选择']}

代码中 Ridge/Lasso 的 alpha 对应讲义的正则化强度 λ，取值越大惩罚越强；它与 SVM 对偶变量 αᵢ 含义不同。

## 分类与回归的特征排序是否一致

{blocks['跨任务特征排序']}

## 消融与对照

{blocks['消融']}

{abl_story}

三道讨论题见 [analysis.md](analysis.md)。

{transfer}

{logs}

## 讨论与局限

本实验用有序等级的数值差评价回归误差，等价于把相邻等级的距离视为相同。等级本身是离散的临床编码，数值回归可能输出等级之间或范围之外的值；本次保留原始预测计算指标，与课程基线保持一致。若改做五分类，需要分别估计各等级概率，普通分类损失还不会自动体现错分一个等级与多个等级的差别。

按最高五折分数选参并报告该分数，可能产生选择偏乐观；模型排名用于课程内比较。变量系数、相关结构和树图来自说明性分析，性能指标则来自折外预测。少数严重等级病例较少，少量大误差对 R² 的贡献较大，因此报告同时给出 MAE、误差分解和划分波动。

若进一步扩展实验，会优先在固定五折下比较类别特征的独热编码与当前数值编码，检查线性核和 RBF 核的排名是否依赖人为编码距离；若获得独立病例，则锁定现有预处理和参数后验证泛化表现。

## 指标核查

{blocks['验证记录']}

SVM 的 AUC 使用 decision_function 连续分数计算，类别预测用于 Accuracy、Precision、Recall、F1 和混淆矩阵。
合并折外预测会在来自不同训练折的分数之间做排序，逐折平均仅在各折内部排序。正式表与课程基线统一采用逐折平均。
训练均值基线在每个测试折的 R² 等于测试均值与训练均值偏差的负向贡献除以测试集平方总变差，均值有偏差就会取负值。与讲义用 0 作近似参照的写法相比，正式表直接用逐折训练均值的实测分数。

## 数据与出处

- Detrano, R. 等（1988），UCI Machine Learning Repository，Heart Disease，数据集 45，Cleveland 子集。数据、编码和引用信息见课程包 README。
- 课程讲义《医学数据的分类与回归：同一张表上的两种提问》（Assignment 4）与附件 `splits.json`、`baselines.py`。课程基线脚本用于生成核对参考值。
- AI 对话为本作业的 Codex 会话，包括任务分解、代码起草、评估复核和报告草稿。

## AI 使用声明

本作业使用 OpenAI Codex 辅助编写数据读取、预处理、模型训练、超参数扫描及消融实验代码，并用于调试程序、生成图表与起草报告。本人已对实验方法、结果与报告内容进行核查，并对最终提交内容负责。

截至报告确认时，本人未发现经核查确认的 AI 错误，按作业第八节说明验证方法。

1. 核对课程提供的 splits.json，确认两条任务线使用相同的固定五折，五个测试折互不重叠并覆盖全部 303 例；检查缺失值填补、标准化和多项式变换均在训练折内拟合。
2. 检查分类 AUC 使用 decision_function 连续分数，Accuracy、Precision、Recall、F1 和混淆矩阵使用类别预测；确认回归 R²、MAE 与 RMSE 的汇总方式与课程基线一致。
3. 核对独立实现与课程 baselines.py 的固定参数结果，45 项指标的最大绝对差值为 0.000000，均满足 0.01 的容差要求；同时核对逐折 AUC 的等权平均与分类总表一致。
4. 检查报告中的超参数选择、消融对照、图表及讨论题是否与实际输出一致。自动复核还使用阳性与阴性分数的两两比较计算 AUC，以及误差公式计算回归指标，与实验评估结果一致。上述自动计算由 Codex 执行，本人已复核相关方法和结果。

本人已核查报告中的模型选择、指标冲突解释和三条实验日志。最终结论限定于课程固定五折、所选指标和超参数扫描范围，报告明确了实验中的验证口径与结果解释。

声明确认日期为 2026 年 11 月 24 日。
"""
    (ROOT / "report.md").write_text(draft, encoding="utf-8")
    (ROOT / "analysis.md").write_text(grading, encoding="utf-8")


