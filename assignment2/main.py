"""Train/evaluate both probability classifiers with the course MNIST loader.

Run: python assignment2/main.py
Optional: --model all|naive|bayes_net --data-dir PATH --output-dir PATH --alpha 1.0
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from naive_bayes import NaiveBayesClassifier
from bayes_net import (BayesNetClassifier, FEATURE_NAMES, FEATURE_PARENTS,
                       FEATURE_STATES, GRID_POSITIONS, MODEL_VERSION, PREPROCESSING)


def find_data_dir():
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "assignment2-data" / "assignment2-data" / "mnist",
        root / "assignment2-data" / "mnist",
    ]
    for path in candidates:
        if (path / "mnist_loader.py").is_file():
            return path
    raise FileNotFoundError("Cannot find course data; pass --data-dir PATH")


def load_course_data(data_dir):
    # Reuse the course reader without modifying or copying its source.
    spec = importlib.util.spec_from_file_location(
        "course_mnist_loader", data_dir / "mnist_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_mnist(data_dir)


def verify_data(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    actual = {}
    for filename, expected in manifest["sha256"].items():
        digest = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Data checksum mismatch: {filename}")
        actual[filename] = digest
    return actual


def classification_metrics(y_true, y_pred):
    """Rows=true class, columns=predicted class; zero divisions become zero."""
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(10), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(10), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {"digit": c, "precision": float(precision[c]), "recall": float(recall[c]),
             "f1": float(f1[c]), "support": int(support[c])} for c in range(10)
        ],
    }


def font(size):
    for name in ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def confusion_figure(matrix, path, model_name="Bernoulli Naive Bayes"):
    canvas = Image.new("RGB", (900, 920), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((450, 30), f"{model_name} - MNIST test set", fill="#172c45",
              font=font(26), anchor="mt")
    draw.text((450, 72), "Counts; rows = true digit; columns = predicted digit",
              fill="#44556a", font=font(18), anchor="mt")
    x0, y0, cell = 120, 155, 70
    maximum = matrix.max()
    for r in range(10):
        draw.text((x0 - 28, y0 + r * cell + cell / 2), str(r),
                  fill="#172c45", font=font(20), anchor="mm")
        draw.text((x0 + r * cell + cell / 2, y0 - 28), str(r),
                  fill="#172c45", font=font(20), anchor="mm")
        for c in range(10):
            strength = float(matrix[r, c] / maximum)
            color = tuple(int(a + strength * (b - a))
                          for a, b in zip((242, 246, 251), (24, 81, 145)))
            x, y = x0 + c * cell, y0 + r * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline="white", width=2)
            draw.text((x + cell / 2, y + cell / 2), str(matrix[r, c]),
                      fill="white" if strength > 0.5 else "#172c45",
                      font=font(18), anchor="mm")
    canvas.save(path)


def error_figure(images, labels, predictions, selected, path, model_name="Bernoulli Naive Bayes"):
    canvas = Image.new("RGB", (1000, 780), "#f4f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((500, 26), f"{model_name} - misclassified examples", fill="#172c45",
              font=font(26), anchor="mt")
    draw.text((500, 66), "First error for each true digit; original 28 x 28 pixels enlarged",
              fill="#44556a", font=font(17), anchor="mt")
    for slot, idx in enumerate(selected):
        col, row = slot % 5, slot // 5
        x, y = 20 + col * 198, 114 + row * 322
        tile = Image.fromarray(images[idx]).resize((168, 168), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(tile, (x + 6, y))
        draw.text((x + 90, y + 184), f"True {labels[idx]}  /  Pred {predictions[idx]}",
                  fill="#172c45", font=font(18), anchor="mt")
        draw.text((x + 90, y + 215), f"Test index {idx}",
                  fill="#44556a", font=font(16), anchor="mt")
    canvas.save(path)


def write_outputs(output, metrics, images, labels, predictions, model_name="Bernoulli Naive Bayes"):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    selected = []
    for c in range(10):
        errors = np.flatnonzero((labels == c) & (predictions != c))
        if len(errors):
            selected.append(int(errors[0]))
    metrics["error_examples"] = [
        {"test_index": i, "true": int(labels[i]), "predicted": int(predictions[i])}
        for i in selected
    ]
    directed = sorted(
        [(int(matrix[a, b]), a, b) for a in range(10) for b in range(10) if a != b],
        reverse=True,
    )[:10]
    metrics["top_directed_confusions"] = [
        {"true": a, "predicted": b, "count": n} for n, a, b in directed
    ]
    pairs = sorted([(int(matrix[a, b] + matrix[b, a]), a, b)
                    for a in range(10) for b in range(a + 1, 10)], reverse=True)[:10]
    metrics["top_confused_pairs"] = [
        {"digits": [a, b], "count_both_directions": n} for n, a, b in pairs
    ]
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row])
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["digit", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_index", "true", "predicted"])
        writer.writerows(zip(range(len(labels)), labels, predictions))
    confusion_figure(matrix, figures / "confusion_matrix.png", model_name)
    error_figure(images, labels, predictions, selected, figures / "misclassified.png", model_name)


def network_figure(path):
    """Draw the fixed graph: light arrows from Y; dark arrows between features."""
    canvas = Image.new("RGB", (2100, 990), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((1050, 22), f"Hand-designed Bayesian network {MODEL_VERSION}", fill="#172c45",
              font=font(30), anchor="mt")

    def arrow(points, color, width=3):
        draw.line(points, fill=color, width=width)
        (x0, y0), (x1, y1) = points[-2:]
        if y0 == y1:
            direction = 1 if x1 > x0 else -1
            draw.polygon([(x1, y1), (x1-11*direction, y1-6), (x1-11*direction, y1+6)], fill=color)
        else:
            draw.polygon([(x1, y1), (x1-6, y1-11), (x1+6, y1-11)], fill=color)

    draw.rounded_rectangle((900, 85, 1200, 145), radius=14, fill="#173d68")
    draw.text((1050, 115), "Y : digit class (0-9)", fill="white", font=font(23), anchor="mm")
    draw.line([(1050, 145), (1050, 182), (110, 182), (1550, 182)], fill="#8cabc7", width=3)
    positions = {}
    for col in range(4):
        bus_x = 110 + col*480
        draw.line([(bus_x, 182), (bus_x, 425 if col == 3 else 765)], fill="#8cabc7", width=3)
    for name, grid_position in zip(FEATURE_NAMES, GRID_POSITIONS):
        # The global descriptor is displayed separately from the spatial grid.
        is_global = grid_position is None
        row, col = (1, 3) if is_global else grid_position
        cy = 255 + row*170
        x0, x1 = 170+col*480, 470+col*480
        if name == "Z41_Z42":
            x1 += 480
        bus_x = 110 + col*480
        positions[name] = ((x0+x1)/2, cy, (x1-x0)/2)
        arrow([(bus_x, cy-28), (x0, cy-28)], "#8cabc7")
        draw.rounded_rectangle(
            (x0, cy-46, x1, cy+46), radius=12,
            fill="#fff1dc" if is_global else "#eef4fa",
            outline="#a76716" if is_global else "#244f7b", width=2,
        )
        draw.text(((x0+x1)/2, cy-16), name, fill="#173d68",
                  font=font(25), anchor="mm")
        caption = ("Whole image; only parent: Y" if is_global else
                   ("Bottom row, columns 1 + 2 ink (merged)" if name == "Z41_Z42" else
                    f"Row {row+1}, column {col+1} ink"))
        draw.text(((x0+x1)/2, cy+20), caption, fill="#44556a",
                  font=font(17), anchor="mm")
    for child, parent in enumerate(FEATURE_PARENTS):
        if parent is not None:
            x0, y0, half0 = positions[FEATURE_NAMES[parent]]
            x1, y1, half1 = positions[FEATURE_NAMES[child]]
            if y0 == y1:
                direction = 1 if x1>x0 else -1
                arrow([(x0+half0*direction,y0),(x1-half1*direction,y1)], "#244f7b", width=5)
            else:
                arrow([(x0, y0+46), (x1, y1-46)], "#244f7b", width=5)
    region_count = sum(position is not None for position in GRID_POSITIONS)
    global_count = len(FEATURE_NAMES) - region_count
    feature_edges = sum(parent is not None for parent in FEATURE_PARENTS)
    parameters = 9 + sum(
        10 * (1 if parent is None else FEATURE_STATES[parent]) * (FEATURE_STATES[j]-1)
        for j, parent in enumerate(FEATURE_PARENTS)
    )
    draw.text((1050, 850),
              f"{region_count} region densities + {global_count} whole-image hole descriptor; 5 states each",
              fill="#172c45", font=font(24), anchor="mt")
    draw.text((1050, 892),
              f"Light: {len(FEATURE_NAMES)} class edges. Dark: {feature_edges} fixed adjacent-region edges. "
              f"{parameters} free parameters.",
              fill="#44556a", font=font(21), anchor="mt")
    draw.text((1050, 934),
              "Z22 is retained. Z41 and Z42 are merged. HoleLocation describes the whole image.",
              fill="#44556a", font=font(20), anchor="mt")
    canvas.save(path)


def write_network_details(output, model):
    details = {
        "model_version": MODEL_VERSION,
        "feature_order": FEATURE_NAMES,
        "extra_feature_parents": {
            name: (None if parent is None else FEATURE_NAMES[parent])
            for name, parent in zip(FEATURE_NAMES, FEATURE_PARENTS)
        },
        "all_features_also_have_class_parent": True,
        "node_count": 1 + len(FEATURE_NAMES),
        "class_to_feature_edges": len(FEATURE_NAMES),
        "feature_to_feature_edges": sum(parent is not None for parent in FEATURE_PARENTS),
        "total_edges": len(FEATURE_NAMES) + sum(parent is not None for parent in FEATURE_PARENTS),
        "whole_image_features": [
            name for name, position in zip(FEATURE_NAMES, GRID_POSITIONS) if position is None
        ],
        "preprocessing": PREPROCESSING, "feature_states": FEATURE_STATES,
        "grid_positions_zero_based": GRID_POSITIONS, "alpha": model.alpha,
        "class_prior": model.class_prior_.tolist(),
        "feature_histograms": dict(zip(FEATURE_NAMES, model.feature_histograms_.tolist())),
        "cpt_axis_order": "[class, state] or [class, feature_parent_state, state]",
        "cpt_counts": {name: value.tolist() for name, value in model.cpt_counts_.items()},
        "cpt": {name: value.tolist() for name, value in model.cpt_.items()},
        "free_parameters": 9 + sum(10*(1 if p is None else FEATURE_STATES[p])*(FEATURE_STATES[j]-1)
                                   for j,p in enumerate(FEATURE_PARENTS)),
        "sparse_parent_configurations": {
            name: {"total": int(np.prod(count.shape[:-1])),
                   "zero_samples": int(np.count_nonzero(count.sum(axis=-1) == 0)),
                   "fewer_than_10_samples": int(np.count_nonzero(count.sum(axis=-1) < 10))}
            for name, count in model.cpt_counts_.items()
        },
    }
    (output / "model_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    network_figure(output / "figures/network_structure.png")


def write_comparison(output, evaluated):
    summary = {name: {metric: result[metric] for metric in
                     ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}
               for name, result in evaluated.items()}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# 两个模型的完整测试集结果", "",
             f"{MODEL_VERSION} 使用 11 个区域墨量特征和 1 个全图闭环特征。"
             "DAG 由人工固定，保留 10 条相邻区域依赖边，共 2089 个自由参数；"
             "最终训练 60,000 张，评价标准测试集 10,000 张。", "",
             "候选仅使用训练集内划分筛选（seed 20261018），"
             "再在 seed 20261022、20261023 的训练内划分确认；这些划分相互重叠。"
             "标准测试集已在先前版本中使用，本轮方案固定后才重新评价，"
             "未用本轮测试结果调参；该结果不应视为全新的独立留出验证。", "",
             "| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in summary.items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.4%}" for v in result.values()) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--model", choices=["all", "naive", "bayes_net"], default="all")
    args = parser.parse_args()
    data_dir = (args.data_dir or find_data_dir()).resolve()
    hashes = verify_data(data_dir)
    train_images, y_train, test_images, y_test = load_course_data(data_dir)
    X_train, X_test = train_images.reshape(-1, 784), test_images.reshape(-1, 784)
    evaluated = {}
    for key, model_class, title in [
        ("naive", NaiveBayesClassifier, "Bernoulli Naive Bayes"),
        ("bayes_net", BayesNetClassifier, f"Bayesian Network {MODEL_VERSION}"),
    ]:
        if args.model not in ("all", key):
            continue
        print(f"Training {title} ...", flush=True)
        model = model_class(alpha=args.alpha)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - start
        metrics = classification_metrics(y_test, predictions)
        metrics["experiment"] = {
            "model": title, "alpha": model.alpha,
            "preprocessing": PREPROCESSING if key=="bayes_net" else {"binary_pixel_threshold_strictly_greater_than":128},
            "n_train": len(y_train), "n_test": len(y_test),
            "train_class_counts": model.class_count_.tolist(),
            "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "python": platform.python_version(), "numpy": np.__version__,
            "data_sha256": hashes,
            "selection": (
                f"{MODEL_VERSION}: candidate screening used the 50,000/10,000 "
                "training-only split with seed 20261018; the frozen candidate "
                "was confirmed on training-only splits with seeds 20261022 and 20261023. "
                "These splits overlap. Manual fixed graph; no automatic structure learning. "
                "Final refit on 60,000. The standard test set was already evaluated for "
                "earlier versions; this round reevaluates it only after model selection, "
                "without tuning on this round's test outcomes. It is not a fresh independent holdout."
                if key=="bayes_net" else
                "Fixed pixel threshold=128; alpha supplied by the run (default 1); "
                "no test-set tuning."
            ),
        }
        output = args.output_dir.resolve()
        if key == "bayes_net":
            output = output / "bayes_net"
        write_outputs(output, metrics, test_images, y_test, predictions, title)
        if key == "bayes_net":
            write_network_details(output, model)
        evaluated[title] = metrics
        for name in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            print(f"{name}: {metrics[name]:.6f}")
        print(f"Train: {len(y_train)}, test: {len(y_test)}")
        print(f"Fit: {fit_seconds:.3f}s, predict: {predict_seconds:.3f}s")
        print(f"Outputs: {output}", flush=True)
    if len(evaluated) == 2:
        write_comparison(args.output_dir.resolve(), evaluated)


if __name__ == "__main__":
    main()
