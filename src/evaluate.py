# -*- coding: utf-8 -*-
"""
evaluate.py
第9周（角色3）：输出 Naive Bayes 混淆矩阵图 cm_nb.png + 评估指标，并写入 outputs/metrics.csv

默认使用：data/features/tfidf_matrix.npz + data/features/labels.csv
数据划分：与第8周一致（seed=42, stratify），确保可复现
模型：Multinomial Naive Bayes（适用于 TF-IDF 非负特征）
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PATH_X = os.path.join(PROJECT_ROOT, "data", "features", "tfidf_matrix.npz")
PATH_Y = os.path.join(PROJECT_ROOT, "data", "features", "labels.csv")

OUT_FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUT_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
OUT_METRICS = os.path.join(PROJECT_ROOT, "outputs", "metrics.csv")


def load_xy():
    # X: sparse matrix
    from scipy.sparse import load_npz
    X = load_npz(PATH_X)

    # y: labels.csv (期望有一列；若有表头则自动处理)
    y_df = pd.read_csv(PATH_Y)
    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0].values
    else:
        # 尝试常见列名
        for col in ["label_encoded", "label", "y"]:
            if col in y_df.columns:
                y = y_df[col].values
                break
        else:
            # 兜底：取第一列
            y = y_df.iloc[:, 0].values

    y = np.asarray(y).astype(int)
    return X, y


def eval_nb_and_save():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_LOG_DIR, exist_ok=True)

    X, y = load_xy()

    # 划分（与第8周一致：可复现 + 分层）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 训练 NB
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 指标（以 spam=1 为正类）
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # 画图保存
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Ham(0)", "Pred Spam(1)"],
        yticklabels=["True Ham(0)", "True Spam(1)"],
    )
    ax.set_title("Confusion Matrix - Naive Bayes (TF-IDF)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out_cm = os.path.join(OUT_FIG_DIR, "cm_nb.png")
    plt.savefig(out_cm, dpi=300)
    plt.close()

    # 写日志（可选但建议）
    report = classification_report(y_test, y_pred, digits=4)
    out_log = os.path.join(OUT_LOG_DIR, "nb_eval.txt")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("=== Naive Bayes Evaluation ===\n")
        f.write(f"seed={SEED}, test_size=0.2\n\n")
        f.write("Metrics (positive=spam=1):\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n\n")
        f.write("Confusion Matrix [[TN FP],[FN TP]]:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")

    # 更新 metrics.csv（若存在则更新 NB 行；不存在则新建）
    row = pd.DataFrame([{
        "Model": "Naive Bayes",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    }])

    if os.path.exists(OUT_METRICS):
        dfm = pd.read_csv(OUT_METRICS)
        if "Model" in dfm.columns and (dfm["Model"] == "Naive Bayes").any():
            dfm.loc[dfm["Model"] == "Naive Bayes", ["Accuracy", "Precision", "Recall", "F1-Score"]] = \
                row.loc[0, ["Accuracy", "Precision", "Recall", "F1-Score"]].values
        else:
            dfm = pd.concat([dfm, row], ignore_index=True)
        # 尽量保持常见列顺序
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        dfm = dfm[[c for c in cols if c in dfm.columns]]
        dfm.to_csv(OUT_METRICS, index=False)
    else:
        row.to_csv(OUT_METRICS, index=False)

    # 终端输出（便于你交作业截图）
    print("=== Naive Bayes Evaluation Done ===")
    print(f"Confusion matrix saved to: outputs/figures/cm_nb.png")
    print(f"Metrics saved/updated: outputs/metrics.csv")
    print(f"Log saved: outputs/logs/nb_eval.txt")
    print(f"Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print("Confusion Matrix [[TN FP],[FN TP]]:")
    print(cm)


if __name__ == "__main__":
    eval_nb_and_save()
