# -*- coding: utf-8 -*-
"""
train_baseline.py
第9周（角色2）：完成基线模型训练（NB/LR）并输出日志与指标
输入：data/processed/sms_clean.csv
输出：outputs/metrics.csv, outputs/logs/nb.txt, outputs/logs/lr.txt
"""

import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 项目根目录

def load_dataset() -> pd.DataFrame:
    """延续角色3的数据加载逻辑"""
    path = os.path.join(PROJECT_ROOT, "data", "processed", "sms_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}，请确保角色1已生成该文件。")
    
    df = pd.read_csv(path)
    # 标签映射
    if "label_encoded" not in df.columns:
        df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    
    # 文本列选择
    if "clean_message" in df.columns:
        df["text_for_model"] = df["clean_message"].astype(str)
    else:
        df["text_for_model"] = df["message"].astype(str)
    
    return df.dropna(subset=["text_for_model", "label_encoded"])

def save_log(model_name, report, train_time):
    """保存训练日志到 outputs/logs/"""
    log_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name.lower()}.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Time: {train_time:.4f}s\n")
        f.write("----------------------------------------\n")
        f.write(report)
    print(f"已生成日志: {log_path}")

def main():
    # 1. 加载数据
    df = load_dataset()
    X = df["text_for_model"].values
    y = df["label_encoded"].values

    # 2. 划分数据集 (延续 Seed=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. 特征工程：TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    metrics_list = []

    # --- 模型 A: Multinomial Naive Bayes ---
    start_nb = time.time()
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    time_nb = time.time() - start_nb
    
    nb_report = classification_report(y_test, y_pred_nb)
    save_log("NB", nb_report, time_nb)
    
    metrics_list.append({
        "Model": "Naive Bayes",
        "Accuracy": accuracy_score(y_test, y_pred_nb),
        "Precision": precision_score(y_test, y_pred_nb),
        "Recall": recall_score(y_test, y_pred_nb),
        "F1": f1_score(y_test, y_pred_nb)
    })

    # --- 模型 B: Logistic Regression ---
    start_lr = time.time()
    # 考虑到 6.9:1 的不平衡，设置 class_weight='balanced'
    lr = LogisticRegression(class_weight='balanced', random_state=SEED)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    time_lr = time.time() - start_lr
    
    lr_report = classification_report(y_test, y_pred_lr)
    save_log("LR", lr_report, time_lr)
    
    metrics_list.append({
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr)
    })

    # 4. 输出指标 CSV
    metrics_df = pd.DataFrame(metrics_list)
    output_path = os.path.join(PROJECT_ROOT, "outputs", "metrics.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"已生成指标文件: {output_path}")

if __name__ == "__main__":
    main()