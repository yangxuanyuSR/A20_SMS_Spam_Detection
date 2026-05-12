#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色3 第12周：基于最终 SVM 模型的误判样本分析与输出
输出：outputs/error_cases.csv（text, true_label, pred_label, reason）
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re

SEED = 42
np.random.seed(SEED)

# 路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "sms_clean.csv")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "error_cases.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

def is_url(text):
    return bool(re.search(r'http[s]?://|www\.', text))

def is_phone(text):
    return bool(re.search(r'\b\d{10,}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text))

def has_money(text):
    return bool(re.search(r'[£$€¥]|win|prize|claim|offer|cash|free', text, re.I))

def generate_reason(text, true_label, pred_label):
    """
    根据文本特征自动生成简短误判原因
    """
    if true_label == 'ham' and pred_label == 'spam':   # 假阳性
        if is_url(text) or is_phone(text):
            return "正常短信被误判为垃圾：含有URL或长数字串"
        elif len(text.split()) < 4:
            return "正常短信被误判为垃圾：文本过短且含特殊符号"
        else:
            return "正常短信被误判为垃圾：可能包含营销类或异常词汇"
    elif true_label == 'spam' and pred_label == 'ham': # 假阴性
        if len(text.split()) < 5:
            return "垃圾短信被漏判：文本过短，缺乏典型垃圾特征词"
        elif re.search(r'\b(love|home|please|help|support)\b', text, re.I):
            return "垃圾短信被漏判：伪装成日常对话，包含温和词汇"
        elif has_money(text):
            return "垃圾短信被漏判：虽含诱导词，但整体结构易被忽略"
        else:
            return "垃圾短信被漏判：语言隐蔽性强，与正常短信混淆"
    return "其他"

def main():
    print("== 加载数据 ==")
    df = pd.read_csv(DATA_PATH)
    # 使用 clean_message 列作为文本，label 列作为标签
    df = df[['label', 'clean_message']].dropna()
    df.columns = ['label', 'message']

    # 编码
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    X_text = df['message'].values
    y = df['label_enc'].values

    print(f"总样本数：{len(df)}")

    # TF-IDF（与训练时完全一致）
    print("== 提取 TF-IDF 特征 ==")
    tfidf = TfidfVectorizer(
        max_features=5000, min_df=2, stop_words='english',
        ngram_range=(1, 2)  # 若你担心不一致可去掉，但保留一般不会错
    )
    X = tfidf.fit_transform(X_text)

    # 划分（保持与训练完全相同）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    df['index_orig'] = np.arange(len(df))
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df['label_enc']
    )
    X_train_text = df_train['message'].values
    y_train_enc = df_train['label_enc'].values
    X_test_text = df_test['message'].values
    y_test_enc = df_test['label_enc'].values

    # 向量化只用训练集fit
    tfidf = TfidfVectorizer(
        max_features=5000, min_df=2, stop_words='english',
        ngram_range=(1, 2)
    )
    X_train_vec = tfidf.fit_transform(X_train_text)
    X_test_vec = tfidf.transform(X_test_text)

    print("== 训练最优 SVM（C=1, class_weight='balanced'） ==")
    model = LinearSVC(
        C=1, class_weight='balanced', random_state=SEED, max_iter=2000, dual=False
    )
    model.fit(X_train_vec, y_train_enc)

    # 预测
    y_pred = model.predict(X_test_vec)
    df_test = df_test.copy()
    df_test['pred_label_enc'] = y_pred
    df_test['pred_label'] = df_test['pred_label_enc'].map({0: 'ham', 1: 'spam'})

    # 提取误判样本
    errors = df_test[df_test['label_enc'] != df_test['pred_label_enc']]
    print(f"误判样本数：{len(errors)}")

    # 生成 reason
    reasons = []
    for _, row in errors.iterrows():
        reasons.append(generate_reason(row['message'], row['label'], row['pred_label']))

    errors_out = pd.DataFrame({
        'text': errors['message'],
        'true_label': errors['label'],
        'pred_label': errors['pred_label'],
        'reason': reasons
    })

    # 保存
    errors_out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"误判案例已保存至：{OUT_PATH}")

    # 同时写入日志（可选）
    log_file = os.path.join(LOG_DIR, "error_analysis_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"SVM 模型误判分析日志\n")
        f.write(f"总测试样本：{len(df_test)}\n")
        f.write(f"误判总数：{len(errors)}\n")
        f.write(f"正常被误判为垃圾：{len(errors[errors['label']=='ham'])}\n")
        f.write(f"垃圾被误判为正常：{len(errors[errors['label']=='spam'])}\n\n")
        f.write(errors_out.to_string(index=False))
    print(f"日志保存至：{log_file}")

if __name__ == "__main__":
    main()
