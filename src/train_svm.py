#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色3：SVM 模型训练与调参
- 使用 TF‑IDF 特征（与基线完全一致）
- GridSearchCV 搜索最优超参数
- 输出 outputs/logs/svm.txt
- 更新 outputs/tables/metrics.csv
"""

import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)

# 统一的随机种子
SEED = 42
np.random.seed(SEED)

# 设置输出路径，无论从哪里运行都能找到
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "sms_clean.csv")
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
METRICS_FILE = os.path.join(BASE_DIR, "outputs", "tables", "metrics.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# 创建必要目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 配置日志：同时输出到控制台和文件
log_file = os.path.join(LOG_DIR, "svm.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("========== SVM 模型训练开始 ==========")
    
    # ================== 1. 加载清洗后的数据 ==================
    logger.info(f"加载数据：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 已知数据集包含四列：['label', 'message', 'label_encoded', 'clean_message']
    # 我们使用 'label' 作为标签，'clean_message' 作为清洗后的文本
    df = df[['label', 'clean_message']].copy()
    df.columns = ['label', 'message']      # 重命名为统一格式
    df = df.dropna(subset=['message'])
    logger.info(f"有效样本数：{len(df)}")
    
    # ================== 2. 标签编码 ==================
    # ham -> 0, spam -> 1
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    X_text = df['message'].values
    y = df['label_enc'].values
    
    # ================== 3. TF‑IDF 特征提取（与基线完全相同） ==================
    logger.info("正在提取 TF‑IDF 特征...")
    stop_words = 'english'   # 使用 sklearn 内置英文停用词，与 NLTK 停用词效果近似
                             # 若你们原脚本明确用了 NLTK，可替换为 from nltk.corpus import stopwords
                             # stop_words = set(stopwords.words('english'))
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        stop_words=stop_words,
        ngram_range=(1, 2)   # 可考虑加入 bigram，通常能略提升性能（你也可以去掉）
    )
    X = tfidf.fit_transform(X_text)
    logger.info(f"特征矩阵维度：{X.shape}")
    
    # ================== 4. 划分训练集/测试集（分层抽样，种子 42） ==================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logger.info(f"训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")
    
    # ================== 5. 定义 SVM 及超参数搜索空间 ==================
    # 使用 LinearSVC：对高维稀疏文本数据非常高效，类似 SVC(kernel='linear')
    estimator = LinearSVC(
        dual=False,            # 样本数 > 特征数时建议设为 False
        random_state=SEED,
        max_iter=2000,
        class_weight='balanced'  # 处理类别不平衡
    )
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    # 使用 F1 作为评估指标（侧重少数类 spam）
    grid = GridSearchCV(
        estimator,
        param_grid,
        scoring='f1',
        cv=5,                 # 5 折交叉验证
        n_jobs=-1,            # 使用所有 CPU 核
        verbose=2
    )
    
    logger.info("开始 GridSearchCV 调参...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    logger.info(f"最优参数：{grid.best_params_}")
    logger.info(f"最优交叉验证 F1：{grid.best_score_:.4f}")
    
    # ================== 6. 在测试集上评估最优模型 ==================
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info("\n========== 测试集性能 ==========")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1-score : {f1:.4f}")
    
    logger.info("\n分类报告：")
    logger.info(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    logger.info("\n混淆矩阵：")
    logger.info(f"          预测ham  预测spam")
    logger.info(f"实际ham     {cm[0,0]:5d}     {cm[0,1]:5d}")
    logger.info(f"实际spam    {cm[1,0]:5d}     {cm[1,1]:5d}")
    
    # ================== 7. 保存混淆矩阵图（可选） ==================
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham','spam'],
                yticklabels=['ham','spam'])
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(FIG_DIR, 'cm_svm.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵图已保存至：{cm_path}")
    
    # ================== 8. 将结果写入 metrics.csv ==================
    metrics_df = pd.DataFrame({
        'model': ['SVM'],
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1': [f1]
    })
    
    # 如果已存在 metrics.csv，就追加，否则新建
    if os.path.exists(METRICS_FILE):
        existing = pd.read_csv(METRICS_FILE)
        # 避免重复写入同一个模型（如果之前有 SVM 行，先删掉）
        existing = existing[existing['model'] != 'SVM']
        updated = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        updated = metrics_df
    
    updated.to_csv(METRICS_FILE, index=False)
    logger.info(f"性能指标已写入：{METRICS_FILE}")
    
    logger.info("========== SVM 训练全部完成 ==========")

    # 保存最终模型和TF-IDF向量化器（方便Demo或后续部署）
    import joblib
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "svm_model.pkl"))
    logger.info(f"模型已保存至：{os.path.join(model_dir, 'svm_model.pkl')}")
    # 如果也想保存向量化器（防止重新fit）
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer_final.pkl"))
    logger.info(f"向量化器已保存至：{os.path.join(model_dir, 'tfidf_vectorizer_final.pkl')}")
if __name__ == "__main__":
    main()