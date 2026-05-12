# -*- coding: utf-8 -*-
"""
数据处理与特征提取脚本 - 角色2
输入：../data/processed/sms_clean.csv (角色1提供)
输出：
  1. ../data/processed/sms_tfidf_ready.csv (去停用词后的文本，TF-IDF输入版本)
  2. ../data/features/tfidf_matrix.npz (TF-IDF稀疏矩阵)
  3. ../models/tfidf_vectorizer.pkl (TF-IDF向量化器模型)
  4. ../data/features/labels.csv (对齐后的标签)
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import joblib
import os

# 首次运行需要下载 NLTK 的停用词表数据
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def remove_stopwords(text):
    """按空格分词并去除英文停用词"""
    # 加载英文停用词表 (如: the, is, in, at, that...)
    stop_words = set(stopwords.words('english'))
    
    # 角色1已经做过小写和去标点，这里直接按空格切割即可
    words = str(text).split()
    
    # 过滤掉停用词
    filtered_words = [word for word in words if word not in stop_words]
    
    # 重新拼接成字符串，因为 TfidfVectorizer 需要输入字符串序列
    return ' '.join(filtered_words)

def main():
    # --- 关键修改：自动获取当前脚本所在文件夹的绝对路径 ---
    # 这会找到 d:/Python作业caokun/垃圾短信/A20_SMS_Spam_Detction/src/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 找到项目根目录 A20_SMS_Spam_Detction/
    project_root = os.path.dirname(base_dir)

    # 定义绝对路径，避免权限错误
    processed_dir = os.path.join(project_root, 'data', 'processed')
    features_dir = os.path.join(project_root, 'data', 'features')
    models_dir = os.path.join(project_root, 'models')
    
    # 1. 确保输出目录存在 (按照通用规范创建 features 和 models 文件夹)
    os.makedirs('../data/features', exist_ok=True)
    os.makedirs('../models', exist_ok=True)

    # 2. 读取角色1清洗后的数据
    input_path = '../data/processed/sms_clean.csv'
    print(f"[*] 正在读取角色1的数据: {input_path}")
    df = pd.read_csv(input_path)

    # 去除缺失值（以防万一）
    df = df.dropna(subset=['clean_message'])

    # 3. 去除停用词，生成 TF-IDF 输入版本
    print("[*] 正在进行分词与停用词去除...")
    df['tokens_text'] = df['clean_message'].apply(remove_stopwords)

    # 剔除处理后变为空字符串的行（有些短信可能全由停用词组成，如 "ok" 如果被当作停用词的话）
    df = df[df['tokens_text'].str.strip() != '']

    # 保存中间版本（方便检查和回溯）
    ready_csv_path = '../data/processed/sms_tfidf_ready.csv'
    df.to_csv(ready_csv_path, index=False, encoding='utf-8')
    print(f"[+] 去除停用词完成！TF-IDF 输入版本已保存至: {ready_csv_path}")

    # 4. 生成 TF-IDF 特征
    print("[*] 正在拟合并生成 TF-IDF 特征矩阵...")
    # max_features=5000: 保留最重要的5000个词，防止矩阵过于稀疏
    # min_df=2: 忽略只在1条短信中出现过的生僻词
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X_tfidf = vectorizer.fit_transform(df['tokens_text'])

    # 5. 存储交付物，供下一位负责模型训练的同学（角色3）使用
    matrix_path = '../data/features/tfidf_matrix.npz'
    model_path = '../models/tfidf_vectorizer.pkl'
    labels_path = '../data/features/labels.csv'

    # 保存稀疏矩阵
    scipy.sparse.save_npz(matrix_path, X_tfidf)
    # 保存向量化器（模型推理/预测新短信时必须用到同一个 vectorizer）
    joblib.dump(vectorizer, model_path)
    # 保存对齐后的标签
    df[['label_encoded']].to_csv(labels_path, index=False)

    print(f"[+] TF-IDF 特征矩阵已保存 (维度: {X_tfidf.shape}) -> {matrix_path}")
    print(f"[+] TF-IDF 向量化器已保存 -> {model_path}")
    print(f"[+] 对应的数值标签已保存 -> {labels_path}")
    print("\n🎉 角色2 的特征提取任务圆满完成！可以交接给角色3了。")

if __name__ == '__main__':
    main()