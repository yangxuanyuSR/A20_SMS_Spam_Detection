# -*- coding: utf-8 -*-
"""
数据清洗脚本 - 角色1
输入：data/raw/sms_spam.csv
输出：data/processed/sms_clean.csv
"""

import pandas as pd
import re

# 统一随机种子
SEED = 42

def clean_text(text):
    """基础文本清洗：小写、去URL/邮箱、去特殊字符、压缩空格"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 读取原始数据（假设原始文件为制表符分隔，列名为 label, message）
    raw_path = 'data/raw/sms_spam.csv'
    df = pd.read_csv(raw_path, sep='\t', header=None, names=['label', 'message'])

    print(f"原始数据量: {len(df)}")
    print(f"类别分布:\n{df['label'].value_counts()}")

    # 去空
    df = df.dropna(subset=['message'])
    # 去重
    initial_len = len(df)
    df = df.drop_duplicates(subset=['message'])
    print(f"去重后数据量: {len(df)} (移除 {initial_len - len(df)} 条)")

    # 标签编码（保持原列名label为ham/spam，方便eda绘图）
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

    # 文本清洗生成 clean_message 列
    df['clean_message'] = df['message'].apply(clean_text)
    df = df[df['clean_message'] != '']

    print(f"清洗后最终数据量: {len(df)}")

    # 保存清洗后数据（统一列名：label, message, label_encoded, clean_message）
    processed_path = 'data/processed/sms_clean.csv'
    df.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"清洗完成，数据已保存至 {processed_path}")

if __name__ == "__main__":
    main()
