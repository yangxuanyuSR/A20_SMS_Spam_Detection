# -*- coding: utf-8 -*-
"""
EDA可视化脚本 - 角色1（第11周完整版）
输入：data/processed/sms_clean.csv
输出：
    outputs/figures/class_dist.png, length_dist.png, length_boxplot.png,
    wordcloud_ham.png, wordcloud_spam.png,
    highfreq_ham.png, highfreq_spam.png,
    tfidf_ham.png, tfidf_spam.png
    outputs/highfreq_words.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re

SEED = 42

# 停用词表
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
    'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now', 'll', 've', 're', 'm'
])

def tokenize(text):
    words = re.findall(r'[a-zA-Z]+', str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def plot_top_words(counter, title, save_path, n=20):
    top = counter.most_common(n)
    words, counts = zip(*top[::-1])
    plt.figure(figsize=(8, 6))
    plt.barh(words, counts, color='#2c7fb8')
    plt.xlabel('Frequency')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"已保存: {save_path}")

def plot_tfidf_words(tfidf_scores, feature_names, title, save_path, n=20):
    """绘制 TF-IDF 关键词条形图（水平）"""
    top_indices = tfidf_scores.argsort()[-n:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = tfidf_scores[top_indices]
    plt.figure(figsize=(8, 6))
    plt.barh(top_words[::-1], top_scores[::-1], color='#d95f02')
    plt.xlabel('TF-IDF Score')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"已保存: {save_path}")

def main():
    df = pd.read_csv('data/processed/sms_clean.csv')

    # ===== 图表1：类别分布 =====
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Class Distribution (Ham vs Spam)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    for i, count in enumerate(df['label'].value_counts()):
        plt.text(i, count + 30, str(count), ha='center')
    plt.tight_layout()
    plt.savefig('outputs/figures/class_dist.png', dpi=300)
    plt.show()
    print("图1已保存: class_dist.png")

    # ===== 图表2：文本长度分布直方图 =====
    df['message_length'] = df['message'].astype(str).apply(len)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette='viridis')
    plt.title('Message Length Distribution by Label')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Frequency')
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig('outputs/figures/length_dist.png', dpi=300)
    plt.show()
    print("图2已保存: length_dist.png")

    # ===== 图表3：长度箱线图（新增） =====
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='label', y='message_length', data=df, palette='viridis')
    plt.title('Message Length Boxplot by Label')
    plt.ylabel('Message Length (characters)')
    plt.tight_layout()
    plt.savefig('outputs/figures/length_boxplot.png', dpi=300)
    plt.show()
    print("图3已保存: length_boxplot.png")

    # ===== 词云 =====
    ham_text = ' '.join(df[df['label'] == 'ham']['message'].astype(str).str.lower())
    spam_text = ' '.join(df[df['label'] == 'spam']['message'].astype(str).str.lower())

    wc_ham = WordCloud(width=800, height=400, background_color='white',
                       stopwords=STOPWORDS, max_words=50, colormap='viridis')
    wc_ham.generate(ham_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title('Ham Messages Word Cloud')
    plt.tight_layout()
    plt.savefig('outputs/figures/wordcloud_ham.png', dpi=300)
    plt.show()
    print("图4已保存: wordcloud_ham.png")

    wc_spam = WordCloud(width=800, height=400, background_color='white',
                        stopwords=STOPWORDS, max_words=50, colormap='plasma')
    wc_spam.generate(spam_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title('Spam Messages Word Cloud')
    plt.tight_layout()
    plt.savefig('outputs/figures/wordcloud_spam.png', dpi=300)
    plt.show()
    print("图5已保存: wordcloud_spam.png")

    # ===== 高频词统计与条形图 =====
    ham_words = []
    spam_words = []
    for _, row in df.iterrows():
        if row['label'] == 'ham':
            ham_words.extend(tokenize(row['message']))
        else:
            spam_words.extend(tokenize(row['message']))

    ham_counter = Counter(ham_words)
    spam_counter = Counter(spam_words)

    plot_top_words(ham_counter, 'Top 20 Words in Ham Messages',
                   'outputs/figures/highfreq_ham.png')
    plot_top_words(spam_counter, 'Top 20 Words in Spam Messages',
                   'outputs/figures/highfreq_spam.png')

    # 保存高频词 CSV
    top_n = 30
    ham_df = pd.DataFrame(ham_counter.most_common(top_n), columns=['word', 'ham_freq'])
    spam_df = pd.DataFrame(spam_counter.most_common(top_n), columns=['word', 'spam_freq'])
    freq_table = pd.merge(ham_df, spam_df, on='word', how='outer').fillna(0)
    freq_table['ham_freq'] = freq_table['ham_freq'].astype(int)
    freq_table['spam_freq'] = freq_table['spam_freq'].astype(int)
    freq_table.to_csv('outputs/highfreq_words.csv', index=False)
    print("高频词统计表已保存: highfreq_words.csv")

    # ===== TF-IDF 关键词对比（新增） =====
    # 用清洗后的文本计算 TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(df['message'].astype(str))
    feature_names = tfidf.get_feature_names_out()

    # 分别取 ham 和 spam 样本的平均 TF-IDF 得分
    ham_idx = df[df['label'] == 'ham'].index
    spam_idx = df[df['label'] == 'spam'].index

    ham_tfidf_avg = np.array(tfidf_matrix[ham_idx].mean(axis=0)).flatten()
    spam_tfidf_avg = np.array(tfidf_matrix[spam_idx].mean(axis=0)).flatten()

    plot_tfidf_words(ham_tfidf_avg, feature_names,
                     'Top 20 TF-IDF Words in Ham Messages',
                     'outputs/figures/tfidf_ham.png')
    plot_tfidf_words(spam_tfidf_avg, feature_names,
                     'Top 20 TF-IDF Words in Spam Messages',
                     'outputs/figures/tfidf_spam.png')

    print("\n===== 所有第11周扩展EDA图表已生成 =====")

if __name__ == "__main__":
    main()
