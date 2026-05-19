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
import os
from matplotlib import ticker

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
    plt.figure(figsize=(5, 5)) 
    plt.barh(words, counts, color='steelblue', height=0.6)
    plt.xlabel('Frequency', fontweight='bold')
    plt.title(title, fontweight='bold')
    
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) 
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

def plot_tfidf_words(tfidf_scores, feature_names, title, save_path, n=20):
    """绘制 TF-IDF 关键词条形图（水平）"""
    top_indices = tfidf_scores.argsort()[-n:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = tfidf_scores[top_indices]
    
    plt.figure(figsize=(5, 5)) 
    plt.barh(top_words[::-1], top_scores[::-1], color='#d95f02',height=0.6)
    plt.xlabel('TF-IDF Score', fontweight='bold')
    plt.title(title, fontweight='bold')
    
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) 
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

def main():
    df = pd.read_csv('data/processed/sms_clean.csv')
    # 基准目录配置：基于 src 文件夹向上返一级进入大文件夹
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 动态拼接所有输入输出路径，防止因运行路径不同而报错
    metrics_csv_path = os.path.join(base_dir, 'outputs', 'tables', 'metrics_comparison.csv')
    figures_dir = os.path.join(base_dir, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    # 统一设置 IEEE 字体与五号字 (10.5 磅)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimSun'],
        'font.size': 10.5, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'axes.linewidth': 0.8, 
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.weight': 'bold'
    })
    # 定义千分位格式化工具
    comma_fmt = ticker.StrMethodFormatter('{x:,.0f}')
    
    # ===== 图表1：类别分布 =====
    plt.figure(figsize=(5, 5))
    ax1 = sns.countplot(x='label', data=df, palette='deep')
    plt.title('Class Distribution (Ham vs Spam)', fontweight='bold')
    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    ax1.yaxis.set_major_formatter(comma_fmt)
    max_val = df['label'].value_counts().max()
    max_val = df['label'].value_counts().max()
    for i, count in enumerate(df['label'].value_counts().sort_index()):
        plt.text(i, count + (max_val * 0.01), f"{count:,}", ha='center', fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig('outputs/figures/class_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图1已保存: class_dist.png")

    # ===== 图表2：文本长度分布直方图 =====
    df['message_length'] = df['message'].astype(str).apply(len)
    plt.figure(figsize=(5, 5))
    ax2 = sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette='deep')
    plt.title('Message Length Distribution by Label', fontweight='bold')
    plt.xlabel('Message Length (characters)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlim(0, 500)
    ax2.yaxis.set_major_formatter(comma_fmt)
    legend = ax2.get_legend()
    if legend:
        legend.set_frame_on(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig('outputs/figures/length_dist.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("图2已保存: length_dist.png")

    # ===== 图表3：长度箱线图（新增） =====
    plt.figure(figsize=(5, 5))
    ax3 = sns.boxplot(x='label', y='message_length', data=df, palette='deep', width=0.5)
    plt.title('Message Length Boxplot by Label', fontweight='bold')
    plt.xlabel('Label', fontweight='bold')
    plt.ylabel('Message Length (characters)', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    ax3.yaxis.set_major_formatter(comma_fmt)
    sns.despine()
    plt.tight_layout()
    plt.savefig('outputs/figures/length_boxplot.png', dpi=300, bbox_inches='tight')
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
    
    # ===== 模型性能指标对比条形图 =====
    if os.path.exists(metrics_csv_path):
        df_metrics = pd.read_csv(metrics_csv_path)
        model_col = df_metrics.columns[0] 
        df_metrics_long = pd.melt(df_metrics, id_vars=[model_col], var_name='Metric', value_name='Value')

        plt.figure(figsize=(5, 5))
        ax_model = sns.barplot(
            data=df_metrics_long, 
            x='Metric', 
            y='Value', 
            hue=model_col, 
            palette='deep',
            edgecolor='black', 
            linewidth=0.6
        )
        plt.title('Model Performance Comparison', pad=15, fontweight='bold')
        plt.xlabel('Evaluation Metrics', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.ylim(0.7, 1.02)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        # 图例核心优化：移动至下方居中且不要边框
        plt.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), 
            ncol=3,            
            frameon=False,     
            columnspacing=1.0  
        )
        sns.despine()
        plt.savefig('outputs/figures/model_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("模型对比条形图已保存: model_comparison_chart.png")
    else:
        print("警告: 未检测到 metrics_comparison.csv，跳过模型对比图绘制。")

if __name__ == "__main__":
    main()
