# A20 垃圾短信/诈骗邮件智能检测项目

## 1. 项目简介
本项目针对 5,574 条中英文短信数据集，构建了一套从原始数据清洗、探索性数据分析（EDA）、特征工程对比到多模型迭代优化的完整检测链路。项目最终实现了对垃圾短信（Spam）高准确率、高精确率的自动识别。通过自然语言处理（NLP）技术与机器学习算法，构建了一套能够自动识别并拦截垃圾短信的智能检测系统。

## 2. 团队成员与分工
| 姓名 | 角色 | 核心任务 | 代表性交付物 |
| :--- | :--- | :--- | :--- |
| **方余文星** | 角色 1 | 数据清洗、数据预处理、误判样本初筛与 EDA 扩展分析 | `src/clean_data.py`, `outputs/error_cases.csv` |
| **曾欣语** | 角色 2 | 基线模型搭建、特征工程对比实验、最终模型量化评估与可视化汇总 | `src/train_baseline.py`, `outputs/figures/`, `README_week8~12.md` |
| **杨萱昱** | 角色 3 | SVM 模型训练与调优、最终误判案例分析、模型评估 | `src/train_svm.py`, `src/error_analysis.py`, `outputs/error_cases.csv` |
| **刘馨喻** | 角色 4 | 项目管理、技术报告撰写、PPT 制作与最终验收 | `demo_code.py`,`reports/final_report.pdf`, `reports/final_ppt.pptx` |

## 3. 项目演进过程
- **Week 8-9 (起步期)**：完成 5,574 条数据的清洗与去重；对比 Naive Bayes 与 Logistic Regression 模型，建立基线。
- **Week 11 (优化期)**：对比 CountVectorizer 与 TF-IDF 特征提取效果；引入 SVM 模型，完成超参数调优。
- **Week 12 (分析期)**：基于最终选定的 SVM 模型进行误判深度分析，抽取 19 条错误案例并分类，总结误判原因与改进方向。
- **Week 14-15 (交付期)**：集成所有结果，输出最终模型，完成性能汇总与环境依赖整理。

## 4. 核心技术链路
1. **数据清洗**：去重（移除去重后保留 5,166 条有效记录）、去 URL、去非英文字符，通过标签编码将 ham/spam 转换为 0/1。
2. **特征工程**：对比了词频（Count）与 TF-IDF 权重，最终采用 **5000 维 TF-IDF 特征**，使用英文停用词并限制最小文档频率为 2。
3. **模型演进**：
    - **V1 (Base)**: 朴素贝叶斯 (NB) —— 简单高效，但召回率较低。
    - **V2 (Mid)**: 逻辑回归 (LR) —— 引入`class_weight='balanced'`处理类别不平衡。
    - **V3 (Final)**: 支持向量机 (SVM) —— 使用线性核（LinearSVC）处理高维文本特征，通过 GridSearchCV 确定最优正则化系数 C，达到最优综合性能。

## 5. 技术栈
- **语言**：Python 3.x
- **核心库**：Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **特征工程**：TfidfVectorizer (max_features=5000, min_df=2)
- **模型**：LinearSVC (C=1, class_weight='balanced')

## 6. 实验结果对比 (核心量化指标)
| 模型 (Model) | 准确率 (Acc) | 精确率 (Precision) | 召回率 (Recall) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Naive Bayes | 96.52% | 0.99 | 0.73 | 0.84 |
| Logistic Regression | 97.82% | 0.92 | 0.91 | 0.92 |
| **SVM (Final)** | **98.36%** | **0.99** | **0.88** | **0.93** |

> **结论**：SVM 在保持极高准确率的同时，Spam 的精确率达到了 99%，这意味着系统极少产生“误拦截”，非常符合实际通信场景的需求。
> 最终模型（SVM）在测试集上取得了 **98.36%** 的准确率。相比基线 NB 模型，F1-Score 提升了约 12%。针对 Spam 类别的召回率达到了 **88%**，有效平衡了“拦截率”与“误伤率”。

*（注：以上指标以 `outputs/tables/metrics_comparison.csv` 为准）*

## 7. 误判深度分析 (Error Analysis)
基于最终选定的 **SVM 模型** 重新抽取测试集（1034 条）上的全部误判样本，得到 **19 条误判**（正常误判为垃圾 6 条，垃圾漏判 13 条），详见 `outputs/error_cases.csv`。分析发现：

- **正常短信被误判为垃圾（False Positive）**：主要因为文本中含有 URL 链接、长数字串，或被误认为营销用语。
- **垃圾短信被漏判（False Negative）**：常见原因是文本过短，伪装成日常口语（如含 “love”、“home”），缺乏典型敏感词。
- **改进方向**：后续可考虑引入文本长度、非词频特征（如数字/符号占比）、或尝试集成学习（如 Voting Classifier）来进一步减少边界误判。

## 8. 环境依赖与运行
### 环境安装
```bash
pip install -r requirements.txt

## 9. 项目结构
```text
├── data/
│   ├── raw/                # 原始 SMS Spam Collection 数据
│   └── processed/          # 经过清洗后的 CSV 文件 (sms_clean.csv, sms_tfidf_ready.csv)
├── outputs/
│   ├── figures/            # 混淆矩阵、性能对比图、词云等 (含 cm_svm.png, model_comparison_chart.png)
│   ├── logs/               # 训练日志 (nb.txt, lr.txt, svm.txt) 与误判分析日志 (error_analysis_log.txt)
│   └── tables/             # 最终模型对比表 (metrics_comparison.csv, error_cases.csv)
├── src/
│   ├── clean_data.py       # 文本清洗（去URL、特殊符号、小写化、去重）
│   ├── eda.py              # 探索性数据分析（类别分布、长度分布、词云、TF-IDF 关键词）
│   ├── extract_features.py # 停用词过滤与 TF-IDF 特征构造
│   ├── train_baseline.py   # NB/LR 基线模型训练与评估
│   ├── train_compare.py    # 特征提取效果对比（Count vs TF-IDF）
│   ├── train_svm.py        # 最终 SVM 模型训练与 GridSearch 调参
│   └── error_analysis.py   # 基于最终 SVM 模型的误判样本抽取与原因分类
├── README.md               # 项目总说明（本文件）
├── requirements.txt        # 环境依赖列表
└── README_week8~12.md      # 各阶段过程记录与交接文档
