# 第11周 扩展EDA与关键词分析

## 角色1 交付物

### 1. 新增可视化图表
| 图表 | 路径 | 说明 |
|------|------|------|
| 长度箱线图 | `outputs/figures/length_boxplot.png` | 对比 ham/spam 文本长度分布（中位数、四分位距、离群值） |
| TF-IDF 关键词（正常） | `outputs/figures/tfidf_ham.png` | TF-IDF 权重的 Top 20 词，反映正常短信的特有词汇 |
| TF-IDF 关键词（垃圾） | `outputs/figures/tfidf_spam.png` | TF-IDF 权重的 Top 20 词，反映垃圾短信的特有词汇 |

已有的图表（第9周已生成，本次一并确认）：
| 类别分布 | `outputs/figures/class_dist.png` |
| 文本长度分布 | `outputs/figures/length_dist.png` |
| 词云（正常） | `outputs/figures/wordcloud_ham.png` |
| 词云（垃圾） | `outputs/figures/wordcloud_spam.png` |
| 高频词条形图（正常） | `outputs/figures/highfreq_ham.png` |
| 高频词条形图（垃圾） | `outputs/figures/highfreq_spam.png` |

### 2. 新增数据文件
| 文件 | 路径 | 说明 |
|------|------|------|
| 高频词统计表 | `outputs/highfreq_words.csv` | 前30个高频词的 ham 与 spam 频次对比 |

### 3. 扩展EDA小结（可直接用于报告）

**长度分析**：
- 箱线图显示垃圾短信的中位数长度明显高于正常短信，且离群值（异常长文本）更多，进一步证实长度作为分类特征的有效性。

**TF-IDF 关键词对比**：
- 正常短信 TF-IDF 高分词：`ok`, `love`, `come`, `time`, `good`, `home` 等，集中体现非商业的日常交流。
- 垃圾短信 TF-IDF 高分词：`free`, `call`, `text`, `win`, `mobile`, `claim`, `urgent` 等，显著偏向营销诱导性与急迫性表达。
- 这种差异保证了 TF‑IDF 向量空间下两类样本的可分性，为后续 SVM、逻辑回归等线性模型提供了清晰的决策边界。

**综合结论**：
- 数据集质量高，特征区分度强，现有 EDA 结果为团队提供了充分的建模信心和可解释证据。

## 运行方式
```bash
cd src
python eda.py

## 角色 2 交付物 

### 1. 核心代码与实验文件
| 文件 | 路径 | 说明 |
| :--- | :--- | :--- |
| 对比实验脚本 | `src/train_compare.py` | 实现了 CountVectorizer 与 TfidfVectorizer 的性能对比实验 |
| 实验结果数据 | `outputs/tables/feature_comparison.csv` | 包含两种特征提取方式的准确率、精确率、召回率、F1值对比 |

### 2. 实验结论 (供角色 3 参考)
* **CountVectorizer**: 在本项目数据集中表现出更高的召回率 (0.948)，能更全面地拦截垃圾短信，但存在极少量误判。
* **TfidfVectorizer**: 精确率极高 (1.0)，但在召回率上表现较弱 (0.781)。
* **建议**: 给角色 3 的模型训练建议是，如果希望系统“宁可错杀不可放过”，可以优先考虑词频特征；如果追求“绝对不误判正常短信”，则 TF-IDF 更优。

### 3. 环境依赖项
* 确保已安装 `scikit-learn` 和 `pandas`。


# 角色3 交付：SVM 模型训练与评估

## 1. 新增/更新文件

| 文件 | 路径 | 说明 |
|------|------|------|
| SVM 训练脚本 | `src/train_svm.py` | 自动读取 `clean_message`，GridSearchCV 搜索最优 C |
| SVM 训练日志 | `outputs/logs/svm.txt` | 含参数搜索过程、分类报告、混淆矩阵 |
| SVM 混淆矩阵图 | `outputs/figures/cm_svm.png` | 直观展示误判分布 |
| 更新性能表 | `outputs/tables/metrics.csv` | 新增 SVM 行（当前仅含 SVM，待角色4合并） |

## 2. 实验设计与结论

- **特征方案**：使用与基线完全一致的 TF‑IDF 提取（`max_features=5000`, `min_df=2`, 英文停用词），保证可比性。
- **模型选择**：`LinearSVC`，设置 `class_weight='balanced'` 处理类别不平衡。
- **调参策略**：5 折交叉验证搜索 `C` ∈ {0.01, 0.1, 1, 10, 100}，以 **F1‑score** 作为优化目标。
- **最优参数**：`C=1`（CV F1=0.8947）。
- **测试集性能**：

| 指标 | 数值 |
|------|------|
| Accuracy | 0.9816 |
| Precision | 0.9516 |
| Recall | 0.9008 |
| F1‑Score | 0.9255 |

- **混淆矩阵**：正常短信误判为垃圾 6 条，垃圾短信漏过 13 条，综合表现优于 NB 基线，与 LR 持平且精确率更高。
- **结论**：SVM 在短文本分类任务上表现出色，特征区分度强，模型已具备实用水平。

## 3. 运行方式

```bash
cd src
python train_svm.py