# 第9周 EDA完成与小结

## 角色1 交付物

### 1. 三类可视化图表
| 图表 | 路径 | 说明 |
|------|------|------|
| 类别分布 | `outputs/figures/class_dist.png` | 正常短信 4,516 条，垃圾短信 653 条，不平衡比例约 6.9:1 |
| 文本长度分布 | `outputs/figures/length_dist.png` | 垃圾短信长度明显更长，中位数和均值均高于正常短信 |
| 正常短信词云 | `outputs/figures/wordcloud_ham.png` | 高频词：ok、call、love、come、time、good 等日常用语 |
| 垃圾短信词云 | `outputs/figures/wordcloud_spam.png` | 高频词：free、call、text、win、mobile、urgent、claim 等营销/诈骗词汇 |

### 2. EDA 小结（可直接用于中期报告）

**（1）数据概况**
- 数据集包含 5,169 条有效短信，其中正常短信占 87.4%，垃圾短信占 12.6%。
- 类别不平衡显著，建模时需关注召回率和 F1 值，避免模型偏向多数类。

**（2）文本长度特征**
- 正常短信长度集中在 0~200 字符，呈右偏分布。
- 垃圾短信长度分布更广，长文本（>300 字符）比例更高。
- **结论**：文本长度可作为有效数值特征，辅助分类。

**（3）关键词特征**
- 正常短信词云以日常对话词为主，无明显商业意图。
- 垃圾短信词云凸显 `free`、`win`、`urgent`、`claim`、`prize` 等诱导性词汇。
- **结论**：两类短信在词频分布上差异显著，TF-IDF 特征将具备良好区分度。

### 3. 对后续建模的建议
- 采用分层抽样划分训练/测试集，保持类别比例一致。
- 评估指标以 F1-score 为主要参考，兼顾精确率和召回率。
- 文本向量化推荐使用 TF-IDF（unigram + bigram），可尝试 CountVectorizer 作为对比。
- 基线模型可选朴素贝叶斯和逻辑回归，后续引入 SVM 对比调优。

## 运行方式
```bash
cd src
python eda.py


## 第9周 基线模型训练（角色2）交付物

### 1. 模型性能对比 (`outputs/metrics.csv`)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Naive Bayes | 0.9652 | 0.9904 | 0.7321 | 0.8420 |
| Logistic Regression | 0.9782 | 0.9231 | 0.9080 | 0.9155 |

### 2. 实验发现与结论
* **模型表现**：逻辑回归（LR）在处理不平衡数据时（使用了 class_weight='balanced'）表现更均衡，F1-Score 显著高于朴素贝叶斯。
* **关键指标**：对于垃圾短信识别，**召回率（Recall）**至关重要。目前 LR 的召回率达到 90.8%，优于 NB 的 73.2%。
* **后续建议**：角色3在第10周调优时，可以尝试进一步优化 LR 的正则化参数，或考虑尝试更复杂的集成模型（如 Random Forest）。

### 3. 本周产出文件
* `src/train_baseline.py`: 完整基线训练脚本。
* `outputs/metrics.csv`: 性能对比表。
* `outputs/logs/nb.txt` & `lr.txt`: 详细训练日志。


## 角色3 交付物（第9周：混淆矩阵 + 评估指标 + 初步对比表）

### 1) Naive Bayes 混淆矩阵图
- 输出文件：`outputs/figures/cm_nb.png`
- 混淆矩阵（[[TN, FP], [FN, TP]]）：`[[902, 0], [23, 108]]`
  - TN=902：正常短信判正常
  - FP=0：正常短信误判垃圾（误报）
  - FN=23：垃圾短信漏判正常（漏报）
  - TP=108：垃圾短信判垃圾

### 2) Naive Bayes 评估指标（以 spam=1 为正类）
- Accuracy：**0.9777**
- Precision：**1.0000**
- Recall：**0.8244**
- F1：**0.9038**
- 评估日志（含更多细节）：`outputs/logs/nb_eval.txt`

### 3) 初步模型对比表（`outputs/metrics.csv`）
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.9574 | 1.0000 | 0.6641 | 0.7982 |
| Logistic Regression | 0.9797 | 0.9044 | 0.9389 | 0.9213 |

> 说明：`outputs/metrics.csv` 用于汇总不同模型在统一评估口径下的核心指标，便于后续调参与中期报告撰写。

### 运行方式
```bash
python src/evaluate.py





