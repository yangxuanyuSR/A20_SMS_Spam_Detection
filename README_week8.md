# 第8周 数据清洗与初步EDA

## 角色1交付物

### 1. 清洗脚本
- 文件：`src/clean_data.py`
- 功能：读取原始数据 `data/raw/sms_spam.csv`，执行去空、去重、文本基础清洗，输出清洗后数据。

### 2. 清洗后数据
- 文件：`data/processed/sms_clean.csv`
- 列说明：
  - `label`: 原始标签（ham/spam）
  - `message`: 原始短信内容
  - `label_encoded`: 数值标签（0/1）
  - `clean_message`: 清洗后文本（小写、去URL/特殊字符）

### 3. EDA可视化
- 脚本：`src/eda.py`
- 输出图表（位于 `outputs/figures/`）：
  - `class_dist.png`：类别分布柱状图（ham vs spam）
  - `length_dist.png`：文本长度分布对比直方图

## 清洗规则说明
1. **缺失值处理**：删除 `message` 为空的行。
2. **去重**：基于 `message` 内容完全匹配的去重。
3. **标签编码**：ham→0，spam→1。
4. **文本清洗步骤**：
   - 转换为小写
   - 移除URL（http://、www.）
   - 移除邮箱地址
   - 移除非字母字符
   - 压缩连续空格为单个空格
5. **空文本过滤**：清洗后若 `clean_message` 为空则删除该记录。

## 数据统计
- 原始记录数：5,574
- 去重后记录数：5,169（移除403条重复）
- 最终记录数：5,169（无缺失）
- 正常短信：4,516 条（87.4%）
- 垃圾短信：653 条（12.6%）

## 运行方式
```bash
cd src
python clean_data.py
python eda.py
```
---
## 角色2交付物 (分词与特征提取)

### 1. 特征提取脚本
- 文件：`src/extract_features.py`
- 功能：
  - 加载 NLTK 英文停用词表，去除清洗后文本中的无意义词汇。
  - 使用 `TfidfVectorizer` 构建 TF-IDF 模型。
  - 将文本转换为 5000 维的稀疏矩阵并持久化存储。

### 2. 特征数据
- `data/processed/sms_tfidf_ready.csv`: 进一步去除停用词后的纯净文本数据。
- `data/features/tfidf_matrix.npz`: 供模型训练使用的 TF-IDF 特征矩阵（Sparse Matrix）。
- `data/features/labels.csv`: 与特征矩阵严格对齐的标签文件。

### 3. 模型组件
- `models/tfidf_vectorizer.pkl`: 保存的 TF-IDF 向量化器，用于后续推理阶段的一致性处理。

## 处理规则补充
1. **分词策略**：基于空格进行英文分词。
2. **停用词过滤**：移除 NLTK 标准英文停用词（如 i, me, my, the, is 等）。
3. **向量化参数**：
   - `max_features=5000`: 选取权重最高的 5000 个词汇。
   - `min_df=2`: 过滤掉在数据集中仅出现过一次的生僻词，减少噪声。


##角色3交付物（数据划分脚本train/test）

###1.划分脚本
- 文件：`src/train_baseline.py`
- 功能
 - 从 `data\processed\sms_clean.csv` 读取数据（优先使用clean_message）
 - 使用固定随机种子 seed=42 
 - 使用 stratify=y 分层抽样，保证训练/测试集类别比例一致
 - 运行后在终端输出：样本量、训练/测试集大小、各自类别分布

###2.输出
1. 总样本数： 5166
2. 训练集： 4132 / 训练集： 1034
3. 训练集 spam： 522 测试集spam： 131

###运行方式
```bash
python .\src\clean_data.py        #清洗
python .\src\eda.py               #EDA
python .\src\extract_features.py  #特征提取
python .\src\train_baseline.py    #划分train/test
```
