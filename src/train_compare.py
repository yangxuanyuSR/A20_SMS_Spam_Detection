import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. 路径自动配置 (已根据你的文件名修正)
# ==========================================
# 获取项目根目录 (A20_SMS_Spam_Detction)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 修正：将 sms_data_cleaned.csv 改为你的实际文件名 sms_clean.csv
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'sms_clean.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'tables')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_data():
    """加载数据"""
    if not os.path.exists(DATA_PATH):
        # 如果还是找不到，打印出尝试寻找的绝对路径，方便排查
        raise FileNotFoundError(f"还是找不到文件！请检查该路径是否存在：\n{DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # 自动识别列名：防止角色 1 把列名写成 'v1','v2' 或 'message'
    if 'text' in df.columns and 'label' in df.columns:
        return df['text'], df['label']
    elif 'message' in df.columns and 'label' in df.columns:
        return df['message'], df['label']
    else:
        # 如果列名不对，尝试取前两列（通常第一列是标签/文本）
        print("警告：未找到标准列名 'text' 和 'label'，将尝试按位置读取前两列。")
        return df.iloc[:, 1], df.iloc[:, 0] # 假设第一列标签，第二列文本

def run_experiment(vec_type, X_train, X_test, y_train, y_test):
    """特征提取与模型训练实验"""
    if vec_type == 'Count':
        vectorizer = CountVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
    
    X_train_vec = vectorizer.fit_transform(X_train.astype(str))
    X_test_vec = vectorizer.transform(X_test.astype(str))
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    return {
        'Method': vec_type,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, pos_label='spam'),
        'Recall': recall_score(y_test, y_pred, pos_label='spam'),
        'F1-Score': f1_score(y_test, y_pred, pos_label='spam')
    }

def main():
    print(f"--- 角色 2 实验：正在读取 {os.path.basename(DATA_PATH)} ---")
    
    try:
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        print("正在运行 Count 词频对比...")
        results.append(run_experiment('Count', X_train, X_test, y_train, y_test))
        
        print("正在运行 TF-IDF 权重对比...")
        results.append(run_experiment('TF-IDF', X_train, X_test, y_train, y_test))
        
        # 汇总结果
        results_df = pd.DataFrame(results)
        print("\n[第11周特征对比实验结果]")
        print(results_df.to_string(index=False))
        
        # 保存
        save_path = os.path.join(OUTPUT_DIR, 'feature_comparison.csv')
        results_df.to_csv(save_path, index=False)
        print(f"\n✅ 成功！结果已保存至: {save_path}")
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")

if __name__ == "__main__":
    main()