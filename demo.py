import gradio as gr
import joblib
import pandas as pd
import re
import os

# 加载模型和向量化器
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer_final.pkl"))

def clean_text(text):
    # 简单清洗（跟训练时保持一致）
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def predict_sms(message):
    # 清洗 -> 向量化 -> 预测
    cleaned = clean_text(message)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.decision_function(vec)[0]  # SVM 的决策分数

    label = "🚫 垃圾短信 (Spam)" if pred == 1 else "✅ 正常短信 (Ham)"
    confidence = abs(prob)  # 绝对值越大越确信
    return f"{label}\n（置信度分数: {confidence:.2f}）"

# 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_sms,
    inputs=gr.Textbox(lines=3, placeholder="输入短信内容..."),
    outputs=gr.Textbox(label="检测结果"),
    title="📱 垃圾短信智能检测",
    description="基于机器学习（SVM）的短信分类器，输入短信内容，自动判断是否为垃圾短信。",
    examples=[
        ["Congratulations! You've won a free trip to Hawaii. Call now to claim your prize!"],
        ["Hey, are we still meeting for lunch today?"],
        ["URGENT: Your account has been compromised. Click here to verify your details."],
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)   # share=True 会生成一个公网链接