import customtkinter as ctk
import joblib
import os
import re
import math
import numpy as np

# --- 资源加载 (根据队友路径) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 优先使用 train_svm.py 产出的最终模型
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_final.pkl")

# 设置 UI 风格
ctk.set_appearance_mode("System")  # 自动跟随系统深色/浅色模式
ctk.set_default_color_theme("blue")

class SpamDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 加载模型
        try:
            self.model = joblib.load(MODEL_PATH)
            self.tfidf = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            print(f"Error: {e}")
            self.model = None

        # 窗口配置
        self.title("SMS Intelligence Guard v1.0")
        self.geometry("700x550")

        # 布局配置
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. 标题栏
        self.header = ctk.CTkLabel(self, text="SMS Spam Detection System", 
                                   font=ctk.CTkFont(size=24, weight="bold"))
        self.header.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.subheader = ctk.CTkLabel(self, text="Powered by SVM & TF-IDF Algorithm", 
                                      font=ctk.CTkFont(size=13))
        self.subheader.grid(row=1, column=0, padx=20, pady=(0, 20))

        # 2. 输入区域
        self.input_text = ctk.CTkTextbox(self, width=600, height=150, corner_radius=15, 
                                         border_width=2, font=("Consolas", 14))
        self.input_text.grid(row=2, column=0, padx=30, pady=10)

        # 3. 示例按钮区 (English Examples)
        self.example_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.example_frame.grid(row=3, column=0, pady=10)
        
        examples = [
            ("Normal", "Are we still watching the movie tonight?"),
            ("Spam", "WINNER! Claim your £1000 prize now. Call 09061..."),
            ("Urgent", "URGENT! Your mobile number has been selected...")
        ]
        
        for i, (tag, text) in enumerate(examples):
            btn = ctk.CTkButton(self.example_frame, text=f"Ex: {tag}", width=100, 
                                height=28, corner_radius=20, fg_color="gray25",
                                command=lambda t=text: self.set_example(t))
            btn.grid(row=0, column=i, padx=5)

        # 4. 检测按钮
        self.detect_btn = ctk.CTkButton(self, text="Analyze Message", font=ctk.CTkFont(size=16, weight="bold"),
                                        height=45, width=200, corner_radius=25, 
                                        command=self.analyze)
        self.detect_btn.grid(row=4, column=0, pady=25)

        # 5. 结果展示卡片
        self.result_frame = ctk.CTkFrame(self, corner_radius=20, height=100, width=600)
        self.result_frame.grid(row=5, column=0, padx=30, pady=(0, 30), sticky="nsew")
        self.result_frame.grid_propagate(False)
        self.result_frame.grid_columnconfigure(0, weight=1)

        self.res_label = ctk.CTkLabel(self.result_frame, text="Waiting for Input...", 
                                      font=ctk.CTkFont(size=18, weight="bold"))
        self.res_label.grid(row=0, column=0, pady=(15, 5))

        self.conf_bar = ctk.CTkProgressBar(self.result_frame, width=400)
        self.conf_bar.grid(row=1, column=0, pady=5)
        self.conf_bar.set(0)

        self.conf_label = ctk.CTkLabel(self.result_frame, text="Confidence: 0%", font=ctk.CTkFont(size=12))
        self.conf_label.grid(row=2, column=0, pady=(0, 10))

    def set_example(self, text):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)

    def clean_text(self, text):
        # 优化点：严格还原队友 train_svm.py 中的 TF-IDF 前置逻辑
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()

        # 修改 analyze 方法中的代码
    def analyze(self):
        # 检查模型
        if self.model is None:
            self.res_label.configure(text="❌ 模型未加载", text_color="red")
            return
            
        msg = self.input_text.get("1.0", "end").strip()
        if not msg:
            self.res_label.configure(text="⚠️ 请输入短信内容", text_color="orange")
            return

        try:
            # 1. 预处理
            cleaned = self.clean_text(msg)
            print(f"📝 原始: {msg[:50]}...")
            print(f"🧹 清洗后: {cleaned[:50]}...")
            
            # 2. TF-IDF 转换
            vec = self.tfidf.transform([cleaned])
            print(f"📊 特征维度: {vec.shape}")
            
            # 3. 获取决策分数并计算概率
            score = self.model.decision_function(vec)[0]
            spam_prob = 1 / (1 + math.exp(-score))
            
            print(f"🎯 决策分数: {score:.4f}")
            print(f"🎯 垃圾概率: {spam_prob:.4f}")
            
            # 4. 判断并显示结果（这就是你要的判断逻辑！）
            if spam_prob > 0.5:
                result_text = "🚩 垃圾短信 (SPAM)"
                result_color = "#FF4B4B"
                confidence = spam_prob
                print(f"✅ 判定: 垃圾短信 (置信度: {confidence:.1%})")
            else:
                result_text = "✅ 正常短信 (HAM)"
                result_color = "#2ECC71"
                confidence = 1 - spam_prob
                print(f"✅ 判定: 正常短信 (置信度: {confidence:.1%})")
            
            # 5. 更新界面
            self.res_label.configure(text=result_text, text_color=result_color)
            self.conf_bar.set(confidence)
            self.conf_bar.configure(progress_color=result_color)
            self.conf_label.configure(text=f"Confidence: {confidence*100:.1f}%")

        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            self.res_label.configure(text="❌ 分析出错", text_color="red")


if __name__ == "__main__":
    app = SpamDetectorApp()
    app.mainloop()
