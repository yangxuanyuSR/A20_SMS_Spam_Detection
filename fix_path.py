import shutil
import os

files_to_fix = ['src/eda.py', 'src/clean_data.py']

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        print(f"文件没找到，跳过：{file_path}")
        continue
    
    # 备份原文件，好习惯
    shutil.copy2(file_path, file_path + '.bak')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 执行路径替换
    content = content.replace('../data/processed/sms_clean.csv', 'data/processed/sms_clean.csv')
    content = content.replace('../outputs/figures/', 'outputs/figures/')
    content = content.replace('../data/raw/sms_spam.csv', 'data/raw/sms_spam.csv')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"搞定！文件已修复：{file_path}")

print("所有能修复的文件都处理完了。")