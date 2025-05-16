import os
import re

def remove_chinese_and_target_string(file_path, target_string=""):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 
        content = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', content)
        # 
        content = content.replace(target_string, '')

        # 
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Skipped: {file_path}, due to error: {e}")

def traverse_and_clean(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            remove_chinese_and_target_string(file_path)

# 
folder_path = r"D:\Users\zhoujw\Desktop\SafeMVDrive"
traverse_and_clean(folder_path)
