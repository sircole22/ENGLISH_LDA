# -*- coding: gbk -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# 配置路径
base_dir = r"E:\Project\New_LDA\LDA_English"
data_path = os.path.join(base_dir, "result", "Eng_cutted.xlsx") # 读取之前已经分好词的数据
output_dir = os.path.join(base_dir, "top_words_export")

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

print("正在加载分词后的数据...")
try:
    data = pd.read_excel(data_path)
except FileNotFoundError:
    print(f"找不到文件：{data_path}。请确保 english_lda.py 已经成功运行并输出了 Eng_cutted.xlsx。")
    exit(1)

# 处理可能缺失的空值
data['content_cutted'] = data['content_cutted'].fillna("")

print("正在提取并统计词频...")
# 采用尽量和原LDA一致的参数来提取词汇特征（不限制 max_features 来计算全局真实的词频）
tf_vectorizer = CountVectorizer(
    strip_accents='unicode', 
    max_df=0.6,               
    min_df=5,                
    ngram_range=(1, 1)  
)

tf_matrix = tf_vectorizer.fit_transform(data['content_cutted'])

# 计算全部文档的特征词频
global_word_counts = np.array(tf_matrix.sum(axis=0)).flatten()
feature_names = tf_vectorizer.get_feature_names_out()

print("正在提取前 300 个重要词汇...")
# 取出频数最高的前300个索引
top_300_indices = global_word_counts.argsort()[-300:][::-1]

top_words = []
top_counts = []

for idx in top_300_indices:
    top_words.append(feature_names[idx].replace('_', ' '))
    top_counts.append(global_word_counts[idx])

# 写入 DataFrame
df_top300 = pd.DataFrame({
    "Rank": range(1, 301),
    "Word": top_words,
    "Frequency": top_counts
})

# 保存至同一文件夹下
output_file = os.path.join(output_dir, "top_300_words.xlsx")
df_top300.to_excel(output_file, index=False)

print(f"完成！所有文档中前 300 个重要性的词已成功导出至:\n{output_file}")
