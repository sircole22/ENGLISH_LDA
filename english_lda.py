# ## 1.预处理

import numpy as np
import os
import pandas as pd
import re
import nltk
import string

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pandas.core.common import random_state


output_path = 'E:\\Project\\New_LDA\\LDA_English\\result'  #结果输出路径
file_path = 'E:\\Project\\New_LDA\\LDA_English\\data'  #数据存放路径
os.chdir(file_path)
data = pd.read_excel("E:\\Project\\New_LDA\\LDA_English\\data\\data.xlsx")  #文件名
os.chdir(output_path)
dic_file = "E:\\Project\\New_LDA\\LDA_English\\stop_dic\\dict.txt"  #专有名词路径
stop_file = "E:\\Project\\New_LDA\\LDA_English\\stop_dic\\stopwords.txt"  #停用词路径


# 过滤掉含有中文的行
def contains_chinese(text):
    if not isinstance(text, str):
        return False
    return bool(re.search('[\u4e00-\u9fff]', text))

data = data[~data['content'].apply(contains_chinese)].reset_index(drop=True)
data.to_excel("data_no_chinese.xlsx", index=False)

#英文分词
def textPrecessing(text):
    stop_list = []  # 停用词空列表
    custom_nouns = []  # 专有名词空列表

    # 打开停用词列表
    try:
        with open(stop_file, encoding='utf-8') as stopword_list:
            for line in stopword_list:
                stop_list.append(line.strip())
    except:
        print("error in stop_file")

    # 打开自定义名词列表
    try:
        with open(dic_file, encoding='utf-8') as dic_list:
            for line in dic_list:
                custom_nouns.append(line.strip())  # 更保险 strip 全空白字符
    except:
        print("error in dic_file")

    # 追加更多常见的副词、介词、代词、连词以防模型漏检
    extra_stops = {
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'like', 'through', 
        'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among', 
        'also', 'very', 'just', 'only', 'even', 'more', 'much', 'really', 'too', 'well', 'so', 'there', 'now', 
        'then', 'always', 'once', 'here', 'often', 'thus', 'however', 'therefore', 'already', 'almost',
        'and', 'or', 'but', 'if', 'because', 'although', 'unless', 'since', 'while', 'than',
        'it', 'this', 'that', 'these', 'those', 'which', 'who', 'what', 'some', 'any', 'such', 'other',
        'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must', 'shall', 'ought',
        'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'not', 'yes', 'no', 'make', 'made', 'take', 'taken', 'get', 'got', 'see', 'saw', 'use', 'using', 'used'
    }
    # 生成停用词集合（内置 + 自定义 + 补充过滤集）
    custom_stopwords = set(stopwords.words('english')) | set(stop_list) | extra_stops

    # 类型检查
    if not isinstance(text, str):
        return ""

    # 小写化
    text = text.lower()

    # 保护专有名词（先长后短，防止部分替换）
    custom_nouns_sorted = sorted(custom_nouns, key=len, reverse=True)
    for noun in custom_nouns_sorted:
        if " " in noun:
            text = text.replace(noun.lower(), noun.replace(" ", "_").lower())

    # 去除标点和数字（保留下划线）
    text = re.sub(r'[^\w\s_]', '', text)
    text = re.sub(r'\d+', '', text)

    # 分词
    words = nltk.word_tokenize(text)

    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()

    # 词性过滤
    tagged_words = nltk.pos_tag(words)
    allowed_tags = ['NN', 'NNS', 'NNP', 'NNPS']  # 名词类
    
    # 获取名词，过滤第一遍停用词，再词形还原（处理复数变单数）
    filtered = []
    for word, pos in tagged_words:
        if pos in allowed_tags and word not in custom_stopwords:
            # 使用 lemmatizer 将复数转为单数
            lemma_word = lemmatizer.lemmatize(word)
            filtered.append(lemma_word)

    # 同义词替换（可扩展）
    replacements = {
        "ai": "artificial intelligence",
         # 1. 缩写 → 完整表达（文档要求保留的特殊缩写还原）
        "hr": "human resources",
        "mba": "master of business administration",
        "phd": "doctor of philosophy",
        "vc": "venture capital",
        "lib": "library",
        "uni": "university",
        "dept": "department",
        "prof": "professor",
        
        # 2. 领域同义词归一化（文档隐含+主题相关）
        "traineeship": "internship",  # 实习类
        "employment": "job",          # 就业类（文档中job为高频词）
        "handbook": "guide",          # 指南类（guide词频更高）
        "competence": "skill",        # 技能类
        "professional": "career",     # 职业类
        "seminar": "workshop",        # 讲座/工作坊类
        "resource": "material",       # 资源类
        "opportunity": "chance",      # 机会类
        "organization": "institution",# 机构类（适配高校/图书馆场景）
        
        # 3. 单复数统一（文档中高频复数→单数，已覆盖核心词汇）
        "libraries": "library",
        "universities": "university",
        "internships": "internship",
        "jobs": "job",
        "skills": "skill",
        "guides": "guide",
        "resources": "resource",
        "opportunities": "opportunity",
        "organizations": "organization",
        "departments": "department",
        "professors": "professor",
        "students": "student",
        "programs": "program",
        "projects": "project",
        "services": "service",
        "documents": "document",
        
        # 4. 词性变体统一（文档中高频动词/形容词变体→原形）
        "using": "use",
        "used": "use",
        "uses": "use",
        "researching": "research",
        "researched": "research",
        "researches": "research",
        "creating": "create",
        "created": "create",
        "creates": "create",
        "providing": "provide",
        "provided": "provide",
        "provides": "provide",
        "including": "include",
        "included": "include",
        "includes": "include",
        "offering": "offer",
        "offered": "offer",
        "offers": "offer",
        "working": "work",
        "worked": "work",
        "works": "work",
        "studying": "study",
        "studied": "study",
        "studies": "study",
        "interviewing": "interview",
        "interviewed": "interview",
        "interviews": "interview",
        
        # 5. 文档中其他需统一的表达
        "development": "growth",      # 避免文档中无区分性的development（已加入停用词，此处可选）
        "practice": "experience",     # 避免无区分性的practice（已加入停用词，按需保留）
        "application": "submission",  # 申请类统一
        "analysis": "analysis",       # 不规则复数analyses已通过词形还原处理，此处无需重复
        "data": "data"                # 不可数名词，无需替换
    }
    
    # 执行同义词替换
    replaced_words = [replacements.get(w, w) for w in filtered]

    # 二次停用词过滤（因为词形还原或同义词替换后可能会生成停用词和笼统词）
    final_words = [w for w in replaced_words if w not in custom_stopwords and len(w) > 2]
    
    return " ".join(final_words)



data["content_cutted"]=data['content'].apply(textPrecessing)
data


data.to_excel("Eng_cutted.xlsx") #输出了分词后的结果

# ## 2.LDA分析



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


#输出概率前n高的单词的函数
def print_top_words(model,feature_names,n_top_words):
    tword=[]
    for topic_idx,topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        # 将下划线还原为空格，方便阅读专有名词
        topic_w = " ".join([feature_names[i].replace('_', ' ') for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword



# 设置特征提取参数
n_features = 2000
tf_vectorizer = CountVectorizer(
    strip_accents='unicode',  # 去除重音符号
    max_features=n_features,  # 最大特征数
    max_df=0.6,               # 忽略在超过60%文档中出现的词
    min_df=5,                # 忽略在少于5个文档中出现的词
    ngram_range=(1, 1)  # 改回一元词，专有名词已通过下划线连接，不再需要自动二元组合
)
tf = tf_vectorizer.fit_transform(data.content_cutted)
# 添加这两行以打印信息：
print(f"Vocabulary size: {len(tf_vectorizer.get_feature_names_out())}")
print(f"TF shape: {tf.shape}")

# ## 3.困惑度和对数似然得分

import matplotlib.pyplot as plt


plexs = []  # 困惑度（越低越好）
scores = []  # 对数似然（越高越好）
n_max_topics = 15

for i in range(1, n_max_topics):
    print(f"Training with {i} topics...")

    lda = LatentDirichletAllocation(
        n_components=i,
        max_iter=50,  # 减少迭代次数以免跑太慢
        learning_method='batch',  # 使用 batch 方法跑困惑度曲线更稳定
        random_state=42,
        n_jobs=-1
    )
    lda.fit(tf)

    log_likelihood = lda.score(tf)
    perplexity = lda.perplexity(tf)

    scores.append(log_likelihood)
    plexs.append(perplexity)

    print(f"Topics: {i}, Perplexity: {perplexity:.2f}, Log-Likelihood: {log_likelihood:.2f}")



# 困惑度曲线
plt.figure(figsize=(10, 6))
x = list(range(1, n_max_topics-1))
plt.plot(x, plexs[1:], marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity (Lower is better)")
plt.title("Perplexity vs Number of Topics")
plt.grid(True)
plt.savefig('improved_perplexity.png', dpi=300)
plt.show()


# 对数似然曲线
plt.figure(figsize=(10, 6))
plt.plot(x, scores[1:], marker='o', color='green')
plt.xlabel("Number of Topics")
plt.ylabel("Log-Likelihood (Higher is better)")
plt.title("Log-Likelihood vs Number of Topics")
plt.grid(True)
plt.savefig('improved_log_likelihood.png', dpi=300)
plt.show()

# ## 4.LDA训练
n_topics=6
lda=LatentDirichletAllocation(n_components=n_topics,max_iter=50,
                              learning_method='batch',
                              learning_offset=50,
#                            doc_topic_prior=0.1
#                              topic_word_prior=0.01,
                              random_state=0)
lda.fit(tf)

# ## 5.输出每个主题对应词语
n_top_words=10
tf_feature_names=tf_vectorizer.get_feature_names_out()#更新
topic_word=print_top_words(lda,tf_feature_names,n_top_words)

# --- 新增功能：输出每个主题包含的重点词到Excel ---
print("正在生成每个主题的重点词表格 (topic_keywords.xlsx)...")
topic_keywords_list = []
# 为了表格更直观，我们可以把每个Topic作为一行，列出它的Top N词汇
topic_keywords_dict = {}
for topic_idx, topic in enumerate(lda.components_):
    top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
    top_features = [tf_feature_names[i].replace('_', ' ') for i in top_features_ind]
    topic_keywords_dict[f"Topic {topic_idx}"] = top_features

df_topic_keywords = pd.DataFrame(topic_keywords_dict)
df_topic_keywords.to_excel("topic_keywords.xlsx", index=False)
print("主题重点词已成功保存至 topic_keywords.xlsx")
# -----------------------------------

# ## 6.输出每篇文章对应主题
import numpy as np


topics=lda.transform(tf)


topic=[]
for t in topics:
    topic.append(list(t).index(np.max(t)))
data['topic']=topic
data.to_excel("English.xlsx",index=False)


topic[0]#0 1 2 对应主题的概率

# --- 新增功能：输出文档-主题分布表格 ---
print("正在生成文档-主题分布表格 (document_topic_distribution.xlsx)...")
# 1) 获取文档主题分布概率
doc_topic_dist = topics

# 2) 构建DataFrame
columns = [f'Topic{i}' for i in range(n_topics)]
df_doc_topic = pd.DataFrame(doc_topic_dist, columns=columns)

# 3) 四舍五入保留4位小数
df_doc_topic = df_doc_topic.round(4)

# 4) 添加辅助列
df_doc_topic.insert(0, 'θᵢⱼ', '')
df_doc_topic.insert(0, '文档编号', range(1, len(df_doc_topic) + 1))

# 5) 定义加粗最大值的函数
def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

# 6) 应用样式并保存到 Excel
# 使用 openpyxl 引擎保存带有样式的 DataFrame，如果缺少 openpyxl，运行前请安装
styled_df = df_doc_topic.style.apply(highlight_max, subset=columns, axis=1)
styled_df.to_excel("document_topic_distribution.xlsx", index=False, engine='openpyxl')
print("表格已成功保存至 document_topic_distribution.xlsx")
# -----------------------------------

# ## 6.5 生成全部文档前50重要性词的词云图
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import numpy as np

    # 提取特征词
    feature_names_for_wc = tf_vectorizer.get_feature_names_out()
    
    # 计算全部文档的词频总和
    global_word_counts = np.array(tf.sum(axis=0)).flatten()
    
    # 提取前50个最重要的词及其在全部文档中的词频
    top_indices = global_word_counts.argsort()[-50:][::-1]
    
    # 构建词频字典
    freq_dict = {feature_names_for_wc[i].replace('_', ' '): global_word_counts[i] for i in top_indices}
    
    # 生成全体文档的词云
    plt.figure(figsize=(10, 8))
    wordcloud = WordCloud(width=800, height=800, background_color='white', colormap='viridis').generate_from_frequencies(freq_dict)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Top 50 Words Across All Documents", fontsize=20, pad=20)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('global_wordcloud.png', dpi=300)
    print("全局词云图已被保存为 global_wordcloud.png")
    plt.show()

except ImportError:
    print("wordcloud package not found. Please install it using: pip install wordcloud")


# ## 7. Visualization
import pyLDAvis
import numpy as np
import webbrowser

# Try sklearn wrapper first; if unavailable, build inputs and call pyLDAvis.prepare directly
try:
    import pyLDAvis as sklearn_pyl
    pic = sklearn_pyl.prepare(lda, tf, tf_vectorizer)
except Exception:
    # Fallback: compute topic-term distributions and document-topic distributions
    topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]
    doc_topic_dists = lda.transform(tf)
    vocab = list(tf_vectorizer.get_feature_names_out())
    # doc_lengths: number of tokens per document
    doc_lengths = np.asarray(tf.sum(axis=1)).ravel().tolist()
    # term_frequency: total frequency of each term in the corpus
    term_frequency = np.asarray(tf.sum(axis=0)).ravel().tolist()
    pic = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)

# Save and open
pyLDAvis.save_html(pic, f'lda_{n_topics}.html')
webbrowser.open(f'lda_{n_topics}.html')









