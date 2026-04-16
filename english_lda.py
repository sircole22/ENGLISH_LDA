# -*- coding: utf-8 -*-
# 包含：预处理、TF‑IDF 特征筛选、LDA 训练、困惑度/对数似然曲线、pyLDAvis 可视化、主题词表、文档‑主题分布等

import numpy as np
import os
import pandas as pd
import re
import importlib
import nltk
import ssl
import warnings
from pathlib import Path
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import webbrowser

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------  NLTK 资源下载  ----------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

required_nltk_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for pkg in required_nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        print(f"正在下载 nltk 资源: {pkg}")
        nltk.download(pkg, quiet=False)

# ----------------------------  可选库兼容  ----------------------------
try:
    WordCloud = importlib.import_module('wordcloud').WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

try:
    import pyLDAvis
    PYLDAvis_AVAILABLE = True
except ImportError:
    pyLDAvis = None
    PYLDAvis_AVAILABLE = False

# ===================== 路径配置 =====================
output_path = 'E:\\Project\\New_LDA\\LDA_English\\result'
file_path = 'E:\\Project\\New_LDA\\LDA_English\\data'
os.makedirs(output_path, exist_ok=True)
os.chdir(file_path)
data = pd.read_excel("data.xlsx")
os.chdir(output_path)

dic_file = "E:\\Project\\New_LDA\\LDA_English\\stop_dic\\dict.txt"
stop_file = "E:\\Project\\New_LDA\\LDA_English\\stop_dic\\stopwords.txt"

# ===================== 自定义停用词表 =====================
CUSTOM_STOPWORDS = {
    'date', 'window', 'site', 'world', 'user', 'time', 'year', 'online',
    'artist', 'location', 'type', 'consumer', 'people', 'range', 'click',
    'level', 'way', 'building', 'of', 'login', 'index', 'campus', 'thousand',
    'york', 'variety', 'group', 'hundred', 'place', 'system', 'web', 'series',
    'individual', 'example', 'box', 'menu', 'reader', 'leader', 'hour', 'part',
    'day', 'california', 'point', 'space', 'home', 'region', 'tab', 'engineer',
    'street', 'team', 'floor', 'step', 'condition', 'washington', 'stage',
    'download', 'size', 'form',
    'use', 'used', 'using', 'via', 'etc', 'thing', 'things', 'based', 'within',
    'without', 'among', 'around', 'could', 'would', 'should', 'may', 'might', 'also',
    'com', 'www', 'http', 'https', 'html', 'php', 'org', 'edu', 'gov', 'home',
    'page', 'pages', 'site', 'sites', 'webpage', 'website', 'online', 'email',
    'link', 'links', 'source', 'sources', 'available', 'open', 'new', 'current',
    'public', 'private', 'button', 'search', 'menu', 'login', 'register', 'view',
    'info', 'information', 'request', 'question', 'questions', 'need', 'learn',
    'create', 'offer', 'select', 'choose', 'click', 'download', 'read', 'access'
}

# ===================== 同义词映射表 =====================
SYNONYM_REPLACEMENTS = {
    "ai": "artificial intelligence",
    "hr": "human resources",
    "mba": "master of business administration",
    "phd": "doctor of philosophy",
    "vc": "venture capital",
    "lib": "library",
    "uni": "university",
    "dept": "department",
    "prof": "professor",
    "publication": "article",
    "journal": "article",
    "newspaper": "report",
    "document": "report",
    "school": "university",
    "college": "university",
    "work": "job",
    "occupation": "job",
    "role": "work",
    "financials": "finance",
    "capital": "finance",
    "profile": "resume",
    "cover": "resume",
    "letter": "resume",
    "cv": "resume",
    "tech": "technology",
    "course": "training",
    "class": "training",
    "workshop": "training"
}

# 加载外部停用词 / 专有名词
try:
    with open(stop_file, encoding='utf-8') as f:
        external_stopwords = [line.strip().lower() for line in f if line.strip()]
except Exception as e:
    print(f"外部停用词文件加载失败: {e}")
    external_stopwords = []

try:
    with open(dic_file, encoding='utf-8') as f:
        custom_nouns = [line.strip().lower() for line in f if line.strip()]
except Exception as e:
    print(f"专有名词文件加载失败: {e}")
    custom_nouns = []

# ===================== 文本预处理函数 =====================
def textPrecessing(text):
    core_stopwords = set(stopwords.words('english')) | set(external_stopwords)
    final_stopwords = core_stopwords | CUSTOM_STOPWORDS

    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 保护专有名词（多词短语用下划线连接）
    custom_nouns_sorted = sorted(custom_nouns, key=len, reverse=True)
    for noun in custom_nouns_sorted:
        if " " in noun:
            text = text.replace(noun, noun.replace(" ", "_"))

    words = nltk.word_tokenize(text)
    if not words:
        return ""
    tagged_words = pos_tag(words)
    allowed_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']

    lemmatizer = WordNetLemmatizer()
    def get_lemma(word, pos):
        if pos.startswith('V'):
            return lemmatizer.lemmatize(word, pos='v')
        elif pos.startswith('J'):
            return lemmatizer.lemmatize(word, pos='a')
        else:
            return lemmatizer.lemmatize(word)

    filtered_words = []
    for word, pos in tagged_words:
        lemma_word = get_lemma(word, pos)
        if (lemma_word not in final_stopwords and len(lemma_word) > 2 and pos in allowed_tags
                and re.fullmatch(r'[a-z_]+', lemma_word)):
            filtered_words.append(lemma_word)

    replaced_words = [SYNONYM_REPLACEMENTS.get(w, w) for w in filtered_words]
    final_words = [w for w in replaced_words if w not in final_stopwords and len(w) > 2]
    return " ".join(final_words)

# ----------------------------  数据清洗与预处理  ----------------------------
def contains_chinese(text):
    if not isinstance(text, str):
        return False
    return bool(re.search('[\u4e00-\u9fff]', text))

data = data[~data['content'].apply(contains_chinese)].reset_index(drop=True)
data.to_excel("data_no_chinese.xlsx", index=False)

print("正在执行文本预处理，请稍候...")
data["content_cutted"] = data['content'].apply(textPrecessing)
data = data[data['content_cutted'].str.strip() != ""].reset_index(drop=True)
data.to_excel("Eng_cutted.xlsx", index=False)
print(f"✅ 预处理完成，有效文本数: {len(data)}")

# ===================== TF‑IDF 特征筛选 =====================
print("\n🔄 正在执行 TF‑IDF 特征提取与筛选...")
tf_vectorizer = CountVectorizer(max_features=3000, max_df=0.7, min_df=3, ngram_range=(1, 1))
tf_matrix = tf_vectorizer.fit_transform(data['content_cutted'])
tf_feature_names = tf_vectorizer.get_feature_names_out()

tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)

global_tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()
tfidf_threshold = np.percentile(global_tfidf_sum, 20)
high_value_mask = global_tfidf_sum >= tfidf_threshold

tf_filtered = tf_matrix[:, high_value_mask]
tf_filtered_feature_names = tf_feature_names[high_value_mask]
print(f"✅ TF‑IDF 过滤完成，原始特征数: {len(tf_feature_names)}，过滤后特征数: {len(tf_filtered_feature_names)}")

# 输出 TF‑IDF Top30 报告
top_global_idx = global_tfidf_sum.argsort()[-30:][::-1]
tfidf_global_report = pd.DataFrame({
    '词语': [tf_feature_names[i].replace('_', ' ') for i in top_global_idx],
    '全局TF-IDF总权重': [round(float(global_tfidf_sum[i]), 6) for i in top_global_idx],
    '词频': [int(tf_matrix.sum(axis=0)[0, i]) for i in top_global_idx]
})
tfidf_global_report.to_excel("TF-IDF_Global_Top30.xlsx", index=False)
print("✅ TF‑IDF 全局 Top30 词报告已保存")

# ===================== 训练最终 LDA 模型（4 主题） =====================
print("\n🚀 正在训练 LDA 模型 (n_topics=4)...")
best_lda = LatentDirichletAllocation(
    n_components=4,
    max_iter=1000,
    doc_topic_prior=0.1,
    topic_word_prior=0.005,
    learning_method='batch',
    learning_offset=50,
    random_state=42,
    n_jobs=1          # Windows 下避免多进程问题
)
best_lda.fit(tf_filtered)
print("✅ LDA 模型训练完成！")

# ===================== 模型评估曲线（困惑度、对数似然、一致性） =====================
print("\n📈 正在生成模型评估曲线（主题数 2～10）...")

def plot_lda_evaluation_curves(tf_mat, output_dir, topic_range=range(2, 11)):
    topic_numbers = []
    perplexities = []
    log_likelihoods = []

    for topic_num in topic_range:
        print(f"  正在评估主题数: {topic_num}")
        lda_eval = LatentDirichletAllocation(
            n_components=topic_num,
            max_iter=500,
            doc_topic_prior=0.05,
            topic_word_prior=0.005,
            learning_method='batch',
            random_state=42,
            n_jobs=1
        )
        lda_eval.fit(tf_mat)
        topic_numbers.append(topic_num)
        perplexities.append(lda_eval.perplexity(tf_mat))
        log_likelihoods.append(lda_eval.score(tf_mat))

    # 保存数据
    curve_df = pd.DataFrame({
        '主题数': topic_numbers,
        '困惑度': perplexities,
        '对数极大似然': log_likelihoods
    })
    curve_df.to_excel("LDA_模型评估曲线数据.xlsx", index=False)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(topic_numbers, perplexities, marker='o', linewidth=2, color='#1f77b4')
    plt.xlabel('主题数 (Number of Topics)', fontsize=12)
    plt.ylabel('困惑度 (Perplexity, 越低越好)', fontsize=12)
    plt.title('LDA模型困惑度曲线', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('LDA_困惑度曲线.png', dpi=300, bbox_inches='tight')
    plt.close()   # 不显示窗口，避免阻塞

    plt.figure(figsize=(10, 6))
    plt.plot(topic_numbers, log_likelihoods, marker='o', linewidth=2, color='#d62728')
    plt.xlabel('主题数 (Number of Topics)', fontsize=12)
    plt.ylabel('对数极大似然 (Log-Likelihood, 越高越好)', fontsize=12)
    plt.title('LDA模型对数极大似然曲线', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('LDA_对数极大似然曲线.png', dpi=300, bbox_inches='tight')
    plt.close()

    return curve_df

# 执行评估曲线绘制（这里使用原始 tf_filtered 矩阵，注意与训练时一致）
lda_curve_df = plot_lda_evaluation_curves(tf_filtered, output_path)

# ===================== 结果输出 =====================
# 8.1 主题关键词表
def print_and_save_top_words(model, feature_names, n_top_words=15):
    topic_words_dict = {}
    print("\n📝 各主题 Top15 关键词:")
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i].replace('_', ' ') for i in top_idx]
        print(f"Topic #{topic_idx}: {' | '.join(top_words)}")
        topic_words_dict[f"Topic {topic_idx}"] = top_words
    df_topic_words = pd.DataFrame(topic_words_dict).T
    df_topic_words.to_excel("LDA_主题关键词表.xlsx", header=[f"Top{i+1}" for i in range(n_top_words)])
    return topic_words_dict

topic_words = print_and_save_top_words(best_lda, tf_filtered_feature_names, 15)

# 8.2 文档‑主题分布表
print("\n📄 正在生成文档‑主题分布表...")
doc_topic_dist = best_lda.transform(tf_filtered)
data['dominant_topic'] = doc_topic_dist.argmax(axis=1)
data.to_excel("LDA_文档主题分类结果.xlsx", index=False)

df_doc_topic = pd.DataFrame(doc_topic_dist.round(4), columns=[f"Topic{i}" for i in range(4)])
df_doc_topic.insert(0, 'θᵢⱼ', '')
df_doc_topic.insert(0, '文档编号', range(1, len(df_doc_topic)+1))

def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

styled_doc_topic = df_doc_topic.style.apply(highlight_max, subset=[f"Topic{i}" for i in range(4)], axis=1)
styled_doc_topic.to_excel("LDA_文档-主题分布表.xlsx", index=False, engine='openpyxl')
print("✅ 文档‑主题分布表已保存")

# ===================== 可视化与补充输出 =====================
# 9.1 全局词云
if WORDCLOUD_AVAILABLE:
    print("\n☁️  正在生成全局词云...")
    global_word_counts = np.array(tf_filtered.sum(axis=0)).flatten()
    top_50_idx = global_word_counts.argsort()[-50:][::-1]
    freq_dict = {tf_filtered_feature_names[i].replace('_', ' '): global_word_counts[i] for i in top_50_idx}
    plt.figure(figsize=(12, 8))
    wc = WordCloud(width=1200, height=800, background_color='white', colormap='tab20', max_words=50,
                   font_path='C:/Windows/Fonts/Arial.ttf')
    wc.generate_from_frequencies(freq_dict)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("全局Top50关键词词云 (TF-IDF过滤后)", fontsize=18)
    plt.tight_layout()
    plt.savefig('LDA_全局词云.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 全局词云已保存")

# 9.2 pyLDAvis 交互式可视化
if PYLDAvis_AVAILABLE:
    print("\n🌐 正在生成 pyLDAvis 交互式可视化...")
    try:
        topic_term_dists = best_lda.components_ / best_lda.components_.sum(axis=1)[:, np.newaxis]
        doc_topic_dists = best_lda.transform(tf_filtered)
        doc_lengths = np.asarray(tf_filtered.sum(axis=1)).ravel()
        term_frequency = np.asarray(tf_filtered.sum(axis=0)).ravel()
        vocab = tf_filtered_feature_names.tolist()

        vis_data = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency,
                                    sort_topics=False, mds='mmds', R=15, lambda_step=0.01)
        output_html_path = os.path.join(output_path, f'LDA_{best_lda.n_components}主题_交互式可视化.html')
        pyLDAvis.save_html(vis_data, output_html_path)
        print(f"✅ 交互式可视化 HTML 已保存至：{output_html_path}")
        if os.path.exists(output_html_path):
            webbrowser.open(Path(output_html_path).resolve().as_uri())
    except Exception as e:
        print(f"❌ pyLDAvis 生成失败: {e}")

# 9.3 最终模型的主题一致性（单独输出）
print("\n📊 正在计算最终模型的主题一致性得分...")
try:
    texts = [doc.split() for doc in data['content_cutted']]
    dictionary = corpora.Dictionary(texts)
    top_n = 10
    topics_for_coherence = []
    for topic in best_lda.components_:
        top_ids = topic.argsort()[:-top_n - 1:-1]
        topics_for_coherence.append([tf_filtered_feature_names[i] for i in top_ids])
    coherence_model = CoherenceModel(topics=topics_for_coherence, texts=texts,
                                     dictionary=dictionary, coherence='c_v')
    overall_coherence = coherence_model.get_coherence()
    topic_coherences = coherence_model.get_coherence_per_topic()
    print(f"✅ 整体主题一致性得分: {overall_coherence:.4f}")
    for i, score in enumerate(topic_coherences):
        print(f"   Topic {i} 一致性得分: {score:.4f}")
    coherence_df = pd.DataFrame({
        '主题': [f"Topic{i}" for i in range(best_lda.n_components)],
        '一致性得分': [round(s, 4) for s in topic_coherences],
        '整体平均得分': round(overall_coherence, 4)
    })
    coherence_df.to_excel("LDA_主题一致性得分.xlsx", index=False)
except Exception as e:
    print(f"主题一致性评估失败: {e}")

print("\n🎉 所有分析流程执行完成！所有结果文件已保存到 result 目录")