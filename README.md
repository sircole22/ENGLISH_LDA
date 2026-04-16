# 英文 LDA 主题模型分析 (English LDA Analysis)

本仓库提供一套用于英文文档的文本清洗、主题建模与可视化流程。脚本会对原始文本进行分词、词形还原、TF-IDF 过滤、LDA 训练，并输出主题词、文档主题分布、评估曲线和交互式可视化结果。

## 项目结构 (Project Structure)

- `english_lda.py`：主脚本，负责读取数据、预处理文本、训练 LDA 模型并导出结果。
- `data/`：放置输入数据文件，默认读取 `data.xlsx`。
- `result/`：输出目录，保存评估图、Excel 结果和 pyLDAvis HTML 页面。
- `stop_dic/`：自定义词典与停用词文件，包括 `dict.txt` 和 `stopwords.txt`。
- `top_words_export/`：主题 Top 词提取相关脚本。
- `word_trans/`：文档格式转换相关脚本。

## 环境要求 (Requirements)

1. Python 3.8 或更高版本。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载 NLTK 资源。首次运行时，脚本会自动尝试下载，但也可以提前手动执行：
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```
4. 如果你使用较新的 NLTK 版本，建议同时检查是否需要额外下载 `punkt_tab`。

## 数据准备 (Data Setup)

1. 将英文原始数据放入 `data/` 目录，并命名为 `data.xlsx`。
2. Excel 文件中需要包含 `content` 列，脚本会以该列作为文本输入。
3. 若要在其他电脑或仓库环境中运行，请将 `english_lda.py` 中的绝对路径改为相对路径，避免依赖本机目录结构。

## 使用说明 (Usage)

1. 确认 `data/data.xlsx` 已准备好。
2. 运行 `english_lda.py`。
3. 查看 `result/` 目录中的输出文件，包括：
   - `LDA_主题关键词表.xlsx`
   - `LDA_文档主题分类结果.xlsx`
   - `LDA_文档-主题分布表.xlsx`
   - `LDA_模型评估曲线数据.xlsx`
   - `LDA_4主题_交互式可视化.html` 或对应主题数的 HTML 文件

## 注意事项 (Notes)

- 脚本默认使用 Windows 路径和字体配置，若在非 Windows 环境运行，请同步调整 `english_lda.py` 中的路径与字体设置。
- `wordcloud` 和 `pyLDAvis` 相关输出属于增强功能，建议安装在完整环境中使用。
