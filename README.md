# 英文 LDA 主题模型分析 (English LDA Analysis)

本仓库包含一个用于英语文档的文本分析和潜在狄利克雷分配 (LDA) 主题建模项目。

## 项目结构 (Project Structure)

- `english_lda.py`: 执行自然语言处理 (NLP) 和 LDA 主题建模的主脚本。
- `data/`: 包含原始数据文件（例如 `data.xlsx`）。
- `result/`: LDA 结果的输出文件夹，包含生成的 HTML 可视化网页文件或提取的 Top 词汇。
- `stop_dic/`: 自定义和标准的停用词列表（包括 `dict.txt`，`stopwords.txt`）。
- `top_words_export/`: 用于提取每个发现主题的 Top 词汇的脚本。
- `word_trans/`: 用于转换文档格式的辅助脚本。

## 安装与环境依赖 (Setup and Requirements)

1. 安装 Python 3.7 或更高版本。
2. 通过 pip 安装所需的第三方依赖包:
   ```bash
   pip install -r requirements.txt
   ```
3. **NLTK 语料库依赖**:
   当您第一次运行代码时，可能需要下载 NLTK 的相关数据:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## 使用说明 (Usage)

1. 将目标数据放入 `data/` 文件夹中并命名为相应的名称（例如 `data.xlsx`）。**注意**：在分享或是克隆该项目后，请将脚本 `english_lda.py` 里面的绝对路径修改为相对路径（如用 `./data/data.xlsx` 代替 `E:\Project\...`），以确保代码可以在任何人的电脑上成功运行。
2. 运行 `english_lda.py` 开始执行文本的数据清洗、分词、词形还原和最终的 LDA 主题建模过程。
3. 在 `result/` 文件夹中查看输出的结果（如交互式的 LDA 可视化 html 文件）。
