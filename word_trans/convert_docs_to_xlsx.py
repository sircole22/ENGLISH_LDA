#!/usr/bin/env python3
# -*- coding: gbk -*-  # 把utf-8改为gbk
"""
批量读取 LDA_English\word_trans\need_trans 中的 doc/docx 文件，汇总到 data.xlsx
适配Windows系统，无需textract，依赖pywin32/python-docx
"""

import os
import sys
import re
import pandas as pd
import pythoncom
import win32com.client

try:
    from docx import Document
except ImportError:
    Document = None

def clean_excel_illegal_chars(text):
    """清理Excel不支持的非法字符，避免写入报错"""
    # 移除ASCII控制字符（\x00-\x1f），保留换行/制表符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 替换全角空格、零宽空格、非断行空格为普通空格
    text = text.replace('\u3000', ' ').replace('\u200b', '').replace('\xa0', ' ')
    # 截断超长文本（Excel单元格最大支持32767字符）
    if len(text) > 32767:
        text = text[:32760] + "..."
    return text

def read_docx_file(file_path):
    """读取docx文件（python-docx失败时自动用pywin32兜底）"""
    if Document is not None:
        try:
            doc = Document(file_path)
            content = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            return content
        except Exception as e:
            print(f"?? python-docx读取{file_path}失败，尝试pywin32兜底: {str(e)[:50]}")
    
    # 兜底：用pywin32读取docx
    return read_doc_file(file_path)

def read_doc_file(file_path):
    """读取doc文件（仅Windows，依赖pywin32）"""
    pythoncom.CoInitialize()
    word_app = None
    try:
        word_app = win32com.client.DispatchEx("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = 0  # 禁用所有弹窗
        doc = word_app.Documents.Open(
            FileName=os.path.abspath(file_path),
            ReadOnly=True,
            ConfirmConversions=False
        )
        content = doc.Content.Text.strip()
        doc.Close(SaveChanges=False)
        # 清理空行
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])
        return content
    except Exception as e:
        raise RuntimeError(f"读取doc文件失败: {str(e)}")
    finally:
        if word_app:
            word_app.Quit()
        pythoncom.CoUninitialize()

def main():
    # 1. 配置路径（核心：适配你的待处理文件夹）
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 待处理文档路径：LDA_English\word_trans\need_trans
    input_dir = os.path.join(script_dir, "need_trans")
    # 输出Excel路径：脚本同目录下的data.xlsx
    output_excel = os.path.join(script_dir, "data.xlsx")

    # 2. 检查待处理文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"? 错误：待处理文件夹不存在！路径: {input_dir}")
        print("   请确认 need_trans 文件夹在当前脚本目录下")
        sys.exit(1)

    # 3. 遍历所有doc/docx文件
    doc_data = []
    all_files = sorted(os.listdir(input_dir))
    if not all_files:
        print(f"?? 待处理文件夹为空！路径: {input_dir}")
        sys.exit(0)

    print(f"? 开始处理文件夹：{input_dir}")
    print(f"? 共发现 {len(all_files)} 个文件，筛选doc/docx...")

    for filename in all_files:
        file_path = os.path.join(input_dir, filename)
        # 跳过文件夹
        if not os.path.isfile(file_path):
            continue
        
        # 筛选doc/docx文件
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in [".doc", ".docx"]:
            print(f"?? 跳过非文档文件：{filename}")
            continue

        # 读取文件内容
        try:
            if file_ext == ".docx":
                content = read_docx_file(file_path)
            else:  # .doc
                content = read_doc_file(file_path)
            
            # 清理内容和非法字符
            content = clean_excel_illegal_chars(content)
            if not content:
                print(f"?? 空文档，跳过：{filename}")
                continue

            # 添加到数据列表
            doc_data.append({
                "filename": filename,  # 保留原文件名，便于追溯
                "content": content     # 文档内容（清理后）
            })
            print(f"? 成功读取：{filename}")

        except Exception as e:
            print(f"? 读取失败：{filename} -> {str(e)[:50]}")
            continue

    # 4. 写入Excel
    if not doc_data:
        print("? 无有效文档可写入Excel")
        sys.exit(0)

    # 转换为DataFrame并去重
    df = pd.DataFrame(doc_data)
    df = df.drop_duplicates(subset=["content"], keep="first")
    
    # 写入Excel（指定openpyxl引擎，兼容所有Excel版本）
    df.to_excel(output_excel, index=False, engine="openpyxl")

    # 输出结果
    print("\n? 处理完成！")
    print(f"? 共成功读取 {len(df)} 个有效文档")
    print(f"? Excel文件保存路径：{output_excel}")
    print(f"? 提示：可将该文件拷贝到 LDA_English/data/data.xlsx 用于后续LDA分析")

if __name__ == "__main__":
    # 检查依赖
    try:
        import openpyxl
    except ImportError:
        print("? 缺少openpyxl依赖，请执行：pip install openpyxl")
        sys.exit(1)
    
    try:
        main()
    except Exception as e:
        print(f"? 程序运行出错：{str(e)}")
        sys.exit(1)