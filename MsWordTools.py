import os
from docx import Document


def remove_toc(doc):
    """
    去除Word文档中的目录
    """
    paragraphs_to_remove = []
    for paragraph in doc.paragraphs:
        # 判断段落是否为目录项的启发式规则
        if "TOC" in paragraph.style.name:  # 样式名包含TOC
            paragraphs_to_remove.append(paragraph)
        elif paragraph.text.strip() and (
                paragraph.text[0].isdigit() or paragraph.text.startswith(("1.", "2.", "3."))):  # 以数字或序号开头
            paragraphs_to_remove.append(paragraph)

    # 删除识别出的目录段落
    for paragraph in paragraphs_to_remove:
        p = paragraph._element
        p.getparent().remove(p)


def remove_headers_footers(doc):
    """
    删除Word文档中的页眉和页脚内容
    """
    # 遍历所有节(section)
    for section in doc.sections:
        # 清空页眉
        header = section.header
        for paragraph in header.paragraphs:
            paragraph.text = ""  # 清空段落文本
        # 清空页脚
        footer = section.footer
        for paragraph in footer.paragraphs:
            paragraph.text = ""  # 清空段落文本


def process_docx_file(file_path):
    """
    处理单个docx文件：提取正文，去除目录和角色信息
    """
    # 加载Word文档
    doc = Document(file_path)

    # 1. 去除目录
    remove_toc(doc)
    remove_headers_footers(doc)

    # 2. 提取正文文本（连接所有段落）
    full_text = '\n'.join([para.text for para in doc.paragraphs])

    return full_text


def process_all_docx_files(directory_path):
    """
    批量处理指定目录下的所有docx文件
    """
    processed_contents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.docx') or filename.endswith('.doc') :
            file_path = os.path.join(directory_path, filename)
            try:
                content = process_docx_file(file_path)
                processed_contents[filename] = content
                print(f"成功处理: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    return processed_contents


def convert_doc_to_docx_batch(input_folder, output_folder):
    """
    批量将指定文件夹中的.doc文件转换为.docx文件

    Args:
        input_folder (str): 包含原始.doc文件的文件夹路径
        output_folder (str): 转换后的.docx文件的输出文件夹路径
    """
    import comtypes.client

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化Word应用
    word = comtypes.client.CreateObject('Word.Application')
    word.Visible = False  # 后台运行，不显示Word界面

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.doc'):
            # 构建完整的输入和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.docx")

            try:
                # 打开.doc文档
                doc = word.Documents.Open(input_path)
                # 另存为.docx格式 (FileFormat=16 表示.docx格式)
                doc.SaveAs(output_path, FileFormat=16)
                doc.Close()
                print(f"成功转换: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"转换失败 {filename}: {e}")

    # 关闭Word应用程序
    word.Quit()
