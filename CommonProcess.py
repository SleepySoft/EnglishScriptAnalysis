import os
import re
import pandas as pd

from MsWordTools import process_all_docx_files
from EnglishAnalysisTools import remove_non_english, count_word_frequency, analyze_collocations


def remove_role_info(text):
    """
    去除正文中的角色信息（例如：Peppa: xxx）
    使用正则表达式匹配并移除角色名称及冒号
    """
    # 匹配模式：角色名（可能包含空格和特殊字符）后跟冒号和可选空格
    pattern = r'^[A-Za-z\s]+:\s*'
    # 逐行处理，移除匹配的角色信息
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(pattern, '', line)
        cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)


def common_process_eng_docs_to_pure_text(directory: str) -> str:
    results = process_all_docx_files(directory)
    file_path = os.path.join(directory, 'pure_text.txt')

    with open(file_path, 'wt') as f:
        for filename, content in results.items():
            clean_text = remove_non_english(content)
            clean_text = remove_role_info(clean_text)
            f.write(clean_text)
    return file_path


def load_pure_text(directory: str) -> str:
    with open(os.path.join(directory, 'pure_text.txt'), 'rt') as f:
        return f.read()


def save_sentences_and_word_frequency(
        sentences, frequency, directory: str,
        file_name: str = 'sentences_and_word_frequency.xlsx'):
    file_path = os.path.join(directory, file_name)

    # 创建句子列表的DataFrame
    df_sentences = pd.DataFrame(sentences, columns=['Sentences'])

    # 创建词频统计的DataFrame
    df_frequency = pd.DataFrame(frequency.items(), columns=['Word', 'Frequency'])
    # 按词频从高到低排序
    df_frequency.sort_values(by='Frequency', ascending=False, inplace=True)

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_sentences.to_excel(writer, sheet_name='Sentences', index=False)
        df_frequency.to_excel(writer, sheet_name='Word Frequency', index=False)

    print(f"分析结果已成功导出到 '{file_path}'")


def dump_collocations(collocations):
    main_seperator = '=' * 50
    sub_seperator = '-' * 35

    print(f"\n{main_seperator} 常见表达方式归类 {main_seperator}")
    for pattern_type, phrases in collocations.items():
        if phrases:  # 只显示有结果的类型
            print(f"{sub_seperator} {pattern_type} {sub_seperator}")
            for phrase, freq in phrases:
                print(f"{phrase}: {freq}")


def save_collocations(collocations, directory: str, file_name: str = 'collocations.xlsx'):
    file_path = os.path.join(directory, file_name)

    data = []
    for collocation_type, phrases_list in collocations.items():
        for phrase, frequency in phrases_list:
            data.append({
                "搭配模式": collocation_type,
                "搭配短语": phrase,
                "出现频率": frequency
            })

    df_collocations = pd.DataFrame(data)
    df_collocations.to_excel(file_path, index=False)

    print(f"搭配分析结果已保存到 '{file_path}'")


def common_flow(directory: str):
    # If pure_text.txt has been generated, we don't have to parse docx again.

    print('*' * 80)
    print('Loading word documents...')
    file_path = common_process_eng_docs_to_pure_text(directory)
    print(f'Pure text is saved to: {file_path}')

    print('*' * 80)
    print('Loading word documents...')
    full_text = load_pure_text(directory)
    print(f'Load finished. Text length: {len(full_text)}')

    print('*' * 80)
    print('Start counting word frequency...')
    sentences, frequency = count_word_frequency(full_text)

    print('*' * 80)
    print('Saving word frequency finished.')
    save_sentences_and_word_frequency(sentences, frequency, directory)

    print('*' * 80)
    print('Analyzing text collocations...')
    collocations = analyze_collocations(full_text)

    dump_collocations(collocations)

    print('*' * 80)
    print('Saving text collocations...')
    save_collocations(collocations, directory)

