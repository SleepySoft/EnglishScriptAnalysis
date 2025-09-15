import re
import nltk
import string
import traceback
import pandas as pd
from collections import Counter
from typing import Tuple, List, Dict
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize


def check_download_nlp_data():
    # 确保已下载所需的NLTK数据
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')  # 用于词性标注，帮助词形还原
    nltk.download('averaged_perceptron_tagger_eng')
check_download_nlp_data()       # Execute immediately when module loading


def remove_non_english(text, keep_space=True, keep_number=False):
    """
    移除字符串中的所有非英文字符。

    Args:
        text (str): 待处理的原始字符串。
        keep_space (bool): 是否保留空格字符。默认为 True。
        keep_number (bool): 是否保留阿拉伯数字。默认为 False。

    Returns:
        str: 只包含英文字母（以及可选空格和数字）的清洗后字符串。

    Raises:
        TypeError: 如果输入 `text` 不是字符串类型。
    """

    # 参数检查
    if not isinstance(text, str):
        raise TypeError("输入参数 text 必须是字符串类型 (str)")

    # 根据参数构建正则表达式模式
    # 基础模式：匹配所有英文字母（大小写）
    base_pattern = r'a-zA-Z'

    # 可选：在模式中添加空格
    if keep_space:
        base_pattern += r' '

    # 可选：在模式中添加数字
    if keep_number:
        base_pattern += r'0-9'

    # 创建正则表达式，匹配所有不在指定集合中的字符
    # 模式 [^...] 表示匹配任何不在方括号内的字符
    pattern = re.compile(f'[^{base_pattern}]')

    # 使用空字符串替换所有非英文字符（以及根据选择不保留的数字和空格）
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text


def penn_treebank_tag_to_wordnet_tag(treebank_tag):
    """
    将 Penn Treebank 词性标签转换为 WordNet 兼容的词性标签。

    参数:
        treebank_tag (str): Penn Treebank 词性标签。

    返回:
        str: WordNet 词性标签 (如 `wn.NOUN`)，如果无法映射则返回 None。
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # 对于其他词性（如介词、连词、代词等），返回 None 或默认处理
        return None
ptb_to_wn_tag = penn_treebank_tag_to_wordnet_tag


def get_wordnet_pos_from_sentence(sentence: str, target_word: str):
    """
    在句子上下文中获取目标单词的所有WordNet词性标签。

    Args:
        sentence: 包含目标单词的完整句子。
        target_word: 需要获取词性的目标单词。

    Returns:
        list: 一个列表，每个元素是一个元组，包含匹配单词的索引、单词本身和其WordNet词性标签。
              例如：[(0, 'Can', 'v'), (2, 'can', 'v'), (4, 'can', 'n')]
    """
    words = word_tokenize(sentence)
    pos_tagged = pos_tag(words)  # 得到Penn Treebank标签
    results = []

    for index, (word, ptb_tag) in enumerate(pos_tagged):
        if word.lower() == target_word.lower():
            wn_tag = ptb_to_wn_tag(ptb_tag)
            results.append((index, word, wn_tag))

    return results


def is_valid_word(word: str, min_length: int = 2) -> bool:
    """
    检查一个字符串是否为有效的单词（过滤数字、纯符号等）。

    Args:
        word (str): 待检查的字符串。
        min_length (int): 单词的最小有效长度。

    Returns:
        bool: 如果是有效单词则返回 True，否则返回 False。
    """
    if not word:
        return False
    if len(word) < min_length:
        return False
    # 过滤掉包含数字的字符串
    if re.search(r'\d', word):
        return False
    # 可以添加其他过滤规则，例如过滤掉纯符号（但经过预处理后通常不会出现）
    return True


def get_top_words(word_freq: Dict[str, int], n: int = 10) -> List[Tuple[str, int]]:
    """
    获取前N个最常出现的单词。

    Args:
        word_freq (Dict[str, int]): 词频字典。
        n (int): 要获取的顶部单词数量，默认为10。

    Returns:
        List[Tuple[str, int]]: 前N个单词及其频率的列表。
    """
    return Counter(word_freq).most_common(n)


def analyze_collocations(text, top_n=20):
    """
    分析常见的词性搭配模式，这有助于发现英语中的习惯用法
    例如：动词+介词（VB+IN）、形容词+名词（JJ+NN）等。
    """

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # 定义一些常见的、有意义的词性组合模式

    patterns = [
        # 基础模式
        (('VB', 'IN'), 'Verb+Prep'),  # 动词+介词，如：look at, depend on, talk about
        (('VB', 'DT', 'NN'), 'Verb+Det+Noun'),  # 动词+限定词+名词，如：have a look, make a decision, take the chance
        (('JJ', 'NN'), 'Adj+Noun'),  # 形容词+名词，如：red apple, important meeting, difficult situation
        (('RB', 'VB'), 'Adv+Verb'),  # 副词+动词，如：quickly run, easily understand, carefully consider
        (('NN', 'IN', 'NN'), 'Noun+Prep+Noun'),  # 名词+介词+名词，如：transition to adulthood, key to success, fear of failure
        (('VB', 'RB'), 'Verb+Adv'),  # 动词+副词，如：speak clearly, work efficiently, respond immediately
        (('IN', 'DT', 'NN'), 'Prep+Det+Noun'),  # 介词+限定词+名词，如：in the morning, on a mission, with an idea
        (('NN', 'NN'), 'Compound Noun'),  # 复合名词，如: coffee cup, business meeting, research paper
        (('VB', 'NN'), 'Verb+Noun'),  # 动词+名词，如: make progress, take notes, set goals

        # 新增模式
        (('VB', 'DT', 'JJ', 'NN'), 'Verb+Det+Adj+Noun'), # 动词+限定词+形容词+名词，如：have a great day, make an important decision, see the beautiful sunset
        (('JJ', 'JJ', 'NN'), 'Adj+Adj+Noun'),  # 形容词+形容词+名词，如：beautiful red rose, large wooden table, small black cat
        (('RB', 'JJ'), 'Adv+Adj'),  # 副词+形容词，如：extremely important, very happy, quite difficult
        (('NN', 'VB'), 'Noun+Verb'),  # 名词+动词，如：problem solving, decision making, time management
        (('VB', 'PRP'), 'Verb+Pronoun'),  # 动词+代词，如：help me, tell them, ask us
        (('IN', 'JJ', 'NN'), 'Prep+Adj+Noun'),  # 介词+形容词+名词，如：in great detail, with special care, on important matters
        (('DT', 'NN', 'IN', 'NN'), 'Det+Noun+Prep+Noun'), # 限定词+名词+介词+名词，如：the end of time, a piece of cake, the beginning of history
        (('MD', 'VB', 'RB'), 'Modal+Verb+Adv'), # 情态动词+动词+副词，如：can easily do, will quickly go, should carefully consider
        (('NN', 'IN', 'DT', 'NN'), 'Noun+Prep+Det+Noun'), # 名词+介词+限定词+名词，如：transition to a new, solution to the problem, key to a mystery
        (('VB', 'TO', 'VB'), 'Verb+To+Verb'),  # 动词+不定式标记+动词，如：want to go, need to see, try to understand
        (('VBG', 'NN'), 'Gerund+Noun'),  # 动名词+名词，如：reading books, making progress, writing letters
        (('VBN', 'IN'), 'PastPart+Prep'),  # 过去分词+介词，如：interested in, covered with, known for
        (('CD', 'NNS'), 'Number+PluralNoun'),  # 基数词+名词复数，如：three books, five years, ten students
        (('JJ', 'CC', 'JJ'), 'Adj+Conj+Adj'),  # 形容词+连词+形容词，如：simple and effective, short but clear, tired yet happy
        (('VB', 'PRP', 'RB'), 'Verb+Pronoun+Adv'),  # 动词+代词+副词，如：tell me quickly, show them clearly, ask us politely
        (('RB', 'RB', 'JJ'), 'Adv+Adv+Adj'), # 副词+副词+形容词，如：very extremely hot, quite surprisingly good, rather unexpectedly cold
        (('DT', 'JJ', 'NN', 'VBZ'), 'Det+Adj+Noun+Verb'), # 限定词+形容词+名词+动词，如：the quick brown fox jumps, a beautiful red rose blooms
        (('PRP', 'MD', 'VB', 'RB'), 'Pron+Modal+Verb+Adv'), # 代词+情态动词+动词+副词，如：I can easily do, you should carefully consider, we will quickly go
        (('NN', 'VBZ', 'JJ'), 'Noun+Verb+Adj'),  # 名词+动词+形容词，如: time flies fast, sun sets red, water runs clear
        (('IN', 'PRP$', 'NN'), 'Prep+Possessive+Noun')  # 介词+物主代词+名词，如：in my opinion, on his behalf, with her permission
    ]

    collocation_counts = {desc: Counter() for _, desc in patterns}
    window_size = 4  # 检查相邻单词的窗口大小

    for i in range(int(len(tagged_tokens) - window_size + 1)):
        window = tagged_tokens[i:i + window_size]
        for (pattern, description) in patterns:
            if len(pattern) <= len(window):
                match = True
                for j in range(len(pattern)):
                    if window[j][1] != pattern[j]:
                        match = False
                        break
                if match:
                    phrase = ' '.join([word for word, pos in window[:len(pattern)]])
                    collocation_counts[description][phrase] += 1

    # 获取每种模式的前top_n个最常见搭配
    top_collocations = {}
    for desc, counter in collocation_counts.items():
        top_collocations[desc] = counter.most_common(top_n)

    return top_collocations


def count_word_frequency(text: str,
                         remove_stopwords: bool = True,
                         min_word_length: int = 2,
                         lemmatize: bool = True) -> Tuple[List[str], Dict[str, int]]:
    """
    统计文本中单词的频率，并进行详细的预处理。

    Args:
        text (str): 要分析的文本。
        remove_stopwords (bool): 是否移除停用词，默认为 True。
        min_word_length (int): 单词最小长度，短于此长度的单词将被过滤，默认为 2。
        lemmatize (bool): 是否进行词形还原，默认为 True。

    Returns:
        Tuple[List[str], Dict[str, int]]: 句子列表和单词频率字典。

    Raises:
        ValueError: 当输入文本为空或过短时。
    """

    # 参数验证
    if not text or not isinstance(text, str):
        raise ValueError("输入文本必须是非空字符串")
    if len(text.strip()) < 10:  # 假设文本至少10个字符
        raise ValueError("输入文本过短，无法进行有意义的分析")

    # 0. 可选：初步清理文本（移除多余空格、换行等）
    clean_text = re.sub(r'\s+', ' ', text.strip())  # 将多个空白字符替换为单个空格

    # 1. 分句
    try:
        sentences = sent_tokenize(clean_text)
    except Exception as e:
        raise RuntimeError(f"分句处理失败: {str(e)}")

    # 初始化工具
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    lemmatizer = WordNetLemmatizer() if lemmatize else None
    # 创建去除标点的翻译表
    translator = str.maketrans('', '', string.punctuation)

    all_words = []

    for sentence in sentences:
        try:
            # 2. 分词
            words = word_tokenize(sentence)

            processed_words = []
            for word in words:
                # 2.1 转换为小写
                word_lower = word.lower()
                # 2.2 去除标点符号（使用translate方法，比循环判断效率高）[10,11](@ref)
                word_no_punct = word_lower.translate(translator)
                # 2.3 检查是否为有效单词（长度、是否包含数字等）
                if not is_valid_word(word_no_punct, min_word_length):
                    continue
                # 2.4 检查停用词
                if remove_stopwords and word_no_punct in stop_words:
                    continue

                processed_words.append(word_no_punct)

            # 如果当前句子经过过滤后没有词，则跳过后续处理
            if not processed_words:
                continue

            # 3. 词性标注与词形还原 (如果需要)
            if lemmatize and lemmatizer:
                # 对处理后的单词进行词性标注（注意：这里标注的是原始clean word，但实际用的是转小写后的，略有误差但可接受）
                pos_tags = pos_tag(processed_words) # 返回形式如 [('word', 'tag'), ...]
                final_words = []
                for word, tag in pos_tags:
                    # 获取WordNet词性
                    wn_tag = ptb_to_wn_tag(tag)
                    if wn_tag:
                        # 进行词形还原
                        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                        final_words.append(lemma)
                    else:
                        final_words.append(word)
            else:
                final_words = processed_words # 不进行词形还原

            all_words.extend(final_words)

        except Exception as e:
            print(f"处理句子时出错: '{sentence}'. 错误: {str(e)}")
            traceback.print_exc()
            continue

    # 4. 统计词频
    word_freq = Counter(all_words)

    return sentences, dict(word_freq)





# def count_word_frequency(text: str):
#     # 1. 分句
#     sentences = sent_tokenize(text)
#
#     # 2. 分词 & 预处理（转换为小写、去除标点和停用词）
#     stop_words = set(stopwords.words('english'))
#     translator = str.maketrans('', '', string.punctuation)
#     all_words = []
#
#     for sentence in sentences:
#         words = word_tokenize(sentence)
#         # 转换为小写并去除标点符号
#         words = [word.translate(translator).lower() for word in words]
#         # 去除停用词和空字符串
#         words = [word for word in words if word not in stop_words and word]
#         all_words.extend(words)
#
#     # 3. 词形还原 (需要先进行词性标注以获得最佳效果)
#     # 注意：为简化示例，我们假设所有词都是名词('n')。实际应用中应进行词性标注。
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in all_words]  # 尝试动词形式
#
#     # 4. 统计词频
#     word_freq = Counter(lemmatized_words)
#     return sentences, word_freq


# def analyze_sentence_patterns(sentences):
#     """
#     一个简单基于词性标签序列的句型统计示例。
#     这只是一个初级方法，更准确的句型分析需要依赖句法分析。
#     """
#     # 常见的词性标记: NN(名词), VB(动词), IN(介词), DT(冠词), JJ(形容词), PRP(人称代词)
#     pattern_freq = Counter()
#
#     for sentence in sentences:
#         words = word_tokenize(sentence)
#         # 获取每个词的词性标签
#         pos_tags = [tag for word, tag in nltk.pos_tag(words)]
#         # 将词性标签序列转换为一个字符串，作为句型的近似表示
#         pattern_key = ' '.join(pos_tags)
#         pattern_freq[pattern_key] += 1
#
#         # 你也可以定义一些规则，将特定的词性序列映射到你定义的句型名称上
#         # if pos_tags starts with 'PRP VB' -> S+V pattern?
#
#     return pattern_freq



# 使用示例
if __name__ == "__main__":
    # directory = "PeppaPig"
    # results = process_all_docx_files(directory)
    #
    # # 查看处理结果（例如，打印第一个文件的内容）
    # with open('pure_text.txt', 'wt') as f:
    #     for filename, content in results.items():
    #         print(f"文件: {filename}")
    #         print("处理后的内容预览:")
    #         print(content[:500] + "..." if len(content) > 500 else content)  # 打印前500字符
    #         print("\n" + "=" * 50 + "\n")
    #         f.write(content)

    with open('pure_text.txt', 'rt') as f:
        full_text = f.read()

    # ------------------------------------------------------------------------------------------------------------------

    sentences, frequency = count_word_frequency(full_text)

    print("词频统计结果 (合并变形后):")
    for word, count in frequency.most_common(10):  # 打印最常见的10个词
        print(f"{word}: {count}")

    # 创建句子列表的DataFrame
    df_sentences = pd.DataFrame(sentences, columns=['Sentences'])

    # 创建词频统计的DataFrame
    df_frequency = pd.DataFrame(frequency.items(), columns=['Word', 'Frequency'])
    # 按词频从高到低排序
    df_frequency.sort_values(by='Frequency', ascending=False, inplace=True)

    with pd.ExcelWriter('text_analysis_results.xlsx', engine='openpyxl') as writer:
        df_sentences.to_excel(writer, sheet_name='Sentences', index=False)
        df_frequency.to_excel(writer, sheet_name='Word Frequency', index=False)

    print("分析结果已成功导出到 'text_analysis_results.xlsx'")

    # ------------------------------------------------------------------------------------------------------------------

    collocations = analyze_collocations(full_text)

    print("\n=== 常见表达方式归类 ===")
    for pattern_type, phrases in collocations.items():
        if phrases:  # 只显示有结果的类型
            print(f"\n--- {pattern_type} ---")
            for phrase, freq in phrases:
                print(f"{phrase}: {freq}")

    data = []
    for collocation_type, phrases_list in collocations.items():
        for phrase, frequency in phrases_list:
            data.append({
                "搭配模式": collocation_type,
                "搭配短语": phrase,
                "出现频率": frequency
            })

    df_collocations = pd.DataFrame(data)

    # 3. 写入 Excel 文件
    excel_filename = "collocation_analysis_results.xlsx"
    df_collocations.to_excel(excel_filename, index=False)

    print(f"搭配分析结果已保存到 '{excel_filename}'")


# ----------------------------------------------------------------------------------------------------------------------

def demo_remove_non_english():
    # 测试用例
    test_text = "Hello, 你好！ This is a test. 123 456 🎉"

    # 示例 1: 默认模式（只保留字母和空格）
    result1 = remove_non_english(test_text)
    print("默认模式 (保留字母和空格):", result1)  # 输出: "Hello  This is a test  "

    # 示例 2: 保留字母和数字
    result2 = remove_non_english(test_text, keep_space=False, keep_number=True)
    print("保留字母和数字:", result2)  # 输出: "HelloThisisatest123456"

    # 示例 3: 只保留英文字母
    result3 = remove_non_english(test_text, keep_space=False, keep_number=False)
    print("只保留英文字母:", result3)  # 输出: "HelloThisisatest"

    # 示例 4: 保留字母、数字和空格
    result4 = remove_non_english(test_text, keep_space=True, keep_number=True)
    print("保留字母、数字和空格:", result4)  # 输出: "Hello  This is a test. 123 456 "


def main():
    demo_remove_non_english()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
    finally:
        pass
