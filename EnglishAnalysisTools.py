import re
import nltk
import string
import traceback
import unicodedata
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


def normalize_punctuation_to_ascii(text):
    """
    将常见的中文标点符号转换为对应的英文标点符号。
    注意：这是一个示例性的映射，并非所有标点都有一一对应关系，你可以根据需要扩充。

    Args:
        text (str): 输入的原始文本，可能包含中文标点。

    Returns:
        str: 转换后的文本，中文标点被替换为英文标点。
    """
    # 构建一个中文标点到英文标点的映射字典
    punctuation_map = {
        '，': ',',  # 中文逗号 -> 英文逗号
        '。': '.',  # 中文句号 -> 英文句号
        '；': ';',  # 中文分号 -> 英文分号
        '：': ':',  # 中文冒号 -> 英文冒号
        '？': '?',  # 中文问号 -> 英文问号
        '！': '!',  # 中文感叹号 -> 英文感叹号
        '“': '"',  # 中文双引号 -> 英文双引号
        '”': '"',
        '‘': "'",  # 中文单引号 -> 英文单引号
        '’': "'",
        '（': '(',  # 中文括号 -> 英文括号
        '）': ')',
        '【': '[',  # 中文方括号 -> 英文方括号
        '】': ']',
        '《': '<',  # 中文书名号 -> 英文尖括号 (或通常也直接去除)
        '》': '>',
        '～': '~',  # 中文波浪号 -> 英文波浪号
        '—': '-',  # 中文破折号 -> 英文连字符 (这是一个近似替换)
        '…': '...',  # 中文省略号 -> 英文省略号
    }

    # 逐个字符检查并替换
    normalized_text = []
    for char in text:
        if char in punctuation_map:
            normalized_text.append(punctuation_map[char])
        else:
            normalized_text.append(char)

    return ''.join(normalized_text)


def full_width_to_ascii(text):
    """
    将全角字母和数字转换为半角（ASCII）字母和数字。
    全角字符的Unicode范围通常从FF01到FF5E（对应数字和字母），
    它们与半角ASCII字符有固定的偏移量（0xFEE0）。
    此函数不会影响标点、汉字等其他字符。

    Args:
        text (str): 输入的文本，可能包含全角字母和数字。

    Returns:
        str: 转换后的文本，全角字母和数字被转换为半角。
    """
    normalized_text = []
    for char in text:
        # 获取字符的Unicode名称，常用于判断字符类型
        name = unicodedata.name(char, '')
        # 检查是否为全角字符（FULLWIDTH ...）
        if 'FULLWIDTH' in name:
            # 尝试将其转换为半角形式
            try:
                # 使用 unicodedata.normalize 转换，但更直接的是计算其半角码点
                # 全角字符与半角字符的码点相差 0xFEE0
                half_width_char = chr(ord(char) - 0xFEE0)
                # 确保转换后的字符确实是ASCII（例如，全角'A'转半角'A'）
                if half_width_char.isascii():
                    normalized_text.append(half_width_char)
                else:
                    # 如果转换后不是ASCII，保留原字符（例如某些全角符号）
                    normalized_text.append(char)
            except (ValueError, TypeError):
                # 如果转换出错，保留原字符
                normalized_text.append(char)
        else:
            # 如果不是全角字符，直接保留
            normalized_text.append(char)
    return ''.join(normalized_text)


def keep_only_ascii(text):
    """
    移除字符串中的所有非ASCII字符。

    Args:
        text (str): 待处理的字符串。

    Returns:
        str: 只包含ASCII字符的字符串。
    """
    # 使用正则表达式匹配非ASCII字符（码点 > 127 的字符）并移除
    # 模式 [^\x00-\x7F] 匹配所有非ASCII字符
    ascii_text = re.sub(r'[^\x00-\x7F]', '', text)
    return ascii_text


def remove_digits(text):
    """
    移除字符串中的所有阿拉伯数字。

    Args:
        text (str): 待处理的字符串。

    Returns:
        str: 不包含数字的字符串。
    """
    # 使用正则表达式移除非数字字符
    # 模式 [0-9] 匹配所有数字
    no_digits_text = re.sub(r'[0-9]', '', text)
    return no_digits_text


def replace_unwanted_symbols(text, keep_chars=""",?!:"'"""):
    """
    增强版的符号替换函数，尝试区分单词连字符和数学减号，同时区分句号和小数点。

    Args:
        text (str): 输入文本
        keep_chars (str): 额外需要保留的字符

    Returns:
        str: 处理后的文本
    """
    # 使用罕见字符作为临时标记
    temp_marker_hyphen = "▦"  # 用于受保护的连字符
    temp_marker_period = "◎"  # 用于受保护的句号

    # 1. 保护单词中的连字符（前后是字母）
    protected_text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', temp_marker_hyphen, text)

    # 2. 保护疑似句号（不在数字间）
    protected_text = re.sub(r'\.(?!(?<=\d\.)\d)', temp_marker_period, protected_text)

    # 3. 定义基础保留集合（字母、数字、空格、临时标记）
    base_keep = r'\w\s'
    # 构建保留字符模式（基础保留 + 用户指定保留 + 临时标记）
    keep_pattern = f"{base_keep}{re.escape(keep_chars)}{temp_marker_hyphen}{temp_marker_period}"

    # 4. 将所有不在保留集中的字符替换为空格
    cleaned_text = re.sub(f"[^{keep_pattern}]", ' ', protected_text)

    # 5. 恢复受保护的连字符和句号
    final_text = cleaned_text.replace(temp_marker_hyphen, '-').replace(temp_marker_period, '.')

    return final_text


def reduce_blank_lines(text, max_blanks=2):
    """
    将文本中连续的空行减少到指定最大数量（默认保留2个）

    Args:
        text (str): 输入的文本字符串
        max_blanks (int): 允许保留的最大连续空行数，默认为2

    Returns:
        str: 处理后的文本字符串
    """
    # 匹配超过max_blanks的连续空行模式
    # \n\s* 匹配一个换行符及其后的任意空白字符（包括后续换行符）
    pattern = r'(\n\s*){' + str(max_blanks + 1) + r',}'
    # 替换为恰好max_blanks个空行（即max_blanks个换行符）
    replacement = '\n' * max_blanks
    # 使用正则替换，re.DOTALL确保.匹配包括换行符在内的所有字符
    cleaned_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return cleaned_text


def remove_non_english(text, keep_number=False):
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

    step1_text = normalize_punctuation_to_ascii(text)
    step2_text = full_width_to_ascii(step1_text)
    step3_text = keep_only_ascii(step2_text)
    step4_text = replace_unwanted_symbols(step3_text)
    step5_text = remove_digits(step4_text) if not keep_number else step4_text
    return step5_text


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

# ----------------------------------------------------------------------------------------------------------------------

def demo_remove_non_english():
    # 测试用例
    test_text = "Hello, 你好！ This is a test. 123 456 🎉"

    # 示例 1: 默认模式（只保留字母和空格）
    result1 = remove_non_english(test_text)
    print("默认模式 (保留字母和空格):", result1)  # 输出: "Hello  This is a test  "

    # 示例 2: 保留字母和数字
    result2 = remove_non_english(test_text, keep_number=True)
    print("保留字母和数字:", result2)  # 输出: "HelloThisisatest123456"


def demo_replace_unwanted_symbols():
    test_cases = [
        # 基础测试：保留字母、数字、空格
        ('Hello World 123', '', 'Hello World 123'),
        # 连字符测试1：单词中的连字符应保留
        ('bi-directional optimization', '', 'bi-directional optimization'),
        # 连字符测试2：数学表达式中的减号应替换为空格
        ('3 - 2 result is 1', '', '3   2 result is 1'),
        ('3-2 result is 1', '', '3 2 result is 1'),
        # 句点测试1：句号应保留
        ('This is a sentence. Another one.', '', 'This is a sentence. Another one.'),
        # 句点测试2：小数点应替换为空格
        ('The value is 3.14', '', 'The value is 3 14'),
        # 混合测试1：同时包含单词连字符、数学减号、句号和小数点
        ('pi is approx 3.14. pre-defined value: 5 - 3 = 2.', ':', 'pi is approx 3 14. pre-defined value: 5   3   2.'),
        # 自定义保留字符测试1：保留@符号
        ('Email me at user@example.com', '@', 'Email me at user@example.com'),
        # 自定义保留字符测试2：保留逗号和问号
        ('Hello, world! How are you?', ',?', 'Hello, world  How are you?'),
        # 边界测试1：字符串开头和结尾的符号
        ('!@#$Hello%^&*', '', '    Hello    '),
        # 边界测试2：空字符串
        ('', '', ''),
        # 边界测试3：只有符号
        ('!@#$%^&*', '', '        '),
        # 复杂符号测试：多种符号混合
        ('a-b c-d e.f g,h i;j k|l', '', 'a-b c-d e.f g h i j k l'),
        # 数字和符号组合测试
        ('1-2-3 4.5 6,7 8:9', '', '1 2 3 4 5 6 7 8 9'),
        # 保留字符中的连字符处理（如果keep_chars中包含'-'，且位于字符集末尾）
        ('This-is-a-test-string', '-', 'This-is-a-test-string'),
        # 保留字符中的句点处理（如果keep_chars中包含'.'）
        ('Version.1.2.3', '.', 'Version.1.2.3'),
    ]

    print("开始测试 replace_unwanted_symbols 函数：")
    print("=" * 60)

    for i, (input_text, keep_chars, expected_output) in enumerate(test_cases, 1):
        result = replace_unwanted_symbols(input_text, keep_chars)
        print(f"测试用例 {i}:")
        print(f"  输入文本: '{input_text}'")
        print(f"  保留字符: '{keep_chars}'")
        print(f"  期望输出: '{expected_output}'")
        print(f"  实际输出: '{result}'")
        print(f"  是否通过: {result == expected_output}")
        print("-" * 40)


def main():
    demo_remove_non_english()
    demo_replace_unwanted_symbols()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
    finally:
        pass
