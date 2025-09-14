import re
from collections import Counter
import nltk
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def preprocess_text(text):
    """
    清洗和预处理文本：转换为小写，去除标点符号和多余空格[1,4,7](@ref)。
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # 只保留字母和空格
    text = re.sub(r'\s+', ' ', text).strip()  # 将多个空格替换为一个空格并去除首尾空格
    return text


def get_ngrams(tokens, n_range=(1, 3)):
    """
    统计不同n-gram的出现频率，n-gram是连续的n个单词序列，常用于识别常见短语[1](@ref)。
    """
    all_ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        n_grams = ngrams(tokens, n)
        all_ngrams.extend([' '.join(gram) for gram in n_grams])
    return Counter(all_ngrams)


def analyze_collocations(tagged_tokens, top_n=20):
    """
    分析常见的词性搭配模式，这有助于发现英语中的习惯用法[6](@ref)。
    例如：动词+介词（VB+IN）、形容词+名词（JJ+NN）等。
    """
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
        (('VB', 'DT', 'JJ', 'NN'), 'Verb+Det+Adj+Noun'),
        # 动词+限定词+形容词+名词，如：have a great day, make an important decision, see the beautiful sunset
        (('JJ', 'JJ', 'NN'), 'Adj+Adj+Noun'),  # 形容词+形容词+名词，如：beautiful red rose, large wooden table, small black cat
        (('RB', 'JJ'), 'Adv+Adj'),  # 副词+形容词，如：extremely important, very happy, quite difficult
        (('NN', 'VB'), 'Noun+Verb'),  # 名词+动词，如：problem solving, decision making, time management
        (('VB', 'PRP'), 'Verb+Pronoun'),  # 动词+代词，如：help me, tell them, ask us
        (('IN', 'JJ', 'NN'), 'Prep+Adj+Noun'),  # 介词+形容词+名词，如：in great detail, with special care, on important matters
        (('DT', 'NN', 'IN', 'NN'), 'Det+Noun+Prep+Noun'),
        # 限定词+名词+介词+名词，如：the end of time, a piece of cake, the beginning of history
        (('MD', 'VB', 'RB'), 'Modal+Verb+Adv'),
        # 情态动词+动词+副词，如：can easily do, will quickly go, should carefully consider
        (('NN', 'IN', 'DT', 'NN'), 'Noun+Prep+Det+Noun'),
        # 名词+介词+限定词+名词，如：transition to a new, solution to the problem, key to a mystery
        (('VB', 'TO', 'VB'), 'Verb+To+Verb'),  # 动词+不定式标记+动词，如：want to go, need to see, try to understand
        (('VBG', 'NN'), 'Gerund+Noun'),  # 动名词+名词，如：reading books, making progress, writing letters
        (('VBN', 'IN'), 'PastPart+Prep'),  # 过去分词+介词，如：interested in, covered with, known for
        (('CD', 'NNS'), 'Number+PluralNoun'),  # 基数词+名词复数，如：three books, five years, ten students
        (('JJ', 'CC', 'JJ'), 'Adj+Conj+Adj'),  # 形容词+连词+形容词，如：simple and effective, short but clear, tired yet happy
        (('VB', 'PRP', 'RB'), 'Verb+Pronoun+Adv'),  # 动词+代词+副词，如：tell me quickly, show them clearly, ask us politely
        (('RB', 'RB', 'JJ'), 'Adv+Adv+Adj'),
        # 副词+副词+形容词，如：very extremely hot, quite surprisingly good, rather unexpectedly cold
        (('DT', 'JJ', 'NN', 'VBZ'), 'Det+Adj+Noun+Verb'),
        # 限定词+形容词+名词+动词，如：the quick brown fox jumps, a beautiful red rose blooms
        (('PRP', 'MD', 'VB', 'RB'), 'Pron+Modal+Verb+Adv'),
        # 代词+情态动词+动词+副词，如：I can easily do, you should carefully consider, we will quickly go
        (('NN', 'VBZ', 'JJ'), 'Noun+Verb+Adj'),  # 名词+动词+形容词，如: time flies fast, sun sets red, water runs clear
        (('IN', 'PRP$', 'NN'), 'Prep+Possessive+Noun')  # 介词+物主代词+名词，如：in my opinion, on his behalf, with her permission
    ]

    collocation_counts = {desc: Counter() for _, desc in patterns}
    window_size = 4  # 检查相邻单词的窗口大小

    for i in range(len(tagged_tokens) - window_size + 1):
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


def remove_stopwords(tokens):
    """移除停用词如'the', 'is', 'in'等，这些词通常对识别有意义表达帮助不大[4](@ref)。"""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def plot_top_ngrams(ngram_counts, n_value, top_k=10):
    """可视化特定n-gram的前top_k个结果"""
    top_ngrams = ngram_counts.most_common(top_k)
    phrases, counts = zip(*top_ngrams)

    plt.figure(figsize=(10, 6))
    plt.barh(phrases, counts)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_k} {n_value}-grams')
    plt.gca().invert_yaxis()
    plt.show()


def main(text, top_n=15):
    """
    主函数：执行完整的文本分析流程
    """
    # 1. 文本预处理
    cleaned_text = preprocess_text(text)
    tokens = word_tokenize(cleaned_text)

    # 可选：移除停用词（对于单词和短语统计，有时保留停用词更有意义，可根据需要选择）
    # tokens = remove_stopwords(tokens)

    # # 2. 统计单词频率
    # word_freq = Counter(tokens)
    # print("=== 高频单词 ===")
    # for word, freq in word_freq.most_common(top_n):
    #     print(f"{word}: {freq}")

    # 3. 统计n-gram短语（1-gram, 2-gram, 3-gram）
    ngram_counts = get_ngrams(tokens, (1, 3))

    # 分别查看不同n-gram的结果
    for n in [1, 2, 3]:
        ngram_n = {gram: count for gram, count in ngram_counts.items() if len(gram.split()) == n}
        ngram_counter = Counter(ngram_n)
        print(f"\n=== 高频 {n}-词短语 ===")
        for phrase, freq in ngram_counter.most_common(top_n):
            print(f"{phrase}: {freq}")
        # 可视化展示
        plot_top_ngrams(ngram_counter, n, top_k=top_n)

    # 4. 词性标注与搭配分析
    tagged_tokens = pos_tag(tokens)
    collocations = analyze_collocations(tagged_tokens, top_n=top_n)

    print("\n=== 常见表达方式归类 ===")
    for pattern_type, phrases in collocations.items():
        if phrases:  # 只显示有结果的类型
            print(f"\n--- {pattern_type} ---")
            for phrase, freq in phrases:
                print(f"{phrase}: {freq}")


# 示例用法
if __name__ == "__main__":
    with open('pure_text.txt', 'r', encoding='utf-8') as file:
        sample_text = file.read()
    main(sample_text, top_n=15)