# EnglishScriptAnalysis

## 起因

为了更有针对性地学习英文，找到日常高频使用的单词和表达句式，因此调教AI写了这个程序（毕竟我不是搞NLP的）。

本程序编码几乎都由AI实现，我做了些许调整及重构：修正了AI的一些错误实现，并为复用调整了代码结构。

## 结论

我已经在各个剧本的目录下生成了统计报告（.xlsx格式），所以你不必运行程序也能看到统计的结果。

[PeppaPig](PeppaPig) | [HoC](HoC) | [Friends](Friends)

## 使用

你也可以亲自运行程序进行分析，或者直接用来分析你希望分析的其它文本。

本程序使用python3.10运行，假设当前环境已经是python3.10：

```cmd
# 安装依赖
pip install -r [requirements.txt](requirements.txt)

# 分析Peppa Pig
python [AnalyzePeppaPig.py](AnalyzePeppaPig.py)

# 分析纸牌屋
python [AnalyzeHoC.py](AnalyzeHoC.py)

# 分析Friends
python [AnalyzeFriends.py](AnalyzeFriends.py)
```
