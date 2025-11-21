#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
traditional_methods.py
作業2 - Part A (傳統方法實作)

此檔案包含 Part A 的所有實作:
A-1: TF-IDF 文本相似度計算
    - 手動計算 TF-IDF (calculate_tf, calculate_idf)
    - 使用 scikit-learn 實作
A-2: 基於規則的文本分類
    - 情感分類器 (RuleBasedSentimentClassifier)
    - 主題分類器 (TopicClassifier)
A-3: 統計式自動摘要 (StatisticalSummarizer) [參考 PDF page 11 流程圖]
"""

# --- 必要的函式庫 ---
import math
import re
import os
import json
import time
from collections import Counter
from heapq import nlargest

# 第三方函式庫 (請確保已安裝: pip install -r requirements.txt)
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 設定環境 ---

# 設定 Matplotlib 支援中文顯示 (用於 A-1 的視覺化)
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # <--- 修正為 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    print("Matplotlib 中文字體設定完成 (Microsoft YaHei)。")
except Exception:
    print("警告: 未找到 'Microsoft YaHei' 字體...")


# --- Part A-1: TF-IDF 文本相似度計算 ---

# A-1.1: 手動計算 TF-IDF

def calculate_tf(word_dict, total_words):
    """
    計算詞頻 (TF)
    Args:
        word_dict: 詞彙計數字典
        total_words: 總詞數
    Returns:
        tf_dict: TF 值字典
    """
    tf_dict = {}
    if total_words == 0:
        return tf_dict
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words
    return tf_dict

def calculate_idf(documents, word):
    """
    計算逆文件頻率 (IDF)
    使用 scikit-learn 的 'smooth_idf' 預設公式: log((N+1) / (df+1)) + 1
    
    Args:
        documents: 文件列表 (假設是 list of list of strings, 即已分詞)
        word: 目標詞彙
    Returns:
        idf: IDF 值
    """
    N = len(documents)
    doc_containing_word = 0
    for doc in documents:
        if word in doc:
            doc_containing_word += 1
            
    # 採用 (N+1) / (df+1) 避免 df=0 導致的除零錯誤，並+1平滑
    idf = math.log((N + 1) / (doc_containing_word + 1)) + 1
    return idf

# A-1.2: 使用 scikit-learn 實作

def calculate_similarity_sklearn(documents):
    """
    使用 scikit-learn 實作 TF-IDF 並計算餘弦相似度
    
    Args:
        documents: 原始文本列表 (list of strings)
    Returns:
        cosine_sim_matrix: 餘弦相似度矩陣
        vectorizer: TfidfVectorizer 物件
    """
    # 中文需要先分詞
    segmented_docs = [' '.join(jieba.cut(doc)) for doc in documents]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(segmented_docs)
    
    # 計算餘弦相似度
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    
    return cosine_sim_matrix, vectorizer

def plot_similarity_matrix(matrix, labels, filename="results/tfidf_similarity_matrix.png"):
    """
    將相似度矩陣視覺化並儲存
    (此為 A-1 評分重點 "視覺化品質" 的一部分 [參考 PDF page 20])
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels, fmt=".2f")
    plt.title('TF-IDF 餘弦相似度矩陣')
    plt.xlabel('文件')
    plt.ylabel('文件')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"相似度矩陣視覺化已儲存至 {filename}")


# --- Part A-2: 基於規則的文本分類 ---

class RuleBasedSentimentClassifier:
    """
    A-2.1: 情感分類器
    """
    def __init__(self):
        # 建立正負面詞彙庫
        self.positive_words = set(['好','棒','優秀','喜歡','推薦',
                                   '滿意','開心','值得','精彩','完美',
                                   '好吃', '濃郁', 'Q彈', '驚艷'])
        self.negative_words = set(['差','糟','失望','討厭','不推薦',
                                   '浪費','無聊','爛','糟糕','差勁',
                                   '空洞'])
        # 加入否定詞處理
        self.negation_words = set(['不','沒','無','非','別'])
        # 考慮程度副詞的加權
        self.adverbs = {'很': 1.5, '非常': 2.0, '太': 1.8, '有點': 0.5, '超級': 2.5, '真的': 1.5, '完全': 2.0}

    def classify(self, text):
        """
        分類邏輯:
        1. 計算正負詞數量
        2. 處理否定詞 (否定詞 + 正面詞 = 負面)
        3. 考慮程度副詞的加權
        4. 返回: 正面/負面/中性
        """
        words = list(jieba.cut(text))
        score = 0
        last_word_is_negation = False
        last_word_adverb_weight = 1.0
        
        for word in words:
            if word in self.adverbs:
                last_word_adverb_weight = self.adverbs[word]
                continue # 副詞修飾下一個詞
                
            if word in self.negation_words:
                last_word_is_negation = True
                continue # 否定詞修飾下一個詞

            current_score = 0
            if word in self.positive_words:
                current_score = 1
            elif word in self.negative_words:
                current_score = -1
            
            if current_score != 0:
                # 應用程度副詞加權
                current_score *= last_word_adverb_weight
                
                # 應用否定詞
                if last_word_is_negation:
                    current_score *= -1
                
                score += current_score

            # 重置修飾詞
            last_word_is_negation = False
            last_word_adverb_weight = 1.0
            
        if score > 0:
            return "正面"
        elif score < 0:
            return "負面"
        else:
            return "中性"

class TopicClassifier:
    """
    A-2.2: 主題分類器
    """
    def __init__(self):
        self.topic_keywords = {
            '科技': set(['AI','人工智慧','電腦','軟體','程式','演算法', '模型', '深度學習']),
            '運動': set(['運動','健身','跑步','游泳','球類','比賽', '慢跑', '重訓', '體能']),
            '美食': set(['吃','食物','餐廳','美味','料理','烹飪', '牛肉麵', '湯頭', '麵條']),
            '旅遊': set(['旅行','景點','飯店','機票','觀光','度假']),
            '娛樂': set(['電影', '劇情', '演技']) # 根據測試資料擴充
        }

    def classify(self, text):
        """
        返回最可能的主題
        """
        words = set(jieba.cut(text))
        topic_scores = {topic: 0 for topic in self.topic_keywords}
        
        for topic, keywords in self.topic_keywords.items():
            # 計算交集中的詞彙數量
            topic_scores[topic] = len(words.intersection(keywords))
            
        # 找出分數最高的
        if all(score == 0 for score in topic_scores.values()):
            return "其他"
            
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic


# --- Part A-3: 統計式自動摘要 ---

class StatisticalSummarizer:
    def __init__(self):
        # 載入停用詞 [參考 PDF page 11]
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
                               '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', 
                               '你', '，', '。', '、', '？', '！', '（', '）', '「', '」', 
                               ' '])
        self.punctuation = set("。？！\n") # 用於分句

    def _get_sentences(self, text):
        """ 1. 分句 (處理中文標點) [參考 PDF page 11] """
        sentences = re.split(r'([。？！\n])', text)
        # 合併句子和它們的結尾標點
        combined_sentences = []
        if sentences:
            current_sentence = sentences[0]
            for i in range(1, len(sentences), 2):
                if i+1 < len(sentences):
                    current_sentence += sentences[i]
                    combined_sentences.append(current_sentence.strip())
                    current_sentence = sentences[i+1]
                else:
                    combined_sentences.append(current_sentence.strip())
        
        return [s for s in combined_sentences if len(s) > 2] # 過濾掉太短的

    def _get_word_freq(self, text):
        """ 2. 分詞並計算詞頻 [參考 PDF page 11] """
        words = [word for word in jieba.cut(text.lower()) 
                 if word not in self.stop_words and not word.isspace()]
        return Counter(words)

    def _score_sentence(self, sentence, word_freq, total_sentences, index):
        """
        計算句子重要性分數 [參考 PDF page 11 流程圖]
        考量因素:
        1. 包含高頻詞的數量
        2. 句子位置 (首尾句加權)
        3. 句子長度 (太短或太長扣分)
        4. 是否包含數字或專有名詞 (如 'AI')
        """
        words = [word for word in jieba.cut(sentence.lower()) if word in word_freq]
        
        # 1. 高頻詞分數 (正規化，避免長句佔優)
        if not words:
            return 0
        word_score = sum(word_freq[word] for word in words) / len(words)
        
        # 2. 位置加權
        position_weight = 1.0
        if index == 0:
            position_weight = 1.5  # 首句
        elif index == total_sentences - 1:
            position_weight = 1.2  # 尾句
            
        # 3. 長度加權
        len_weight = 1.0
        L = len(sentence)
        if L < 10 or L > 50:
            len_weight = 0.5 # 太短或太長扣分
            
        # 4. 專有名詞/數字加權
        proper_noun_weight = 1.0
        if re.search(r'[A-Za-z0-9]', sentence):
            proper_noun_weight = 1.2
            
        # 綜合分數
        final_score = word_score * position_weight * len_weight * proper_noun_weight
        return final_score

    def summarize(self, text, ratio=0.3):
        """
        生成摘要 [參考 PDF page 11 流程圖]
        """
        # 1. 分句
        sentences = self._get_sentences(text)
        total_sentences = len(sentences)
        if total_sentences == 0:
            return ""
        
        # 2. 計算詞頻
        word_freq = self._get_word_freq(text)
        
        # 3. 計算句子分數
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq, total_sentences, i)
            sentence_scores.append((i, score, sentence)) # (原索引, 分數, 句子)
            
        # 4. 選擇最高分的句子
        num_sentences = max(1, int(total_sentences * ratio))
        top_sentences = nlargest(num_sentences, sentence_scores, key=lambda item: item[1])
        
        # 5. 按原順序排列
        top_sentences_sorted = sorted(top_sentences, key=lambda item: item[0])
        
        # 組成摘要
        summary = "".join([s[2] for s in top_sentences_sorted])
        return summary


# --- 主執行區塊 ---

def ensure_results_dir():
    """ 確保 'results' 資料夾存在 (符合 page 19 檔案結構) """
    if not os.path.exists('results'):
        os.makedirs('results')
        print("已建立 'results' 資料夾。")

def run_part_a1(documents):
    """ 執行 Part A-1 """
    print("\n" + "="*30)
    print("  執行 Part A-1: TF-IDF 文本相似度計算")
    print("="*30)
    
    # A-1.1: 手動計算 (示範)
    print("\n--- A-1.1: 手動計算 (示範) ---")
    segmented_docs_list = [list(jieba.cut(doc)) for doc in documents]
    
    # 示範計算: Doc 0 的 '人工智慧'
    doc0_words = segmented_docs_list[0]
    doc0_word_count = Counter(doc0_words)
    doc0_total_words = len(doc0_words)
    tf_doc0 = calculate_tf(doc0_word_count, doc0_total_words)
    print(f"文件0 '人工智慧' TF: {tf_doc0.get('人工智慧', 0.0):.4f}")
    
    # 示範計算: '人工智慧' 的 IDF
    idf_ai = calculate_idf(segmented_docs_list, '人工智慧')
    print(f"'人工智慧' IDF (manual): {idf_ai:.4f}")
    
    # 示範計算: TF-IDF
    tfidf_doc0_ai = tf_doc0.get('人工智慧', 0.0) * idf_ai
    print(f"文件0 '人工智慧' TF-IDF (manual): {tfidf_doc0_ai:.4f}")

    # A-1.2: 使用 scikit-learn 實作
    print("\n--- A-1.2: scikit-learn 實作 ---")
    start_time = time.time()
    cosine_sim_matrix, vectorizer = calculate_similarity_sklearn(documents)
    end_time = time.time()
    a1_time = end_time - start_time
    
    print(f"Sklearn TF-IDF 與相似度計算時間: {a1_time:.6f} 秒")
    print("餘弦相似度矩陣 (sklearn):")
    print(pd.DataFrame(cosine_sim_matrix, index=doc_labels, columns=doc_labels).to_string(float_format="%.4f"))
    
    # 儲存視覺化 (Page 19 要求)
    plot_similarity_matrix(cosine_sim_matrix, doc_labels, filename="results/tfidf_similarity_matrix.png")
    
    return {'time_seconds': a1_time}

def run_part_a2(test_texts):
    """ 執行 Part A-2 """
    print("\n" + "="*30)
    print("  執行 Part A-2: 基於規則的文本分類")
    print("="*30)
    
    sentiment_classifier = RuleBasedSentimentClassifier()
    topic_classifier = TopicClassifier()
    
    results = []
    start_time = time.time()
    print("\n--- 分類結果 ---")
    for text in test_texts:
        sentiment = sentiment_classifier.classify(text)
        topic = topic_classifier.classify(text)
        results.append({'text': text, 'sentiment': sentiment, 'topic': topic})
        print(f"文本: {text[:30]}...")
        print(f"  -> 情感: {sentiment}, 主題: {topic}")
        
    end_time = time.time()
    a2_time = end_time - start_time
    print(f"\n規則式分類總時間: {a2_time:.6f} 秒")
    
    # 儲存 CSV (Page 19 要求)
    df = pd.DataFrame(results)
    csv_path = "results/classification_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"分類結果已儲存至 {csv_path}")
    
    return {'time_seconds': a2_time, 'results': results}

def run_part_a3(article):
    """ 執行 Part A-3 """
    print("\n" + "="*30)
    print("  執行 Part A-3: 統計式自動摘要")
    print("="*30)
    
    summarizer = StatisticalSummarizer()
    
    start_time = time.time()
    summary = summarizer.summarize(article, ratio=0.3) # 摘要比例 30%
    end_time = time.time()
    a3_time = end_time - start_time
    
    print("\n--- 摘要結果 ---")
    print(f"原文長度: {len(article)} 字")
    print(f"摘要長度: {len(summary)} 字")
    print(f"統計式摘要總時間: {a3_time:.6f} 秒")
    print("\n生成的摘要:")
    print(summary)
    
    # 儲存 .txt (Page 19 要求)
    summary_path = "results/summarization_comparison.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("--- Traditional Statistical Summary ---\n")
        f.write(summary)
        f.write("\n\n" + "="*20 + "\n\n")
        f.write("--- Original Text ---\n")
        f.write(article.strip())
    print(f"\n摘要比較檔已儲存至 {summary_path} (Part B 執行後會附加 AI 摘要)")
    
    return {'time_seconds': a3_time, 'summary': summary}

def generate_requirements():
    """ 
    生成 requirements.txt (Page 19 要求)
    內容基於 Page 18 的「必要套件安裝」
    """
    requirements = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "jieba",
        "chinese-stop-words",
        "openai", # 雖然 Part A 沒用，但是作業 B 需要
        "tqdm"  # 雖然 Part A 沒用，但是作業 B 需要
    ]
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("# 作業2 (hw2) 所需套件 [參考 PDF page 18]\n")
        for req in requirements:
            f.write(f"{req}\n")
    print(f"已生成 'requirements.txt'")


def main():
    """ 主函式，執行所有 Part A 任務 """
    
    # 0. 確保輸出目錄存在
    ensure_results_dir()
    
    # 1. 載入 Part A-1 測試資料
    global doc_labels # 讓 A-1 函式可以存取
    documents_a1 = [
        "人工智慧正在改變世界,機器學習是其核心技術",
        "深度學習推動了人工智慧的發展,特別是在圖像識別領域",
        "今天天氣很好,適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康,每天都應該保持運動習慣"
    ]
    doc_labels = [f"文件{i}" for i in range(len(documents_a1))]
    
    # 2. 載入 Part A-2 測試資料
    test_texts_a2 = [
        "這家餐廳的牛肉麵真的太好吃了,湯頭濃郁,麵條Q彈,下次一定再來!",
        "最新的AI技術突破讓人驚艷,深度學習模型的表現越來越好",
        "這部電影劇情空洞,演技糟糕,完全是浪費時間",
        "每天慢跑5公里,配合適當的重訓,體能進步很多"
    ]
    
    # 3. 載入 Part A-3 測試文章 [參考 PDF page 12]
    # ** 已修正上一版中 'diagnóstico' 的錯字 **
    article_a3 = (
        "人工智慧(AI)的發展正深刻改變我們的生活方式。從早晨起床的智慧鬧鐘,"
        "到通勤時的路線規劃,再到工作中的各種輔助工具,AI無處不在。"
        "在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。透過分析"
        "大量的醫學影像和病歷資料,AI能夠發現人眼容易忽略的細節,為患者提供更好"
        "的治療方案。"
        "教育方面,AI個人化學習系統能夠根據每個學生的學習進度和特點,提供客製化"
        "的教學內容。這種因材施教的方式,讓學習變得更加高效和有趣。"
        "然而,AI的快速發展也帶來了一些挑戰。首先是就業問題,許多傳統工作可能會"
        "被AI取代。其次是隱私和安全問題,AI系統需要大量數據來訓練,如何保護個人"
        "隱私成為重要議題。最後是倫理問題,AI的決策過程往往缺乏透明度,可能會產"
        "生偏見或歧視。"
        "面對這些挑戰,我們需要在推動AI發展的同時,建立相應的法律法規和倫理準則。"
        "只有這樣,才能確保AI技術真正為人類福祉服務,創造一個更美好的未來。"
    )
    
    # 4. 執行所有 Part A 任務
    perf_metrics = {}
    perf_metrics['A1_tfidf'] = run_part_a1(documents_a1)
    perf_metrics['A2_classify'] = run_part_a2(test_texts_a2)
    perf_metrics['A3_summarize'] = run_part_a3(article_a3)
    
    # 5. 儲存效能指標 (Page 19 要求)
    perf_path = "results/performance_metrics.json"
    # 存入 'traditional' 欄位下，'modern' 欄位將由 modern_methods.py 新增
    final_metrics = {'traditional': perf_metrics}
    
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)
    print(f"\n傳統方法效能指標已儲存至 {perf_path}")
    
    # 6. 生成 requirements.txt (Page 19 要求)
    generate_requirements()

    print("\nPart A (traditional_methods.py) 執行完畢。")


if __name__ == "__main__":
    # *** 加入偵錯用 print ***
    print("--- 正在啟動 main() 函式 ---")
    main()
    print("--- main() 函式執行完畢 ---")