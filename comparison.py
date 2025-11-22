#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
comparison.py
作業2 - Part C (比較分析)

此檔案包含 Part C 的所有實作:
C-1: 文本相似度比較 (TF-IDF vs AI Embedding)
    - 呼叫 A-1 的 calculate_similarity_sklearn
    - 呼叫 B-1 的 get_embedding_gemini
    - 視覺化並排比較圖
C-2: 文本分類比較 (規則 vs AI)
    - 讀取 results/classification_results.csv
    - 計算雙方的準確率
C-3: 自動摘要比較 (統計 vs AI)
    - 提示使用者手動比較 results/summarization_comparison.txt
    
此程式會:
1. 更新 results/performance_metrics.json (加入 C-1 和 C-2 的效能數據)
2. 建立 README.md (執行說明)
3. 建立 report.md 
"""

# --- 必要的函式庫 ---
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- 匯入 A, B 部分的函式 ---
try:
    # 匯入 A-1 的 TF-IDF 函式
    from traditional_methods import calculate_similarity_sklearn, ensure_results_dir
except ImportError:
    print("錯誤: 找不到 traditional_methods.py。請確保檔案在同一目錄下。")
    exit()

try:
    # 匯入 B-1 的 Embedding 函式
    from modern_methods import get_embedding_gemini
except ImportError:
    print("錯誤: 找不到 modern_methods.py。請確保檔案在同一目錄下。")
    exit()
except Exception as e:
    print(f"錯誤: 匯入 modern_methods.py 失敗: {e}")
    print("請確保 'google-generativeai' 已安裝 (pip install google-generativeai)")
    exit()

# --- Part C-1: 文本相似度比較 ---

def plot_comparison_matrix(matrix1, matrix2, labels, filename="results/similarity_comparison.png"):
    """
    ...
    """
    # 設定 Matplotlib 支援中文顯示
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # <--- 修正為 'Microsoft YaHei'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未找到 'Microsoft YaHei' 字體...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    sns.heatmap(matrix1, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels, fmt=".2f", ax=ax1)
    ax1.set_title('A-1: TF-IDF 相似度矩陣', fontsize=16)
    
    sns.heatmap(matrix2, annot=True, cmap='cividis', xticklabels=labels, yticklabels=labels, fmt=".2f", ax=ax2)
    ax2.set_title('B-1: AI Embedding (Gemini) 相似度矩陣', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n相似度並排比較圖已儲存至 {filename}")

def run_part_c1(documents, doc_labels):
    """
    執行 C-1: 比較 TF-IDF 和 AI Embedding 的相似度
    """
    print("\n" + "="*30)
    print("  執行 Part C-1: 文本相似度比較")
    print("="*30)
    
    metrics = {}
    
    # --- A-1: 傳統 TF-IDF ---
    print("正在計算 A-1 (TF-IDF)...")
    start_time_a1 = time.time()
    try:
        tfidf_matrix, _ = calculate_similarity_sklearn(documents)
        time_a1 = time.time() - start_time_a1
        print(f"TF-IDF 計算時間: {time_a1:.4f} 秒")
        metrics['C1_A_tfidf_comparison'] = {'time_seconds': time_a1}
    except Exception as e:
        print(f"錯誤: 執行 A-1 (TF-IDF) 失敗: {e}")
        tfidf_matrix = np.zeros((len(documents), len(documents)))
        metrics['C1_A_tfidf_comparison'] = {'time_seconds': 0, 'error': str(e)}

    # --- B-1: 現代 AI Embedding ---
    print("正在計算 B-1 (Gemini Embedding)...")
    start_time_b1 = time.time()
    try:
        # (確保 API Key 已設定)
        embeddings = [get_embedding_gemini(doc) for doc in documents]
        
        # 檢查是否有 API 錯誤
        if any(e is None for e in embeddings):
            print("錯誤: 取得 Gemini Embedding 失敗 (API Key 是否設定?)。跳過 B-1。")
            embedding_matrix = np.zeros((len(documents), len(documents)))
            time_b1 = time.time() - start_time_b1
            metrics['C1_B_embedding_comparison'] = {'time_seconds': time_b1, 'error': 'API call failed'}
        else:
            embedding_matrix = cosine_similarity(np.array(embeddings))
            time_b1 = time.time() - start_time_b1
            print(f"Gemini Embedding 計算時間: {time_b1:.4f} 秒")
            metrics['C1_B_embedding_comparison'] = {'time_seconds': time_b1}

    except Exception as e:
        print(f"錯誤: 執行 B-1 (Embedding) 失敗: {e}")
        embedding_matrix = np.zeros((len(documents), len(documents)))
        metrics['C1_B_embedding_comparison'] = {'time_seconds': 0, 'error': str(e)}

    # --- 輸出比較 ---
    print("\n--- TF-IDF 相似度矩陣 (A-1) ---")
    print(pd.DataFrame(tfidf_matrix, index=doc_labels, columns=doc_labels).to_string(float_format="%.4f"))
    
    print("\n--- Gemini Embedding 相似度矩陣 (B-1) ---")
    
    # (v3 修正: 確保此行完整)
    print(pd.DataFrame(embedding_matrix, index=doc_labels, columns=doc_labels).to_string(float_format="%.4f"))

    # --- 視覺化 ---
    plot_comparison_matrix(tfidf_matrix, embedding_matrix, doc_labels)
    
    return metrics

# --- Part C-2: 文本分類比較 ---

def run_part_c2():
    """
    執行 C-2: 比較規則式分類和 AI 分類的準確率
    """
    print("\n" + "="*30)
    print("  執行 Part C-2: 文本分類準確率比較")
    print("="*30)
    
    csv_path = "results/classification_results.csv"
    
    # C-2.1: 定義標準答案 (Ground Truth)
    ground_truth = [
        {'sentiment': '正面', 'topic': '美食'},
        {'sentiment': '正面', 'topic': '科技'},
        {'sentiment': '負面', 'topic': '娛樂'},
        {'sentiment': '正面', 'topic': '運動'},
    ]
    df_gt = pd.DataFrame(ground_truth)
    
    try:
        df_results = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {csv_path}。請先執行 traditional_methods.py 和 modern_methods.py。")
        return {}
    
    if 'ai_sentiment' not in df_results.columns:
        print(f"錯誤: {csv_path} 中缺少 'ai_sentiment' 欄位。請先執行 modern_methods.py。")
        return {}

    # C-2.2: 計算準確率
    # 過濾掉 API 執行失敗的 'Error' 行, 避免計算錯誤
    valid_results = df_results[df_results['ai_sentiment'] != 'Error'].copy()
    valid_gt = df_gt.iloc[valid_results.index]

    if valid_results.empty:
        print("警告: AI 分類結果均為 'Error', 無法計算 AI 準確率。")
        ai_sent_acc = 0.0
        ai_topic_acc = 0.0
    else:
        ai_sent_acc = (valid_results['ai_sentiment'] == valid_gt['sentiment']).mean() * 100
        ai_topic_acc = (valid_results['ai_topic'] == valid_gt['topic']).mean() * 100

    # 傳統方法準確率 (假設 A-2 總是會執行)
    trad_sent_acc = (df_results['sentiment'] == df_gt['sentiment']).mean() * 100
    trad_topic_acc = (df_results['topic'] == df_gt['topic']).mean() * 100

    # C-2.3: 輸出比較表格
    comparison_data = {
        '方法': ['A-2: 規則式 (Traditional)', 'B-2: AI (Gemini)'],
        '情感分析準確率 (%)': [f"{trad_sent_acc:.2f}", f"{ai_sent_acc:.2f}"],
        '主題分類準確率 (%)': [f"{trad_topic_acc:.2f}", f"{ai_topic_acc:.2f}"]
    }
    df_comp = pd.DataFrame(comparison_data)
    print("\n--- 分類任務準確率比較 ---")
    print(df_comp.to_string(index=False))

    # C-2.4: 準備回傳的 metrics
    metrics = {
        'A2_accuracy': {
            'sentiment_accuracy_%': trad_sent_acc,
            'topic_accuracy_%': trad_topic_acc
        },
        'B2_accuracy': {
            'sentiment_accuracy_%': ai_sent_acc,
            'topic_accuracy_%': ai_topic_acc
        }
    }
    return metrics

# --- Part C-3: 自動摘要比較 ---

def run_part_c3():
    """
    執行 C-3: 提示使用者手動比較摘要
    """
    print("\n" + "="*30)
    print("  執行 Part C-3: 自動摘要比較 (質化)")
    print("="*30)
    
    summary_path = "results/summarization_comparison.txt"

# # --- 主執行區塊 ---

def load_metrics(path):
    """ 載入 Part A+B 的效能 JSON """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {path}。請先執行 traditional_methods.py。")
        return {}
    except json.JSONDecodeError:
        print(f"錯誤: {path} 檔案格式錯誤。")
        return {}

def save_metrics(path, data):
    """ 儲存更新後的效能 JSON """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n效能指標已更新並儲存至 {path}")
    except Exception as e:
        print(f"錯誤: 儲存 metrics JSON 失敗: {e}")

def main():
    """ 主函式, 執行所有 C 部分任務 """
    
    print("--- 正在啟動 comparison.py (Part C) ---")
    
    # 0. 確保 'results' 資料夾存在
    ensure_results_dir()
    
    # 1. 載入 Part A+B 的效能數據
    perf_path = "results/performance_metrics.json"
    final_metrics = load_metrics(perf_path)
    if not final_metrics:
        print("效能指標檔案載入失敗, 程式中止。")
        return
        
    # 2. 載入 C-1 所需的測試資料
    documents_c1 = [
        "人工智慧正在改變世界,機器學習是其核心技術",
        "深度學習推動了人工智慧的發展,特別是在圖像識別領域",
        "今天天氣很好,適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康,每天都應該保持運動習慣"
    ]
    doc_labels_c1 = [f"文件{i}" for i in range(len(documents_c1))]
    
    # 3. 執行 C-1 (相似度)
    metrics_c1 = run_part_c1(documents_c1, doc_labels_c1)
    # 將 C-1 的結果 (C1_A..., C1_B...) 新增到 metrics 頂層
    final_metrics.update(metrics_c1)
    
    # 4. 執行 C-2 (分類)
    metrics_c2 = run_part_c2()
    # 將 C-2 的結果 (A2_accuracy, B2_accuracy) 更新到 metrics 的 A, B 內部
    if 'A2_accuracy' in metrics_c2 and 'traditional' in final_metrics and 'A2_classify' in final_metrics['traditional']:
        final_metrics['traditional']['A2_classify'].update(metrics_c2['A2_accuracy'])
    if 'B2_accuracy' in metrics_c2 and 'modern' in final_metrics and 'B2_classify_ai' in final_metrics['modern']:
        final_metrics['modern']['B2_classify_ai'].update(metrics_c2['B2_accuracy'])

    # 5. 執行 C-3 (摘要)
    run_part_c3()
    
    # 6. 回存更新後的 metrics
    save_metrics(perf_path, final_metrics)
    
    # 7. 建立 README 和 預填的 report.md
    # create_placeholder_files(final_metrics)
    
    print("\n--- Part C (comparison.py) 執行完畢 ---")
    # print("請檢查 'results/' 資料夾中的所有輸出, 並開啟 'report.md' 完成您的分析報告。")

if __name__ == "__main__":
    main()