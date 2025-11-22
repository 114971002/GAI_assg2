#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
modern_methods.py
作業2 - Part B (現代方法實作)

此檔案包含 Part B 的所有實作 (使用 Google Gemini API):
B-1: AI 文本相似度 (get_embedding_gemini)
B-2: AI 文本分類 (classify_text_batch_gemini)
B-3: AI 自動摘要 (summarize_text_gemini)
"""

# --- 必要的函式庫 ---
import os
import json
import time
import pandas as pd
from tqdm import tqdm

# 第三方函式庫 (請確保已安裝)
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Part B-1: AI 文本相似度 (Embedding) ---

def get_embedding_gemini(text, model="models/text-embedding-004"): # <--- 修正#1: 使用 'text-embedding-004'
    """
    ...
    """
    try:
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"  # <--- 修正#2: 使用 'RETRIEVAL_DOCUMENT'
        )
        return response['embedding']
    except Exception as e:
        print(f"錯誤: 取得 Embedding 失敗 (文本: {text[:20]}...): {e}")
        return None

# --- Part B-2: AI 文本分類 ---

def classify_text_batch_gemini(model, text_list, batch_size=5):
    """
    使用 Gemini API 進行批次文本分類 (使用 JSON Mode)
    
    Args:
        model: 已初始化的 Gemini GenerativeModel
        text_list: 待分類的文本列表
        batch_size: 批次大小 (Gemini API 通常不需要手動批次,
                    但為模擬 PDF 流程並加上進度條, 我們保留此結構)
    
    Returns:
        list: 包含 JSON 物件 (或 None) 的結果列表
    """
    results = []
    
    # Gemini 的 JSON 模式必須在 prompt 中明確指示
    # 我們將 PDF 中的 system prompt 和 user prompt 合併
    json_schema = '{"sentiment": "正面|負面|中性", "topic": "科技|運動|美食|旅遊|娛樂|其他"}'
    
    # 使用 tqdm 顯示進度
    for text in tqdm(text_list, desc="AI 文本分類中"):
        prompt = f"""
你是一個專業的文本分類器。
請根據以下 JSON 格式回覆: {json_schema}

請將以下文本分類:
文本: {text}
"""
        
        try:
            # 設定 JSON 輸出模式
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json"
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Gemini 在 JSON 模式下會直接返回結構化的 JSON 字串
            result_json = json.loads(response.text)
            results.append(result_json)
            
            # 稍微暫停, 避免觸及 API 速率限制
            time.sleep(0.5) 
            
        except (json.JSONDecodeError, google_exceptions.InvalidArgument) as e:
            # 捕獲因為模型不支援 JSON 模式或回傳格式錯誤導致的錯誤
            print(f"錯誤: AI 分類失敗 (文本: {text[:20]}...): {e}")
            if 'response' in locals():
                print(f"  -> Gemini 回應 (非 JSON): {response.text}")
            results.append(None)
        except Exception as e:
            print(f"發生未預期錯誤 (檢查您的模型權限): {e}")
            results.append(None)

    return results

def run_part_b2(model, test_texts):
    """
    執行 Part B-2: 讀取 Part A 結果, 新增 AI 分類欄位
    """
    print("\n" + "="*30)
    print("  執行 Part B-2: AI 文本分類 (Gemini)")
    print("="*30)
    
    csv_path = "results/classification_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {csv_path}。請先執行 traditional_methods.py。")
        return {'time_seconds': 0, 'error': 'File not found'}

    # 取得 'text' 欄位的列表
    texts_to_classify = df['text'].tolist()
    
    start_time = time.time()
    
    # ----------------------------------------------------
    # *** 檢查: 如果傳入的 model 是 None (因為模型載入失敗) ***
    # ----------------------------------------------------
    if model is None:
        print("警告: 傳入的模型為 None。跳過 AI 分類。")
        ai_results = [None] * len(texts_to_classify)
        b2_time = 0
    else:
        # 呼叫 Gemini API 進行分類
        ai_results = classify_text_batch_gemini(model, texts_to_classify)
        end_time = time.time()
        b2_time = end_time - start_time
        print(f"\nAI 分類總時間: {b2_time:.6f} 秒")
    
    # 處理 AI 結果並新增至 DataFrame
    ai_sentiments = []
    ai_topics = []
    
    for res in ai_results:
        if res and isinstance(res, dict):
            ai_sentiments.append(res.get('sentiment', 'N/A'))
            ai_topics.append(res.get('topic', 'N/A'))
        else:
            ai_sentiments.append('Error')
            ai_topics.append('Error')
            
    df['ai_sentiment'] = ai_sentiments
    df['ai_topic'] = ai_topics
    
    # 存回 CSV (覆蓋)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"AI 分類結果已更新至 {csv_path}")
    
    print("\n--- AI 分類結果預覽 ---")
    print(df[['text', 'ai_sentiment', 'ai_topic']].to_string())
    
    return {'time_seconds': b2_time}


# --- Part B-3: AI 自動摘要 ---

def summarize_text_gemini(model, article):
    """
    使用 Gemini API 生成摘要
    
    Args:
        model: 已初始化的 Gemini GenerativeModel
        article (str): 完整文章
    Returns:
        str: AI 生成的摘要
    """
    
    # PDF 中的 Prompt
    prompt = f"""
你是一個專業的摘要產生器。
請將以下文章濃縮成 100 字左右的摘要, 保留核心觀點。

文章:
{article}

摘要:
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"錯誤: AI 摘要生成失敗 (檢查您的模型權限): {e}")
        return "摘要生成失敗"

def run_part_b3(model, article):
    """
    執行 Part B-3: 生成 AI 摘要並附加到 Part A 的檔案中
    """
    print("\n" + "="*30)
    print("  執行 Part B-3: AI 自動摘要 (Gemini)")
    print("="*30)
    
    summary_path = "results/summarization_comparison.txt"
    
    start_time = time.time()
    # ----------------------------------------------------
    # *** 檢查: 如果傳入的 model 是 None (因為模型載入失敗) ***
    # ----------------------------------------------------
    if model is None:
        print("警告: 傳入的模型為 None。跳過 AI 摘要。")
        ai_summary = "摘要生成失敗 (模型無法載入)"
        b3_time = 0
    else:
        ai_summary = summarize_text_gemini(model, article)
        end_time = time.time()
        b3_time = end_time - start_time
    
    print(f"AI 摘要生成時間: {b3_time:.6f} 秒")
    
    print("\n--- AI 生成的摘要 ---")
    print(ai_summary)
    
    # 將 AI 摘要附加到 Part A 已建立的檔案中
    try:
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write("\n\n--- Modern AI Summary (Gemini) ---\n")
            f.write(ai_summary)
        print(f"\nAI 摘要已附加至 {summary_path}")
    except FileNotFoundError:
        print(f"錯誤: 找不到 {summary_path}。請先執行 traditional_methods.py。")
        return {'time_seconds': b3_time, 'error': 'File not found'}
        
    return {'time_seconds': b3_time, 'summary': ai_summary}


# --- 主執行區塊 ---

def ensure_results_dir():
    """ 確保 'results' 資料夾存在 """
    if not os.path.exists('results'):
        os.makedirs('results')
        print("已建立 'results' 資料夾。")

def main():
    """ 主函式, 執行 Part B 任務 """
    
    print("--- 正在啟動 modern_methods.py ---")
    
    # 0. 檢查 API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("="*50)
        print("!! 錯誤: 找不到環境變數 'GOOGLE_API_KEY' !!")
        print("請先設定您的 Google API 金鑰才能執行此程式。")
        print("="*50)
        return
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"API 金鑰設定失敗: {e}")
        return

    # 確保 'results' 資料夾存在
    ensure_results_dir()
    
    # 1. 載入 Part A-2 測試資料 (用於 B-2)
    # (同 traditional_methods.py, 確保此腳本可獨立測試)
    test_texts_a2 = [
        "這家餐廳的牛肉麵真的太好吃了,湯頭濃郁,麵條Q彈,下次一定再來!",
        "最新的AI技術突破讓人驚艷,深度學習模型的表現越來越好",
        "這部電影劇情空洞,演技糟糕,完全是浪費時間",
        "每天慢跑5公里,配合適當的重訓,體能進步很多"
    ]
    
    # 2. 載入 Part A-3 測試文章 (用於 B-3)
    # (同 traditional_methods.py)
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
    
    # 3. 初始化 Gemini 模型 (*** 根據 check_models.py 的結果修正 ***)
    
    model_b2 = None
    model_b3 = None

    # B-2 (分類) 需要 JSON 模式, 嘗試 models/gemini-pro-latest
    try:
        # *** 修正 #1: 使用您帳號可用的 'models/gemini-pro-latest' ***
        model_b2 = genai.GenerativeModel('models/gemini-pro-latest')
        # 嘗試發送一個簡單請求來 "預檢" 權限 (包含 JSON 模式)
        model_b2.generate_content(
            "test", 
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        print("模型 'models/gemini-pro-latest' (B-2 分類) 載入成功。")
    except Exception as e:
        print(f"警告: 載入 'models/gemini-pro-latest' 失敗: {e}")
        print("     -> Part B-2 AI 分類將被跳過。")
        model_b2 = None

    # B-3 (摘要) 使用 models/gemini-flash-latest
    try:
        # *** 修正 #2: 使用您帳號可用的 'models/gemini-flash-latest' ***
        model_b3 = genai.GenerativeModel('models/gemini-flash-latest')
        # Try to send a simple request to "pre-flight" permissions
        model_b3.generate_content("test")
        print("模型 'models/gemini-flash-latest' (B-3 摘要) 載入成功。")
    except Exception as e:
        print(f"警告: 載入 'models/gemini-flash-latest' 失敗: {e}")
        print("     -> Part B-3 AI 摘要將被跳過。")
        model_b3 = None
    
    # 4. 執行 Part B-2 和 B-3
    perf_metrics = {}
    perf_metrics['B2_classify_ai'] = run_part_b2(model_b2, test_texts_a2)
    perf_metrics['B3_summarize_ai'] = run_part_b3(model_b3, article_a3)
    
    # 5. 更新效能指標
    perf_path = "results/performance_metrics.json"
    
    try:
        # 讀取 Part A 產生的 JSON
        with open(perf_path, 'r', encoding='utf-8') as f:
            final_metrics = json.load(f)
            
        # 新增 'modern' 欄位
        final_metrics['modern'] = perf_metrics
        
        # 寫回 JSON
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        print(f"\n現代方法效能指標已更新至 {perf_path}")
        
    except FileNotFoundError:
        print(f"錯誤: 找不到 {perf_path}。請先執行 traditional_methods.py。")
    except json.JSONDecodeError:
        print(f"錯誤: {perf_path} 檔案內容不是有效的 JSON。")

    print("\nPart B (modern_methods.py) 執行完畢。")


if __name__ == "__main__":
    main()