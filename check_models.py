import os
import google.generativeai as genai

print("--- 正在檢查可用的 Gemini 模型 ---")

# 1. 讀取您設定的 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("!! 錯誤: 找不到環境變數 'GOOGLE_API_KEY' !!")
    print("請先在同一個終端機視窗中設定金鑰:")
    print(r'  $env:GOOGLE_API_KEY="您的金鑰"')
    exit()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"API 金鑰設定失敗: {e}")
    exit()

print("API Key 讀取成功。正在查詢模型列表...")
print("="*30)

# 2. 迭代並印出所有可用的模型
try:
    count = 0
    for m in genai.list_models():
        # 我們只關心 'generateContent' (生成式) 和 'embedContent' (Embedding)
        if 'generateContent' in m.supported_generation_methods or 'embedContent' in m.supported_generation_methods:
            print(f"模型名稱 (Model Name): {m.name}")
            print(f"  支援的方法: {m.supported_generation_methods}")
            print(f"  顯示名稱: {m.display_name}")
            print(f"  描述: {m.description[:70]}...") # 只印出前70字
            print("-"*20)
            count += 1
    
    if count == 0:
        print("\n!! 查詢成功，但未找到任何可用的模型 !!")
        print("這表示您的帳號尚未被授權使用任何 Gemini 模型。")
        print("請檢查您的 Google AI Studio 或 Google Cloud 專案設定。")
    else:
        print(f"\n查詢完畢。共找到 {count} 個可用模型。")

except Exception as e:
    print(f"\n!! 查詢模型時發生錯誤 !!: {e}")
    print("這可能表示您的 API 金鑰無效，或網路連線有問題。")

print("="*30)