# 作業2 - 傳統 vs 現代文本處理

## 學號
114971002
## 檔案結構
```
114971002_hw2/
|-- traditional_methods.py  # Part A 實作
|-- modern_methods.py       # Part B 實作
|-- comparison.py           # Part C 實作 (本檔案)
|-- report.md               # Part C 分析報告 (已預填模板)
|-- requirements.txt        # Python 套件
|-- README.md               # 執行說明 (本檔案)
+-- results/                # 存放所有輸出結果
    |-- tfidf_similarity_matrix.png     (A-1 輸出)
    |-- classification_results.csv      (A-2 + B-2 輸出)
    |-- summarization_comparison.txt    (A-3 + B-3 輸出)
    |-- performance_metrics.json        (A+B+C 輸出)
    +-- similarity_comparison.png       (C-1 輸出)
```

## 執行說明

1.  **安裝必要套件**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **設定 Google API 金鑰**:
    您必須擁有 Google AI Studio 的 API 金鑰。

    (Windows PowerShell)
    ```powershell
    $env:GOOGLE_API_KEY="您的API金鑰"
    ```
    (macOS / Linux)
    ```bash
    export GOOGLE_API_KEY='您的API金鑰'
    ```

3.  **依序執行程式**:

    a. **執行 Part A**:
       ```bash
       python ./traditional_methods.py
       ```
       *這會建立 `results/` 資料夾並產生 A-1, A-2, A-3 的基礎結果。*

    b. **執行 Part B**:
       ```bash
       python ./modern_methods.py
       ```
       *這會使用 Gemini API 更新 Part A 的檔案 (加入 B-2, B-3 結果)。*

    c. **執行 Part C**:
       ```bash
       python ./comparison.py
       ```
       *這會執行 C-1 和 C-2 的比較, 產生 `similarity_comparison.png` 並更新 `performance_metrics.json`
