# Alibaba Cloud AI Model Testing Platform

這是一個基於 FastAPI (Python) 與原生 JavaScript/HTML 建構的 AI 模型測試平台，支援阿里雲 (DashScope) 以及 MuleRouter (MuleAI) 的模型 API。

## 專案架構

- **`app.py`**: FastAPI 後端主程式（負責 API 端點、模型清單管理、路由轉發、非同步串流處理與檔案下載）。
- **`templates/index.html`**: 前端主頁面（UI 介面結構、各功能分頁、登入畫面）。
- **`static/js/app.js`**: 前端核心邏輯（API 呼叫、狀態管理、UI 互動、動態渲染、歷史紀錄與輪詢機制）。
- **`static/css/style.css`**: 全局視覺樣式。

## 核心功能模組

1. **登入與驗證機制**:
   使用 API Key 進行登入。前端呼叫 `/login` 端點，成功後自 `/api/models` 取得最新模型清單，自動隱藏登入框並展示主應用程式 (`mainApp`)。
2. **文字生成 (Text Generation)**: 
   支援 Qwen 等大語言模型，並具備 SSE (Server-Sent Events) 即時串流輸出與 Thinking 過程顯示功能。支援 `Ctrl+Enter` 快速發送。
3. **圖片與影片生成**: 
   支援非同步任務提交。提交後取得 `task_id`，前端會自動輪詢狀態。生成完畢後，後端會自動將檔案下載至伺服器本機的 `outputs/` 資料夾，並於網頁預覽及供使用者下載。
4. **MuleAI (進階圖生影片)**:
   提供專屬頁籤，支援額外輸入 MuleAI API Key (`X-MuleAI-API-Key`)，目前綁定 `wan2.7-i2v-spicy` (圖生影片) 模型。支援上傳首幀參考圖片、解析度與時長設定，直接將遠端生成結果渲染為 `<video>` 播放器。

## 近期重要更新內容

- **歷史紀錄保存 (Local History)**：前端導入 `localStorage` 機制，使用者生成的圖片與影片任務會保留在瀏覽器中，重新整理網頁後自動還原歷史卡片，不再遺失生成進度。
- **本機非同步下載機制**：全面重構後端的檔案下載 (`_download_image` 與 `_download_video`)，改用 `httpx.AsyncClient` 非同步串流下載，確保大檔案影片存入本地端時不阻塞文字生成的 SSE 串流。
- **MuleAI (I2V) 深度整合**：修復了 MuleRouter 的 401 錯誤攔截邏輯（填錯 MuleAI Key 時不再強制登出），並完美對接圖生影片的 FormData 上傳架構。
- **語音模型維護中**：語音模型 (Voice) 相關功能標示為修復中。

## 部署與啟動方式

本專案強烈建議使用 Docker 進行環境隔離與快速部署。

### 前置需求
- 已安裝 [Docker](https://docs.docker.com/get-docker/) 與 [Docker Compose](https://docs.docker.com/compose/install/)
- 準備好您的 DashScope API Key（格式通常為 `sk-...`）

### 啟動步驟

1. **複製專案到本地端**：
   ```bash
   git clone <你的專案網址>
   cd ai-model-tester
   ```

2. **使用 Docker Compose 建置並啟動服務**：
   ```bash
   docker-compose up -d --build
   ```

3. **開始使用**：
   - 開啟您的瀏覽器，前往：`http://localhost:5050`
   - 在登入畫面輸入您的 API Key 即可開始操作。

### 常用維護指令

- **查看運行日誌**：
  ```bash
  docker logs -f ai-model-tester-ai-model-tester-1
  ```
- **關閉服務**：
  ```bash
  docker-compose down
  ```
- **徹底清除快取並重新建置 (強制更新)**：
  ```bash
  docker-compose build --no-cache && docker-compose up -d
  ```

## 開發者指南：如何新增模型
1. 打開 `app.py`。
2. 找到 `MODELS` 字典變數。
3. 依照現有格式，在對應的陣列 (例如 `"text"`, `"image"`, `"video"`) 中加入新的字典物件（需包含 `id`, `name`, `group`, `desc` 等欄位）。
4. 儲存檔案後，執行 `docker-compose restart ai-model-tester` 重啟後端。前端重新載入後，下拉選單即會自動更新。
