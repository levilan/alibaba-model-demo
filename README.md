# Alibaba Cloud AI Model Testing Platform

這是一個基於 FastAPI (Python) 與原生 JavaScript/HTML 建構的 AI 模型測試平台，支援阿里雲 (DashScope) 以及 MuleRouter (MuleAI) 的模型 API。

## 專案架構

- **`app.py`**: FastAPI 後端主程式（負責 API 端點、模型清單管理、路由轉發、非同步串流處理）。
- **`templates/index.html`**: 前端主頁面（UI 介面結構、各功能分頁、登入畫面）。
- **`static/js/app.js`**: 前端核心邏輯（API 呼叫、狀態管理、UI 互動、動態渲染與輪詢機制）。
- **`static/css/style.css`**: 全局視覺樣式。

## 核心功能模組

1. **登入與驗證機制**:
   使用 API Key 進行登入。前端呼叫 `/login` 端點，成功後自 `/api/models` 取得最新模型清單，自動隱藏登入框並展示主應用程式 (`mainApp`)。
2. **文字生成 (Text Generation)**: 
   支援 Qwen 等大語言模型，並具備 SSE (Server-Sent Events) 即時串流輸出與 Thinking 過程顯示功能。
3. **圖片與影片生成**: 
   支援非同步任務提交。提交後取得 `task_id`，前端會自動輪詢狀態，並在完成後直接於網頁預覽及下載。
4. **MuleAI (進階影片生成)**:
   提供專屬頁籤，支援額外輸入 MuleAI API Key (`X-MuleAI-API-Key`)，目前綁定 `wan2.7-i2v-spicy` (圖生影片) 模型。支援上傳首幀圖片、解析度與時長設定，直接將遠端生成結果渲染為 `<video>` 播放器。

## 近期更新內容

- **新增 MuleAI (I2V) 整合**：完美對接 MuleRouter 平台，支援 `wan2.7-i2v-spicy` 模型，支援圖片上傳與 FormData 傳輸。
- **解決跨網域下載 (CORS) 問題**：影片生成不再下載佔用伺服器空間，改為直接提供遠端網址。點擊下載會以 `target="_blank"` 開啟，避開瀏覽器限制。
- **優化非同步串流 (Streaming)**：後端從同步的 `OpenAI` 客戶端升級為 `AsyncOpenAI`，徹底解決了 FastAPI 事件迴圈阻塞的問題，實現真正的文字逐字渲染。
- **UI 結構重構**：修復了 `mainApp` 容器閉合異常導致的登入後畫面空白問題。

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
- **更新程式碼後重新建置**：
  ```bash
  docker-compose build && docker-compose up -d
  ```

## 開發者指南：如何新增模型
1. 打開 `app.py`。
2. 找到 `MODELS` 字典變數。
3. 依照現有格式，在對應的陣列 (例如 `"text"`, `"image"`, `"video"`) 中加入新的字典物件（需包含 `id`, `name`, `group`, `desc` 等欄位）。
4. 儲存檔案後，執行 `docker-compose restart ai-model-tester` 重啟後端。前端重新載入後，下拉選單即會自動更新。
