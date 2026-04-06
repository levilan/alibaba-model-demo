# Alibaba Cloud AI Model Testing Platform

一個基於 Flask + DashScope SDK 的 Web 測試平台，用於測試阿里雲（Alibaba Cloud）的 AI 模型，支援文字生成、圖片生成與影片生成。

## 功能概覽

| 分類 | 支援任務 | 代表模型 |
|------|---------|---------|
| 文字生成 | 對話、推理 | Qwen3 Max、Qwen3.6 Plus、DeepSeek V3.2 |
| 圖片生成 | 文生圖 (T2I) | 千問圖像 2.0 Pro、萬相 2.6 T2I、Z-Image Turbo |
| 圖片生成 | 圖像編輯 (I2I) | 萬相 2.7 Image Pro、千問圖像編輯 Max |
| 影片生成 | 文生影片 (T2V) | 萬相 2.7 T2V（支援自動配音）、萬相 2.6 T2V |
| 影片生成 | 圖生影片 (I2V) | 萬相 2.7 I2V、萬相 2.6 I2V |
| 影片生成 | 參考生影片 (R2V) | 萬相 2.7 R2V、萬相 2.6 R2V |

### 主要特色

- **API Key 驗證**：使用者透過 DashScope API Key 登入，所有 AI 呼叫均使用各自的 Key
- **SSE 串流輸出**：文字生成即時串流顯示，支援 Thinking 模式
- **非同步影片任務**：影片任務提交後立即返回 task_id，前端自動輪詢進度
- **動態計時顯示**：圖片生成有 Loading 計時器，影片任務卡片即時顯示已等待時間
- **模型參數動態配置**：依所選模型自動調整尺寸選項、最大張數、時長範圍、配音開關
- **本地預覽與下載**：生成完成後圖片 / 影片均儲存至本地，可直接預覽與下載

## 快速啟動

### 前置需求

- [Docker](https://www.docker.com/get-started) & Docker Compose
- [DashScope API Key](https://dashscope.aliyun.com/)（格式：`sk-...`）

### 安裝與啟動

```bash
git clone https://github.com/levilan/alibaba-model-demo.git
cd alibaba-model-demo

docker compose up -d --build
```

開啟瀏覽器前往 `http://localhost:5050`，輸入 DashScope API Key 後即可使用。

### 常用指令

```bash
# 查看 log
docker compose logs -f

# 停止
docker compose down

# 重新建構（代碼更新後）
docker compose up -d --build
```

> 生成的圖片與影片會掛載至本機 `outputs/` 目錄，容器重啟後資料不會遺失。

## 專案結構

```
.
├── app.py              # Flask 後端（API 路由、DashScope SDK 呼叫）
├── requirements.txt    # Python 依賴
├── Dockerfile          # Docker 映像定義
├── docker-compose.yml  # 服務編排（含 volume 掛載）
├── templates/
│   └── index.html      # 前端 HTML（單頁應用）
└── static/
    ├── css/style.css   # 樣式（Dark Mode 設計）
    └── js/app.js       # 前端邏輯（SSE 串流、任務輪詢、動態 UI）
```

## 支援模型清單

### 文字模型

| 模型 ID | 說明 | 思考模式 |
|---------|------|---------|
| `qwen3-max` | 最強推理，262K context | ✓ |
| `qwen3.6-plus` | 1M context，性價比最佳 | ✓ |
| `qwen3.5-plus` | 前代均衡模型 | ✓ |
| `qwen3.5-flash` | 速度快、成本低 | ✓ |
| `qwen-flash` | 前代極速模型 | ✓ |
| `deepseek-v3.2` | 深度推理（國際版可用） | — |

### 圖片模型

| 模型 ID | 任務類型 | 說明 |
|---------|---------|------|
| `qwen-image-2.0-pro` | T2I | 文字渲染突出 |
| `qwen-image-2.0` | T2I | 標準文生圖 |
| `wan2.6-t2i` | T2I | 自由選尺寸 |
| `z-image-turbo` | T2I | 輕量級快速生成 |
| `wan2.7-image-pro` | I2I | 多圖融合、風格遷移 |
| `wan2.7-image` | I2I | 標準圖像編輯 |
| `wan2.6-image` | I2I | 前代編輯模型 |
| `qwen-image-edit-max` | I2I | 複雜圖文編輯 |

### 影片模型

| 模型 ID | 任務類型 | 說明 |
|---------|---------|------|
| `wan2.7-t2v` | T2V | 多鏡頭、自動配音 |
| `wan2.6-t2v` | T2V | 前代文生影片 |
| `wan2.7-i2v` | I2V | 首幀生影片 |
| `wan2.6-i2v` | I2V | 前代圖生影片 |
| `wan2.7-r2v` | R2V | 角色形象參考 |
| `wan2.6-r2v` | R2V | 前代參考生影片 |

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/login` | 驗證 DashScope API Key |
| `GET` | `/api/models` | 取得模型清單 |
| `POST` | `/api/text/generate` | 文字生成（SSE 串流） |
| `POST` | `/api/image/generate` | 文生圖 |
| `POST` | `/api/image/edit` | 圖像編輯（FormData + 圖片上傳） |
| `POST` | `/api/video/t2v` | 文生影片（返回 task_id） |
| `POST` | `/api/video/i2v` | 圖生影片（FormData + 圖片上傳） |
| `POST` | `/api/video/r2v` | 參考生影片（FormData + 多檔上傳） |
| `GET` | `/api/video/status/<task_id>` | 查詢影片任務狀態 |
| `GET` | `/outputs/images/<filename>` | 取得生成圖片 |
| `GET` | `/outputs/videos/<filename>` | 取得生成影片 |

## 技術架構

- **後端**：Python / Flask，DashScope SDK，OpenAI SDK（相容模式）
- **前端**：原生 HTML / CSS / JavaScript，無額外框架依賴
- **AI 呼叫**：
  - 文字：OpenAI 相容模式 (`dashscope-intl.aliyuncs.com/compatible-mode/v1`)
  - 圖片：DashScope `ImageGeneration` SDK
  - 影片：DashScope `VideoSynthesis` SDK（`async_call` 非同步模式）
- **檔案上傳**：`file://` 協定，SDK 自動上傳至 DashScope OSS

## 注意事項

- 影片生成通常需要 1～5 分鐘，任務提交後請等待輪詢完成
- R2V（參考生影片）上傳的圖片 / 影片會以 `media` 格式傳送，需使用 `wan2.6-r2v` 以上模型
- 生成的圖片與影片儲存於 `outputs/` 目錄（已列入 `.gitignore`）
- 本平台使用者端直接持有 API Key，適合個人測試環境使用

## License

MIT
