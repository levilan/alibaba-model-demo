# Alibaba Cloud AI Model Testing Platform

一個基於 Flask + DashScope SDK 的 Web 測試平台，用於測試阿里雲（Alibaba Cloud）的 AI 模型，支援文字生成、圖片生成、影片生成與語音模型。

## 功能概覽

| 分類 | 支援任務 | 代表模型 |
|------|---------|---------|
| 文字生成 | 對話、推理 | Qwen3 Max、Qwen3.6 Plus、DeepSeek V3.2 |
| 圖片生成 | 文生圖 (T2I) | 千問圖像 2.0 Pro、萬相 2.6 T2I、Z-Image Turbo |
| 圖片生成 | 圖像編輯 (I2I)，0–9 張參考圖 | 萬相 2.7 Image Pro、千問圖像編輯 Max |
| 影片生成 | 文生影片 (T2V)，2–15 秒 | 萬相 2.7 T2V、HappyHorse T2V |
| 影片生成 | 圖生影片 (I2V)，2–15 秒 | 萬相 2.7 I2V、HappyHorse I2V |
| 影片生成 | 參考生影片 (R2V)，2–15 秒 | 萬相 2.7 R2V、HappyHorse R2V |
| 影片生成 | 視頻編輯 (VideoEdit) | 萬相 2.7 VideoEdit、HappyHorse Video Edit |
| 語音模型 | 語音識別 (ASR) | Qwen3 ASR Flash、Fun-ASR 多語言 |
| 語音模型 | 語音合成 (TTS) | Qwen TTS（HTTP 同步） |

### 主要特色

- **API Key 驗證**：使用者透過 DashScope API Key 登入，所有 AI 呼叫均使用各自的 Key
- **SSE 串流輸出**：文字生成即時串流顯示，支援 Thinking 模式
- **非同步影片任務**：影片任務提交後立即返回 task_id，前端自動輪詢進度
- **動態時長範圍**：依所選模型自動調整時長 slider（Wan 2\~15 秒、HappyHorse 3\~15 秒）
- **多圖參考上傳**：圖像編輯支援 0–9 張參考圖，縮圖網格預覽，可逐張刪除
- **語音識別 & 合成**：上傳音訊辨識文字；輸入文字合成語音並提供下載
- **本地預覽與下載**：圖片 / 影片 / 音訊均儲存至本地，可直接預覽與下載

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

> 生成的圖片、影片與音訊會掛載至本機 `outputs/` 目錄，容器重啟後資料不會遺失。

### 使用 GHCR 映像（無需本地建構）

```bash
docker pull ghcr.io/levilan/alibaba-model-demo:latest

docker run -d \
  -p 5050:5050 \
  -v $(pwd)/outputs:/app/outputs \
  ghcr.io/levilan/alibaba-model-demo:latest
```

## 專案結構

```
.
├── app.py                    # Flask 後端（API 路由、DashScope SDK 呼叫）
├── requirements.txt          # Python 依賴
├── Dockerfile                # Docker 映像定義
├── docker-compose.yml        # 服務編排（含 volume 掛載）
├── .github/
│   └── workflows/
│       └── docker.yml        # CI：push main → 自動建構並推送 Docker 映像
├── docs/
│   └── bailian-models.md     # 百炼控制台模型參考文件
├── templates/
│   └── index.html            # 前端 HTML（單頁應用）
└── static/
    ├── css/style.css         # 樣式
    └── js/app.js             # 前端邏輯（SSE 串流、任務輪詢、動態 UI）
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
| `wan2.7-image-pro` | I2I | 多圖融合、風格遷移（0–9 張參考圖） |
| `wan2.7-image` | I2I | 標準圖像編輯（0–9 張參考圖） |
| `wan2.6-image` | I2I | 前代編輯模型 |
| `qwen-image-edit-max` | I2I | 複雜圖文編輯 |

### 影片模型

| 模型 ID | 任務類型 | 時長 | 說明 |
|---------|---------|------|------|
| `wan2.7-t2v` | T2V | 2–15 s | 多鏡頭、自動配音 |
| `wan2.6-t2v` | T2V | 2–15 s | 前代文生影片 |
| `happyhorse-1.0-t2v` | T2V | 3–15 s | 高還原度文生影片 |
| `wan2.7-i2v` | I2V | 2–15 s | 首幀生影片 |
| `wan2.6-i2v` | I2V | 2–15 s | 前代圖生影片 |
| `happyhorse-1.0-i2v` | I2V | 3–15 s | 高還原度圖生影片 |
| `wan2.7-r2v` | R2V | 2–15 s | 角色形象參考 |
| `wan2.6-r2v` | R2V | 2–15 s | 前代參考生影片 |
| `happyhorse-1.0-r2v` | R2V | 3–15 s | 多圖參考（最多 9 張） |
| `wan2.7-videoedit` | VideoEdit | 2–15 s | 文字/參考圖驅動編輯 |
| `happyhorse-1.0-video-edit` | VideoEdit | 3–15 s | 自然語言指令（最多 5 張參考圖） |

### 語音模型

| 模型 ID | 任務類型 | 說明 |
|---------|---------|------|
| `qwen3-asr-flash` | ASR | 新一代極速識別，多語言 |
| `paraformer-v2` | ASR | 高精度普通話識別 |
| `sensevoice-v1` | ASR | 中/英/日/韓/粵多語言 |
| `qwen-tts` | TTS | HTTP 同步語音合成 |

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/login` | 驗證 DashScope API Key |
| `GET` | `/api/models` | 取得模型清單與 TTS 音色 |
| `POST` | `/api/text/generate` | 文字生成（SSE 串流） |
| `POST` | `/api/image/generate` | 文生圖 |
| `POST` | `/api/image/edit` | 圖像編輯（FormData，image_1…image_9） |
| `POST` | `/api/video/t2v` | 文生影片（返回 task_id） |
| `POST` | `/api/video/i2v` | 圖生影片（FormData + 圖片上傳） |
| `POST` | `/api/video/r2v` | 參考生影片（FormData + 多檔上傳） |
| `POST` | `/api/video/vedit` | 視頻編輯（FormData + 影片 + 參考圖） |
| `GET` | `/api/video/status/<task_id>` | 查詢影片任務狀態 |
| `POST` | `/api/voice/asr` | 語音識別（FormData + 音訊檔） |
| `POST` | `/api/voice/tts` | 語音合成（返回音訊 URL） |
| `GET` | `/outputs/images/<filename>` | 取得生成圖片 |
| `GET` | `/outputs/videos/<filename>` | 取得生成影片 |
| `GET` | `/outputs/audio/<filename>` | 取得合成音訊 |

## 技術架構

- **後端**：Python / Flask，DashScope SDK，OpenAI SDK（相容模式）
- **前端**：原生 HTML / CSS / JavaScript，無額外框架依賴
- **AI 呼叫**：
  - 文字：OpenAI 相容模式 (`dashscope-intl.aliyuncs.com/compatible-mode/v1`)
  - 圖片：DashScope `ImageGeneration` SDK
  - 影片：DashScope `VideoSynthesis` SDK（`async_call` 非同步模式）
  - 語音識別：DashScope `Recognition` SDK（instance call）
  - 語音合成：DashScope `SpeechSynthesizer` v1（HTTP 同步）
- **檔案上傳**：`file://` 協定，SDK 自動上傳至 DashScope OSS

## 注意事項

- 影片生成通常需要 1～5 分鐘，任務提交後請等待輪詢完成
- 圖像編輯支援 0–9 張參考圖，萬相 2.7 系列效果最佳
- HappyHorse 系列（I2V / T2V / R2V / VideoEdit）為 2026 年 4 月新上線模型
- 生成的圖片、影片與音訊儲存於 `outputs/` 目錄（已列入 `.gitignore`）
- 本平台使用者端直接持有 API Key，適合個人測試環境使用

## License

MIT
