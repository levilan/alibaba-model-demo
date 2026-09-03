# NenAI Playground

NenAI（nen.com.tw）模型測試平台。只要一把 NenAI API Key，就能在瀏覽器裡直接測試文字、圖片、影片、語音與 NenAI Spicy 擴充模型——不用寫程式、不用分別申請各家供應商的帳號。

後端是 FastAPI 單檔應用（`app.py`），前端是原生 JavaScript，沒有資料庫、沒有 build 流程，開箱即用。

> 🎨 介面美學規範見 [`DESIGN-AESTHETIC.md`](DESIGN-AESTHETIC.md)——與 NEN 官網同一套
> 色料與紀律。目前只是規範、尚未實作：現有的阿里雲 demo 配色維持原狀。

> 工程細節、參數驗證紀錄與踩坑歷史不在本文件——見 `update.md`（更新紀錄）與 `memory.md`（跨專案協作與驗證原則）。

---

## 產品功能

### 主測試台（`/`）

以頁籤切換的網頁測試介面：

| 頁籤 | 功能 |
|------|------|
| 文字生成 | SSE 串流輸出、思考模式／推理強度控制、多輪對話記憶、視覺模型圖片輸入 |
| 圖片生成 | 文生圖 (T2I) 與圖像編輯 (I2I)，支援多張參考圖、自訂尺寸、點擊放大預覽 |
| 影片生成 | 文生影片／圖生影片／參考生影片／視頻編輯／動作動畫，即時輪詢進度、配音選項 |
| NenAI Spicy | Wan 2.7 I2V Spicy、Z-Image Spicy、圖像編輯 Spicy、圖像換臉 |
| 語音與音樂 | 語音辨識（ASR，含串流）、語音合成（TTS）、即時語音對話（Realtime）、音樂生成（Lyria） |

貫穿各頁籤的輔助功能：

- **參考單價**：每個模型選單旁顯示換算後的參考價格（token 計費顯示「輸入→輸出每百萬 token 美金」、按次計費顯示「每次呼叫美金」），資料來自網關計費表。僅供參考，實際扣款以 NenAI 後台為準。
- **本次花費**：header 徽章累加這個瀏覽器分頁裡所有呼叫的估計花費，重新整理歸零。
- **查看實際請求**：生成完成後可展開送給網關的端點與完整 body，一鍵複製 cURL。金鑰不顯示（固定呈現為 `Bearer $NENAI_API_KEY`），base64 內容換成長度摘要。
- **剪貼簿貼圖**：任何分頁按 Ctrl+V／⌘V，剪貼簿裡的圖片會自動放進當前頁面對應的圖片欄位。
- **使用者統計**：每次 API 呼叫記一行（時間、使用者雜湊、端點、成功與否、耗時）到雲端物件儲存；不記 prompt、生成結果與 IP。報表用 `scripts/usage_stats.py` 在本機產生 HTML，沒有網頁後台。

### AI Canvas（`/canvas`）

節點式視覺化畫布（類似 ComfyUI），基於 [litegraph.js](https://github.com/jagenjo/litegraph.js)，用拖拉連線的方式組合多個模型呼叫：

| 節點 | 功能 |
|---|---|
| 文字 | 手動輸入 prompt 或呼叫文字模型生成；接上圖片輸入可改做圖片分析 |
| 圖片 | 文生圖；接上參考圖自動切換為圖像編輯 |
| 影片 | 依連接的圖片組合自動切換 t2v／i2v（首尾幀）／r2v（參考圖） |
| 圖像編輯 | 需連接一張來源圖片 |
| 語音 TTS／語音辨識 ASR | 文字轉語音、語音轉文字，可互相串接組成閉環 |
| MuleAI Spicy | 對應 NenAI Spicy 四個模型，依模型動態切換輸入與輸出型別 |
| 上傳圖片／上傳影片 | 把本機檔案帶進畫布；影片可接「來源影片」或 Spicy 節點的「參考影片」（例如 Blender 輸出的灰模影片，15 秒以內） |
| 姿勢 | 拖曳關節編出 OpenPose 風格骨架圖，當圖片節點的參考圖 |
| 標記 | 在圖片上畫筆刷／箭頭／編號，輸出畫好記號的圖當首幀圖或參考圖，提示詞裡說明記號的意思來指引動作 |
| 3D 灰模 | 用文字描述搭粗塊場景（可由提示詞生成），跑一段運鏡輸出灰模影片當運鏡參考；支援會動的量體、人形假人、自訂相機路徑；可匯出場景檔給 Blender 腳本 |

支援一鍵依拓樸順序執行整張圖、內建 8 個常用節點組合範本、選到模型時彈出模型特性介紹。畫布狀態自動存在瀏覽器本機。

**與 Blender 銜接**：`scripts/blender_greybox_export.py` 可以（a）把你自己的 `.blend` 場景用 Workbench 灰階算成符合參考影片規格的 mp4（832×480、24 fps、H.264、上限 15 秒），或（b）把灰模節點「匯出場景檔」下載的 `greybox-scene.txt` 在 Blender 裡重建（含運鏡與位移動畫），細修後再輸出。用法見腳本檔頭。灰模的場景描述語法在瀏覽器（`canvas.js`）與腳本裡各有一份解析器，兩邊要同步改；`tests/` 有鎖住。

### MCP 服務（`/mcp`）

平台同時是一個 MCP（Model Context Protocol）伺服器，讓 Claude 等 AI 助理直接呼叫平台能力。無狀態 JSON-RPC 端點，帶 NenAI API Key 即可使用，提供 7 個工具：

| 工具 | 功能 |
|---|---|
| `nenai_list_models` | 列出可用模型與各自的參數約束 |
| `nenai_generate_image` | 文生圖 |
| `nenai_edit_image` | 圖像編輯（可帶參考圖網址） |
| `nenai_generate_video` | 影片生成（回傳任務 id） |
| `nenai_task_status` | 查詢影片任務進度與結果 |
| `nenai_tts` | 語音合成 |
| `nenai_asr` | 語音辨識 |

參數驗證由模型清單驅動：帶了該模型不支援的參數會得到明確錯誤與合法值列表，而不是被靜默忽略。

---

## 運作邏輯

平台是一層**薄轉譯層**，不是模型服務本身：

1. **金鑰不落地。** 沒有伺服器端 session 與資料庫。瀏覽器登入時呼叫一次 NenAI 驗證金鑰有效，之後金鑰存在瀏覽器 `sessionStorage`，每次請求以 `Authorization: Bearer ...` 帶上，後端原封不動轉發給網關，不驗證也不保存。登入端點有依來源 IP 的失敗次數鎖定，防止暴力嘗試。
2. **請求格式轉譯。** 每個模型家族的上游請求格式不同（OpenAI 相容、DashScope 風格、Gemini 原生、task-based API……），`app.py` 把瀏覽器送來的統一表單轉成對應格式再轉發，並把回應轉回前端統一的協定。
3. **同步與非同步兩種模式。** 文字／圖片／語音走同步（文字為 SSE 串流）；影片與 Spicy 走 task 制——送出任務拿到 `task_id`，前端輪詢狀態端點直到終態。
4. **參數誠實原則。** UI 上出現的每個控制項都對應「送了會生效」的參數；模型不支援的參數整組隱藏，不放沒有作用的選項。各模型的尺寸、張數、時長等約束以 `MODELS` 清單（`GET /api/models`）為單一真實來源，前端選單全部動態產生。
5. **產出儲存與隱私。** 客戶內容（生成的圖片／影片／音訊）預設**不上雲**，寫入容器本機 `outputs/` 目錄（無狀態環境下實例回收即消失，這是刻意的政策）。雲端物件儲存憑證只供使用者統計使用；要讓客戶內容上雲回簽名網址需明確設 `STORE_OUTPUTS=true`。
6. **計價顯示。** 後端代理並快取網關的 `/api/pricing` 計費表 1 小時，前端據此換算參考單價。

---

## 基礎環境架構示意

```
┌───────────────────────────────────────────────┐
│                使用者端                        │
│  主測試台 (/)   AI Canvas (/canvas)   MCP 客戶端│
│        金鑰存瀏覽器，每次請求隨附               │
└──────────────────────┬────────────────────────┘
                       │ HTTPS / WebSocket / JSON-RPC
┌──────────────────────▼────────────────────────┐
│        NenAI Playground（FastAPI，app.py）      │
│  ・MODELS 模型清單（單一真實來源）              │
│  ・各家族請求格式轉譯                          │
│  ・task 輪詢代理（影片／Spicy）                 │
│  ・/ws/omni 即時語音代理（金鑰不進前端握手）     │
│  ・/mcp MCP 伺服器（7 工具）                    │
│  ・/api/pricing 計價快取                       │
└───────┬──────────────────────────┬────────────┘
        │ Bearer <NenAI API Key>   │ 產出檔案
┌───────▼───────────────┐  ┌───────▼───────────────┐
│  NenAI 網關            │  │ 本機 outputs/（預設）   │
│  nen.com.tw           │  │ OSS／S3／GCS（統計；    │
│  統一計費・渠道分發     │  │ 客戶內容需 STORE_       │
└───────┬───────────────┘  │ OUTPUTS=true 才上雲）   │
        │                  └───────────────────────┘
┌───────▼───────────────────────────────────────┐
│  上游模型供應商                                 │
│  阿里（千問／萬相）・OpenAI・Anthropic・Google   │
│  xAI・ByteDance・智譜・DeepSeek・月之暗面 …     │
└───────────────────────────────────────────────┘
```

部署形態：Docker Compose（自架）或容器平台（如 Cloud Run，無狀態、可水平擴展）。

---

## 可用模型列表

模型清單以 `app.py` 的 `MODELS` 為準（`GET /api/models` 取得即時清單），以下為分類總覽。

### 文字生成

| 模型 | 分類 | 思考控制 |
|---|---|---|
| qwen3.8-max／3.7-max／3.6-max-preview | Qwen 旗艦 | enable_thinking |
| qwen3.7-plus／3.6-plus／3.5-plus | Qwen 均衡 | enable_thinking |
| qwen3.6-flash／3.5-flash | Qwen 極速 | enable_thinking |
| qwen3-coder-plus／flash | Qwen 代碼 | — |
| qwen-plus-character | Qwen 角色 | — |
| qwen3-vl-plus／flash | 視覺語言（支援圖片輸入） | — |
| deepseek-v4-pro／v4-flash／v3.2 | DeepSeek | enable_thinking |
| glm-5.2／5.1 | 智譜 GLM | enable_thinking ＋ reasoning_effort 分段 |
| kimi/kimi-k3 | 月之暗面（支援圖片輸入） | 思考模型 |
| dola-seed-sc／2.0-lite／2.0-pro | ByteDance Seed | — |
| dola-seed-2.1-turbo | ByteDance Seed（支援圖片輸入） | reasoning_effort 7 段（none 可關思考） |
| claude-opus-5、claude-opus-4-8～4-1 | Claude Opus | — |
| claude-sonnet-5／4-6／4-5、claude-haiku-4-5、claude-fable-5 | Claude | — |
| gpt-5.6-terra／sol／luna、gpt-5.5～5-mini | GPT | reasoning_effort |
| gemini-3.8-flash 等 10 個型號 | Gemini（走原生 API；3.8-flash 支援圖片輸入） | thinkingConfig |
| grok-4.3（支援圖片輸入）、grok-4-20-*、grok-4-1-fast-* | xAI Grok | 依型號 |

思考控制依家族各有機制：Qwen／DeepSeek／GLM 用布林開關（`enable_thinking`），GPT 用推理強度分段（`reasoning_effort`），Gemini 走原生 `thinkingConfig`，GLM 5.x 兩種都支援。各模型可用的分段選項由 UI 依模型動態顯示，有思考過程回傳的模型會在回答上方顯示可收合的「思考過程」區塊。

### 圖片生成

**文生圖（T2I）**

| 模型 | 分類 | 特點 |
|---|---|---|
| qwen-image-3.0-pro／3.0 | 千問文生圖 | 尺寸為自由的 `寬*高`（面積約束），一次可出 1–6 張 |
| qwen-image-2.0-pro／2.0 | 千問文生圖 | 生成＋編輯融合模型 |
| wan2.7-image-pro／wan2.7-image | 萬相文生圖 | 支援組圖模式（一次生成連貫故事圖組）；pro 支援 2K／4K |
| wan2.6-t2i | 萬相文生圖 | |
| z-image-turbo | Z-Image | |
| MAI-Image-2.5-Pro／2.5／2.5-Flash | MAI Image | 支援自訂寬高（即時驗證與對齊提示） |
| gpt-image-2／1.5 | GPT Image | 支援 quality／background（透明背景）／output_format |
| dola-seedream-5.0-pro／lite | ByteDance Seedream | 尺寸 `WIDTHxHEIGHT`，也接受 2k／3k／4k |
| gemini-3-pro-image、gemini-3.1-flash-image、gemini-2.5-flash-image、gemini-3.1-flash-lite-image | Gemini Image | 以比例（10 種）＋像素等級（1K／2K／4K，依型號）控制輸出 |

**圖像編輯（I2I）**

上表中千問全系、萬相 2.6／2.7、MAI 全系、GPT Image 全系、Seedream 兩型、Gemini Image 全系皆同時支援圖像編輯；參考圖張數上限依模型不同（1～9 張），UI 依模型自動限制。

### 影片生成

| 模型 | 模式 | 特點 |
|---|---|---|
| wan3.0-video | t2v／i2v／r2v／視頻編輯 | all-in-one 模型，最長 30 秒，480P～1080P，支援智能時長（模型自行決定長度） |
| wan3.0-video-prime | t2v | 萬相 3.0 高速版，快速文生影片 |
| wan2.7-t2v／i2v／r2v、wan2.7-videoedit | 各對應模式 | 自動配音 |
| wan2.6-t2v／i2v／i2v-flash／r2v／r2v-flash | 各對應模式 | i2v-flash 可開關配音 |
| wan2.2-animate-mix／move | 動作動畫 | 視頻換人／圖生動作 |
| happyhorse-1.1／1.0（t2v／i2v／r2v／video-edit） | HappyHorse | |
| bytedance-seedance-1.5-pro | t2v／i2v／r2v | 配音可開關 |
| dreamina-seedance-2.5／2.0／2.0-fast | t2v／i2v／r2v | 2.5 支援長片（最長 30 秒）與多素材參考 |
| veo-3.1-generate／fast-generate／lite-generate-001 | t2v | Google Veo，時長 4／6／8 秒，含原生配音 |
| gemini-omni-flash-preview | 多模態影片 | 模型自行決定長度與解析度，含原生配音 |

解析度與時長的可選範圍依模型不同，UI 依 `MODELS` 自動限制選項。

### NenAI Spicy

| 模型 | 分類 | 輸入 |
|---|---|---|
| wan2.7-i2v-spicy | 影片生成 | 文字／圖片 |
| z-image-spicy | 圖片生成 | 文字 prompt |
| qwen-image-edit-spicy | 圖像編輯 | prompt ＋ 來源圖 |
| face-swap | 圖像換臉 | 來源圖 ＋ 換臉參考圖 |

### 語音與音樂

| 模型 | 分類 | 功能 |
|---|---|---|
| qwen3.5-omni-plus／flash-realtime | 即時語音 | WebSocket 雙向串流，可聽可說、看得懂圖片與影片，支援語意斷句與插話 |
| qwen-audio-3.0-realtime-plus／flash | 即時語音 | 純語音即時對話，15 個專屬音色 |
| qwen-audio-3.0-asr-flash（＋串流版） | 語音辨識 | 上傳音檔回逐字稿；串流版 SSE 回傳中間結果 |
| qwen-audio-3.0-tts-plus／flash | 語音合成 | CosyVoice 音色、語氣風格描述（instructions） |
| gemini-2.5-pro-tts、gemini-2.5-flash-tts、gemini-3.1-flash-tts-preview | 語音合成 | Google TTS，共用官方 30 個音色 |
| lyria-3-clip-preview | 音樂生成 | 30 秒音樂片段（MP3），可附圖片作為靈感 |
| lyria-3-pro-preview | 音樂生成 | 完整歌曲約三分鐘（MP3）＋曲式說明，可附圖片靈感 |
| lyria-002 | 音樂生成 | 30 秒高音質音樂（48kHz WAV） |

TTS 音色選單依模型動態產生：Qwen TTS 各型號有專屬音色（[官方音色列表](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list)）、Gemini TTS 共用 [官方 30 個音色](https://ai.google.dev/gemini-api/docs/speech-generation)，皆可留空使用預設。

---

## 快速部署

### 前置需求

- [Docker](https://docs.docker.com/get-docker/) 與 Docker Compose
- NenAI API Key（格式：`sk-...`）

### Docker 部署（推薦）

```bash
git clone <專案網址>
cd nenai-playground
docker compose up -d --build
```

瀏覽器開啟 `http://localhost:5050`，輸入 NenAI API Key 登入。

```bash
docker logs -f nenai-playground-ai-model-tester-1        # 查看日誌
docker compose down                                      # 停止服務
docker compose build --no-cache && docker compose up -d  # 更新程式碼後重建
```

### 本機直接執行

```bash
pip install -r requirements.txt
python app.py            # http://localhost:5050
```

---

## 雲端物件儲存（選用）

**客戶內容預設不上雲**（見「運作邏輯」第 5 點）。設定以下任一組環境變數即可啟用雲端後端——統計一定會用；客戶內容僅在 `STORE_OUTPUTS=true` 時上雲（簽名網址有效期 7 天）：

| 後端 | 需要的環境變數 |
|---|---|
| 阿里雲 OSS | `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` |
| AWS S3 | `S3_ACCESS_KEY_ID`、`S3_SECRET_ACCESS_KEY`、`S3_BUCKET_NAME`（可選 `S3_REGION`、`S3_ENDPOINT`） |
| GCP GCS | `GCS_BUCKET_NAME` ＋ 身分設定（服務帳戶金鑰 `GCS_CREDENTIALS_JSON`／`GOOGLE_APPLICATION_CREDENTIALS`，或 `GCS_USE_ADC=true` 用部署環境附加身分——後者需開啟 IAM Credentials API 並授予服務帳戶 `serviceAccountTokenCreator` 與 bucket 的 `storage.objectAdmin`） |

三組都沒設定或上傳失敗，自動退回寫入本機 `outputs/` 目錄。同時設定多組時依 OSS → S3 → GCS 取第一個憑證齊全者，或用 `STORAGE_BACKEND=oss`／`s3`／`gcs` 明確指定。

---

## 主要依賴套件

| 套件 | 版本 |
|---|---|
| fastapi | 0.136.3 |
| uvicorn[standard] | 0.48.0 |
| python-multipart | 0.0.29 |
| openai | 2.38.0 |
| python-dotenv | 1.2.2 |
| requests | 2.34.2 |
| Pillow | 12.2.0 |
| httpx | 0.28.1 |
| websockets | 16.0 |
| aiohttp | 3.13.5 |
| pydantic | 2.13.4 |
| oss2 | 2.19.1 |
| boto3 | 1.43.58 |
| google-cloud-storage | 3.13.0 |
