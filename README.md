# NenAI Testing Platform

基於 FastAPI + 原生 JavaScript 的 AI 模型測試平台，整合 NenAI (nen.com.tw) 所提供的文字、圖片、影片與 NenAI Spicy 擴充模型，統一使用一把 NenAI API Key。

---

## 快速部署

### 前置需求

- [Docker](https://docs.docker.com/get-docker/) 與 Docker Compose
- NenAI API Key（格式：`sk-...`）

### Docker 部署（推薦）

```bash
git clone <專案網址>
cd alibaba-model-nenAI
docker compose up -d --build
```

瀏覽器開啟 `http://localhost:5050`，輸入 NenAI API Key 登入。

**常用指令**

```bash
# 查看日誌
docker logs -f alibaba-model-nenai-ai-model-tester-1

# 停止服務
docker compose down

# 強制重新建置（更新程式碼後使用）
docker compose build --no-cache && docker compose up -d
```

### 本機直接執行

```bash
pip install -r requirements.txt
python app.py
```

---

## 功能模組

| 頁籤 | 功能說明 |
|------|----------|
| 文字生成 | SSE 串流輸出、Thinking 模式、多模型切換 |
| 圖片生成 | 文生圖 (T2I) 與圖像編輯 (I2I)，支援多張參考圖、點擊放大預覽 |
| 影片生成 | 文生影片 / 圖生影片 / 參考生影片 / 視頻編輯 / 動作動畫，含即時輪詢進度與配音 |
| NenAI Spicy | Wan 2.7 I2V Spicy、Z-Image Spicy、圖像編輯 Spicy、圖像換臉 |
| 語音模型 | 語音辨識 (ASR，含串流) 與語音合成 (TTS) |
| AI Canvas | 節點式視覺化畫布（`/canvas`），可拖拉連線組合文字／圖片／影片／圖像編輯／MuleAI 節點，串接多個模型呼叫 |

---

## 可用模型列表

### 文字生成

| 模型 ID | 名稱 | 分類 | Thinking |
|---|---|---|---|
| qwen3.8-max | Qwen3.8 Max | 旗艦 | ✓ |
| qwen3.7-max | Qwen3.7 Max | 旗艦 | ✓ |
| qwen3.6-max-preview | Qwen3.6 Max | 旗艦 | ✓ |
| qwen3.7-plus | Qwen3.7 Plus | 均衡 | ✓ |
| qwen3.6-plus | Qwen3.6 Plus | 均衡 | ✓ |
| qwen3.5-plus | Qwen3.5 Plus | 均衡 | ✓ |
| qwen3.6-flash | Qwen3.6 Flash | 極速 | ✓ |
| qwen3.5-flash | Qwen3.5 Flash | 極速 | ✓ |
| qwen3-coder-plus | Qwen3 Coder Plus | 代碼 | ✓ |
| qwen3-coder-flash | Qwen3 Coder Flash | 代碼 | ✓ |
| qwen-plus-character | Qwen Plus Character | 角色 | — |
| deepseek-v4-pro | DeepSeek V4 Pro | 第三方 | — |
| deepseek-v4-flash | DeepSeek V4 Flash | 第三方 | — |
| deepseek-v3.2 | DeepSeek V3.2 | 第三方 | — |
| glm-5.1 | GLM 5.1 | 第三方 | — |
| glm-5.2 | GLM 5.2 | 第三方 | — |

### 圖片生成

**文生圖 (T2I)**

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| qwen-image-2.0-pro | 千問圖像 2.0 Pro | 千問文生圖 |
| qwen-image-2.0 | 千問圖像 2.0 | 千問文生圖 |
| qwen-image-max | 千問圖像 Max | 千問文生圖 |
| qwen-image-plus | 千問圖像 Plus | 千問文生圖 |
| wan2.6-t2i | 萬相 2.6 T2I | 萬相文生圖 |
| z-image-turbo | Z-Image Turbo | Z-Image |

**圖像編輯 (I2I)**

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| wan2.7-image-pro | 萬相 2.7 Image Pro | 萬相圖像編輯 |
| wan2.7-image | 萬相 2.7 Image | 萬相圖像編輯 |
| wan2.6-image | 萬相 2.6 Image | 萬相圖像編輯 |
| qwen-image-edit-max | 千問圖像編輯 Max | 千問圖像編輯 |
| qwen-image-edit-plus | 千問圖像編輯 Plus | 千問圖像編輯 |
| qwen-image-2.0-pro（編輯） | 千問圖像 2.0 Pro | 千問圖像編輯 |
| qwen-image-2.0（編輯） | 千問圖像 2.0 | 千問圖像編輯 |

> qwen-image-2.0 系列為生成與編輯融合模型：最多 3 張參考圖、可一次輸出 1–6 張，並以 `prompt_extend` 取代 `ref_strength` 參數。

**GPT Image（文生圖，尺寸格式為 `WIDTHxHEIGHT`）**

| 模型 ID | 名稱 |
|---|---|
| gpt-image-2 | GPT Image 2（OpenAI 旗艦圖像模型） |
| gpt-image-1.5 | GPT Image 1.5（OpenAI 前代圖像模型） |

**Gemini Image（文生圖，走 `/v1/chat/completions` + `modalities`，不支援自訂尺寸）**

| 模型 ID | 名稱 |
|---|---|
| gemini-3-pro-image | Gemini 3 Pro Image（旗艦，畫質最佳） |
| gemini-3.1-flash-image | Gemini 3.1 Flash Image（速度與品質平衡） |
| gemini-2.5-flash-image | Gemini 2.5 Flash Image（穩定版） |
| gemini-3.1-flash-lite-image | Gemini 3.1 Flash Lite Image（輕量極速） |

圖片輸出支援點擊放大預覽（lightbox）。

### 影片生成

| 模型 ID | 名稱 | 分類 | 配音 |
|---|---|---|---|
| wan2.7-t2v | 萬相 2.7 T2V | 文生影片 | ✓ |
| wan2.6-t2v | 萬相 2.6 T2V | 文生影片 | ✓ |
| wan2.7-i2v | 萬相 2.7 I2V | 圖生影片 | ✓ |
| wan2.6-i2v | 萬相 2.6 I2V | 圖生影片 | ✓ |
| wan2.6-i2v-flash | 萬相 2.6 I2V Flash | 圖生影片 | ✓ |
| wan2.7-r2v | 萬相 2.7 R2V | 參考生影片 | ✓ |
| wan2.6-r2v | 萬相 2.6 R2V | 參考生影片 | ✓ |
| wan2.6-r2v-flash | 萬相 2.6 R2V Flash | 參考生影片 | ✓ |
| happyhorse-1.1-t2v | HappyHorse 1.1 T2V | HappyHorse | — |
| happyhorse-1.0-t2v | HappyHorse 1.0 T2V | HappyHorse | — |
| happyhorse-1.1-i2v | HappyHorse 1.1 I2V | HappyHorse | — |
| happyhorse-1.0-i2v | HappyHorse 1.0 I2V | HappyHorse | — |
| happyhorse-1.1-r2v | HappyHorse 1.1 R2V | HappyHorse | — |
| happyhorse-1.0-r2v | HappyHorse 1.0 R2V | HappyHorse | — |
| happyhorse-1.0-video-edit | HappyHorse Video Edit | HappyHorse | — |
| wan2.7-videoedit | 萬相 2.7 視頻編輯 | 萬相視頻編輯 | — |
| wan2.2-animate-mix | 萬相 2.2 視頻換人 | 萬相動作動畫 | — |
| wan2.2-animate-move | 萬相 2.2 圖生動作 | 萬相動作動畫 | — |

> 萬相 2.6/2.7 系列 T2V/I2V/R2V 皆支援自動配音（BGM 自動生成或自訂音訊上傳）。
> 動作動畫模型：視頻換人（將參考影片角色替換為人物圖片）、圖生動作（將參考影片動作遷移到人物圖片）。

**Veo（Google，duration 僅接受 4/6/8 秒）**

| 模型 ID | 名稱 |
|---|---|
| veo-3.1-generate-001 | Veo 3.1（旗艦，含原生配音） |
| veo-3.1-fast-generate-001 | Veo 3.1 Fast（極速，含原生配音） |
| veo-3.1-lite-generate-001 | Veo 3.1 Lite（輕量，含原生配音） |

**Gemini Omni（走 `/v1beta/interactions`，模型自行決定長度/解析度，固定含原生配音）**

| 模型 ID | 名稱 |
|---|---|
| gemini-omni-flash-preview | Gemini Omni Flash Preview（多模態影片生成預覽版，最長約 10 秒） |

### NenAI Spicy（需 NenAI API Key）

| 模型 ID | 名稱 | 分類 | 輸入 |
|---|---|---|---|
| wan2.7-i2v-spicy | Wan 2.7 I2V Spicy | 影片生成 | 文字 / 圖片 |
| z-image-spicy | Z-Image Spicy | 圖片生成 | 文字 prompt |
| qwen-image-edit-spicy | 圖像編輯 Spicy | 圖像編輯 | prompt + 來源圖 |
| face-swap | 圖像換臉 | 圖像換臉 | 來源圖 + 換臉參考圖（無需 prompt）|

### 語音模型

| 模型 ID | 名稱 | 分類 | 功能 |
|---|---|---|---|
| qwen-audio-3.0-asr-flash | Qwen Audio 3.0 ASR Flash | 語音辨識 | 上傳完整音檔，一次回傳逐字稿 |
| qwen-audio-3.0-asr-flash-streaming | Qwen Audio 3.0 ASR Flash（串流） | 語音辨識 | SSE 串流回傳中間辨識結果 |
| qwen-audio-3.0-tts-plus | Qwen Audio 3.0 TTS Plus | 語音合成 | 高品質語音合成 |
| qwen-audio-3.0-tts-flash | Qwen Audio 3.0 TTS Flash | 語音合成 | 極速語音合成 |

> ASR 走 NenAI 網關 OpenAI 相容的 `/v1/audio/transcriptions`；TTS 走 DashScope 風格的 `/v1/services/audio/tts/SpeechSynthesizer`（回傳 JSON，音檔網址在 `output.audio.url`），支援選填的 `voice`（CosyVoice v3 音色 id，例如 `longanlingxin`、`loongjohn`）、`instructions`（語氣風格描述）與 `sample_rate`/`volume`/`language_hints`（`metadata` 子欄位）。

---

## AI Canvas（`/canvas`）

節點式視覺化畫布，以拖拉連線的方式組合平台上的模型呼叫（類似 ComfyUI），基於 [litegraph.js](https://github.com/jagenjo/litegraph.js)。

| 節點 | 說明 |
|---|---|
| 文字 Text | 手動輸入 prompt，或選模型做真正的文字生成；若連接圖片輸入則可改用「分析圖片」 |
| 圖片 Image | 文生圖 (t2i)；若連接參考圖輸入則自動切換為圖像生成 (i2i) |
| 影片 Video | 依連接的圖片組合自動切換 t2v / i2v（首尾幀）/ r2v（最多 6 張參考圖） |
| 圖像編輯 Editing | 圖像編輯 (i2i)，需連接一張來源圖片 |
| MuleAI Spicy | 對應 NenAI Spicy 四個模型，依選擇的模型動態切換必填輸入與輸出型別（image/video） |

節點之間可用連線傳遞文字/圖片/影片輸出，設定面板僅在節點被選取時以固定大小浮層顯示於節點下方。

---

## 雲端物件儲存（選用）

生成的圖片/影片預設寫入容器本機的 `outputs/` 目錄。若部署在無狀態、多實例的環境
（例如 GCP Cloud Run）本機磁碟不可靠，可設定以下任一組環境變數，改把檔案上傳到
雲端物件儲存、以簽名網址（有效期 7 天）提供下載：

| 後端 | 需要的環境變數 |
|---|---|
| 阿里雲 OSS | `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET` |
| AWS S3 | `S3_ACCESS_KEY_ID`、`S3_SECRET_ACCESS_KEY`、`S3_BUCKET_NAME`（可選 `S3_REGION`，預設 `us-east-1`；`S3_ENDPOINT` 供 S3 相容服務使用） |
| GCP GCS | `GCS_BUCKET_NAME` + 下列其中一種身分（見下方說明） |

三組都沒設定，或設定了但上傳失敗，都會自動退回寫入本機 `outputs/` 目錄。若同時設
定了多組，預設依 OSS → S3 → GCS 的順序，選第一個「憑證齊全」的啟用；也可以用
`STORAGE_BACKEND=oss` / `s3` / `gcs` 明確指定要用哪一個。

### GCS 的兩種身分設定方式

**方式一：服務帳戶金鑰**（`GCS_CREDENTIALS_JSON` 直接放金鑰 JSON 內容，或
`GOOGLE_APPLICATION_CREDENTIALS` 指向掛載的金鑰檔路徑）——本地就有私鑰可以直接簽
署，設定最簡單，但要另外管理一份長期有效的金鑰。

**方式二：`GCS_USE_ADC=true`**，改用部署環境本身附加的服務帳戶（Cloud Run 的
Service Account、GCE 的附加身分），不需要額外建立、保管任何金鑰檔。缺點是附加身
分沒有私鑰，簽名網址得改呼叫 IAM SignBlob API 遠端簽章，需要多做兩件 IAM 設定：

1. 該服務帳戶要能「模擬自己」（signBlob 是這樣運作的）：
   ```bash
   gcloud services enable iamcredentials.googleapis.com --project=$PROJECT_ID
   gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
     --member="serviceAccount:$SA_EMAIL" \
     --role="roles/iam.serviceAccountTokenCreator"
   ```
2. 該服務帳戶要有目標 bucket 的物件讀寫權限：
   ```bash
   gsutil iam ch serviceAccount:$SA_EMAIL:roles/storage.objectAdmin gs://$BUCKET
   ```
   `$SA_EMAIL` 是 Cloud Run 服務（或 GCE 執行個體）綁定的服務帳戶信箱；沒有另外指
   定的話預設是 Compute Engine 預設服務帳戶
   `PROJECT_NUMBER-compute@developer.gserviceaccount.com`。

兩種方式擇一即可，`GCS_CREDENTIALS_JSON`/`GOOGLE_APPLICATION_CREDENTIALS` 其中一個
有值時一律優先於 `GCS_USE_ADC`。

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
