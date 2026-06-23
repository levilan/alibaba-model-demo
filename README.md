# AI Model Tester

阿里巴巴雲端 AI 模型測試平台，支援文字生成、圖片生成、影片生成、語音辨識／合成、全模態即時通話，以及 NenAI 擴充模型。

---

## 快速部署

### 前置需求

- [Docker](https://docs.docker.com/get-docker/) 與 Docker Compose
- DashScope API Key（格式：`sk-...`）
- NenAI API Key（NenAI 模型專用，選用）

### Docker 部署（推薦）

```bash
git clone <專案網址>
cd ai-model-tester
docker compose up -d --build
```

瀏覽器開啟 `http://localhost`，輸入 API Key 登入。

**常用指令**

```bash
# 查看日誌
docker logs -f ai-model-tester-ai-model-tester-1

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

## 可用模型列表

### 文字生成（需 DashScope API Key）

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| qwen3.7-max | Qwen3.7 Max | 旗艦 |
| qwen3.6-max-preview | Qwen3.6 Max | 旗艦 |
| qwen3.6-plus | Qwen3.6 Plus | 均衡 |
| qwen3.5-plus | Qwen3.5 Plus | 均衡 |
| qwen3.6-flash | Qwen3.6 Flash | 極速 |
| qwen3.5-flash | Qwen3.5 Flash | 極速 |
| qwen3-coder-plus | Qwen3 Coder Plus | 代碼 |
| qwen3-coder-flash | Qwen3 Coder Flash | 代碼 |
| qwen-mt-plus | Qwen MT Plus | 翻譯 |
| qwen-mt-flash | Qwen MT Flash | 翻譯 |
| qwen-mt-lite | Qwen MT Lite | 翻譯 |
| qwen-flash-character | Qwen Flash Character | 角色 |
| deepseek-v3.2 | DeepSeek V3.2 | 第三方 |

---

### 圖片生成（需 DashScope API Key）

| 模型 ID | 名稱 | 分類 | 允許尺寸 |
|---|---|---|---|
| qwen-image-2.0-pro | 千問圖像 2.0 Pro | 千問文生圖 | 1024×1024 等 |
| qwen-image-2.0 | 千問圖像 2.0 | 千問文生圖 | 1024×1024 等 |
| qwen-image-max | 千問圖像 Max | 千問文生圖 | 1024×1024 等 |
| qwen-image-plus | 千問圖像 Plus | 千問文生圖 | 1328×1328、1664×928、928×1664、1472×1104、1104×1472 |
| wan2.6-t2i | 萬相 2.6 T2I | 萬相文生圖 | 1024×1024 等 |
| z-image-turbo | Z-Image Turbo | Z-Image | 1024×1024 等 |
| wan2.7-image-pro | 萬相 2.7 Image Pro | 萬相圖像編輯 | 1024×1024 等 |
| wan2.7-image | 萬相 2.7 Image | 萬相圖像編輯 | 1024×1024 等 |
| wan2.6-image | 萬相 2.6 Image | 萬相圖像編輯 | 1024×1024 等 |
| qwen-image-edit-max | 千問圖像編輯 Max | 千問圖像編輯 | 1024×1024 等 |
| qwen-image-edit-plus | 千問圖像編輯 Plus | 千問圖像編輯 | 1024×1024 等 |

圖片輸出支援點擊放大預覽（lightbox），以自然比例顯示不裁切。

---

### 影片生成（需 DashScope API Key）

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| wan2.7-t2v | 萬相 2.7 T2V | 文生影片 |
| wan2.6-t2v | 萬相 2.6 T2V | 文生影片 |
| wan2.7-i2v | 萬相 2.7 I2V | 圖生影片 |
| wan2.6-i2v | 萬相 2.6 I2V | 圖生影片 |
| wan2.6-i2v-flash | 萬相 2.6 I2V Flash | 圖生影片 |
| wan2.7-r2v | 萬相 2.7 R2V | 參考生影片 |
| wan2.6-r2v | 萬相 2.6 R2V | 參考生影片 |
| wan2.6-r2v-flash | 萬相 2.6 R2V Flash | 參考生影片 |
| happyhorse-1.0-t2v | HappyHorse T2V | HappyHorse |
| happyhorse-1.0-i2v | HappyHorse I2V | HappyHorse |
| happyhorse-1.0-r2v | HappyHorse R2V | HappyHorse |
| happyhorse-1.0-video-edit | HappyHorse Video Edit | HappyHorse |
| wan2.7-videoedit | 萬相 2.7 視頻編輯 | 萬相視頻編輯 |

影片輸出支援「⛶ 放大」按鈕在 lightbox 中預覽，播放器填滿卡片寬度。

---

### NenAI 模型（需 NenAI API Key）

| 模型 ID | 名稱 | 分類 | 輸入 |
|---|---|---|---|
| wan2.7-i2v-spicy | Wan 2.7 I2V Spicy | 影片生成 | 文字 / 圖片 |
| z-image-spicy | Z-Image Spicy | 圖片生成 | 文字 prompt |
| qwen-image-edit-spicy | 圖像編輯 Spicy | 圖像編輯 | prompt + 來源圖 |
| face-swap | 圖像換臉 | 圖像換臉 | 來源圖 + 換臉參考圖（無需 prompt）|

---

### 語音（需 DashScope API Key）

**ASR 語音辨識**

| 模型 ID | 名稱 | 描述 |
|---|---|---|
| qwen3-asr-flash | Qwen3 ASR Flash | 新一代極速識別，多語言 |
| paraformer-v2 | Fun-ASR 語音識別 | 高精度普通話識別 |
| sensevoice-v1 | Fun-ASR 多語言 | 中／英／日／韓／粵 |

**TTS 語音合成**

| 模型 ID | 名稱 | 可用音色 |
|---|---|---|
| qwen3-tts-flash-2025-11-27 | Qwen3 TTS Flash | Cherry、Ethan、Serena、Wayne、Summer、Belle、Cove、Aria、Kai、Luna |
| cosyvoice-v3-plus | CosyVoice v3 Plus | 龍安洋、龍安歡 |
| cosyvoice-v3-flash | CosyVoice v3 Flash | 龍安洋、龍安歡、龍安柔、龍安昀、龍安溫、龍小淳、龍小夏、YUMI、龍華、龍橙、龍飛、龍妙、龍悅、龍碩、龍書、Bella3.0、龍嘉欣（粵）、龍老鐵（東北話）、Riko（日）、loongkyong（韓）共 20 個 |
| cosyvoice-v3.5-plus | CosyVoice v3.5 Plus | 設計 / 複刻音色（北京地域，需設計 API Key） |
| cosyvoice-v3.5-flash | CosyVoice v3.5 Flash | 設計 / 複刻音色（北京地域，需設計 API Key） |

> **聲音複刻**：上傳 10–20 秒人聲音檔即可複刻專屬音色（國際版）。
>
> **聲音設計**：用文字描述目標音色特徵即可生成定制音色（北京地域，需額外填入北京區 DashScope API Key）。設計音色可跨 v3 / v3.5 模型使用，合成時系統自動依 voice_id 選用正確模型。

---

### 全模態即時通話（需 DashScope API Key）

**即時通話（Realtime WebSocket）**

| 模型 ID | 名稱 |
|---|---|
| qwen3.5-omni-flash-realtime | Qwen3.5 Omni Flash Realtime |
| qwen3.5-omni-plus-realtime | Qwen3.5 Omni Plus Realtime |
| qwen2.5-omni-3b-realtime | Qwen2.5 Omni 3B Realtime |
| qwen2.5-omni-7b-realtime | Qwen2.5 Omni 7B Realtime |

**非同步對話（HTTP Streaming）**

| 模型 ID | 名稱 |
|---|---|
| qwen3.5-omni-flash | Qwen3.5 Omni Flash |
| qwen3.5-omni-plus | Qwen3.5 Omni Plus |

**音色（依模型系列）**

| 模型系列 | 預設音色 | 音色數量 |
|---|---|---|
| qwen3.5-omni-* | Tina（甜甜） | 56 個（通用普通話 / 方言 / 國際） |
| qwen2.5-omni-* | Ethan（晨煦） | 2 個 |

qwen3.5-omni 部分音色：Tina、Cindy（台灣腔）、Ethan、Serena、Harvey、Maia、Ryan、Jennifer、Katerina、Sunny（四川）、Dylan（北京）、Kiki（粵語）、Sohee（韓）、Ono Anna（日）、Emilien（法）…等共 56 個。

---

## 主要依賴套件

| 套件 | 版本 |
|---|---|
| fastapi | 0.136.3 |
| uvicorn[standard] | 0.48.0 |
| openai | 2.38.0 |
| dashscope | 1.25.19 |
| Pillow | 12.2.0 |
| httpx | 0.28.1 |
| websockets | 16.0 |
| aiohttp | 3.13.5 |
| pydantic | 2.13.4 |
