# AI Model Tester

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

瀏覽器開啟 `http://localhost:5050`，輸入 API Key 登入。

**常用指令**

```bash
# 查看日誌
docker logs -f ai-model-tester-ai-model-tester-1

# 停止服務
docker compose down

# 強制重新建置
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
| qwen3-max | Qwen3 Max | 旗艦 |
| qwen3.6-plus | Qwen3.6 Plus | 均衡 |
| qwen3.5-plus | Qwen3.5 Plus | 均衡 |
| qwen-plus | Qwen Plus | 均衡 |
| qwen3.6-flash | Qwen3.6 Flash | 極速 |
| qwen3.5-flash | Qwen3.5 Flash | 極速 |
| qwen-flash | Qwen Flash | 極速 |
| qwen3-coder-plus | Qwen3 Coder Plus | 代碼 |
| qwen3-coder-flash | Qwen3 Coder Flash | 代碼 |
| qwen-mt-plus | Qwen MT Plus | 翻譯 |
| qwen-mt-flash | Qwen MT Flash | 翻譯 |
| qwen-mt-lite | Qwen MT Lite | 翻譯 |
| qwen-flash-character | Qwen Flash Character | 角色 |
| deepseek-v3.2 | DeepSeek V3.2 | 第三方 |

### 圖片生成（需 DashScope API Key）

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| qwen-image-2.0-pro | 千問圖像 2.0 Pro | 千問文生圖 |
| qwen-image-2.0 | 千問圖像 2.0 | 千問文生圖 |
| qwen-image-max | 千問圖像 Max | 千問文生圖 |
| qwen-image-plus | 千問圖像 Plus | 千問文生圖 |
| wan2.6-t2i | 萬相 2.6 T2I | 萬相文生圖 |
| z-image-turbo | Z-Image Turbo | Z-Image |
| wan2.7-image-pro | 萬相 2.7 Image Pro | 萬相圖像編輯 |
| wan2.7-image | 萬相 2.7 Image | 萬相圖像編輯 |
| wan2.6-image | 萬相 2.6 Image | 萬相圖像編輯 |
| qwen-image-edit-max | 千問圖像編輯 Max | 千問圖像編輯 |
| qwen-image-edit-plus | 千問圖像編輯 Plus | 千問圖像編輯 |

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

### NenAI 模型（需 NenAI API Key）

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| wan2.7-i2v-spicy | Wan 2.7 I2V Spicy | 影片生成 |
| z-image-spicy | Z-Image Spicy | 圖片生成 |

### 語音（需 DashScope API Key）

**ASR 語音辨識**

| 模型 ID | 名稱 |
|---|---|
| qwen3-asr-flash | Qwen3 ASR Flash |
| paraformer-v2 | Fun-ASR 語音識別 |
| sensevoice-v1 | Fun-ASR 多語言 |

**TTS 語音合成**

| 模型 ID | 名稱 | 音色 |
|---|---|---|
| qwen-tts / Cherry | 芊悅 | 女・親切 |
| qwen-tts / Ethan | 逸軒 | 男・穩重 |
| qwen-tts / Serena | 晨煦 | 女・清爽 |
| qwen-tts / Wayne | 韋恩 | 男・磁性 |
| qwen-tts / Summer | 甜茶 | 女・活潑 |
| qwen-tts / Belle | 不吃魚 | 女・元氣 |
| qwen-tts / Cove | 詹妮弗 | 女・知性 |
| qwen-tts / Aria | 卡捷琳娜 | 女・優雅 |
| qwen-tts / Kai | 嘉熙 | 男・輕快 |
| qwen-tts / Luna | 月桐 | 女・溫柔 |
