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

---

## 可用模型列表

### 文字生成

| 模型 ID | 名稱 | 分類 | Thinking |
|---|---|---|---|
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

### NenAI Spicy（需 NenAI API Key）

| 模型 ID | 名稱 | 分類 | 輸入 |
|---|---|---|---|
| wan2.7-i2v-spicy | Wan 2.7 I2V Spicy | 影片生成 | 文字 / 圖片 |
| z-image-spicy | Z-Image Spicy | 圖片生成 | 文字 prompt |
| qwen-image-edit-spicy | 圖像編輯 Spicy | 圖像編輯 | prompt + 來源圖 |
| face-swap | 圖像換臉 | 圖像換臉 | 來源圖 + 換臉參考圖（無需 prompt）|

---

## 主要依賴套件

| 套件 | 版本 |
|---|---|
| fastapi | 0.136.3 |
| uvicorn[standard] | 0.48.0 |
| openai | 2.38.0 |
| httpx | 0.28.1 |
| Pillow | 12.2.0 |
| websockets | 16.0 |
| oss2 | 2.19.1 |
| pydantic | 2.13.4 |
