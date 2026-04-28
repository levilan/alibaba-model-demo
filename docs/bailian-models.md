# 百炼控制台 - 模型體驗功能總覽

> 資料截止日期：2026-04-25

---

## 一、文本模型（`/model_experience_center/text`）

**預設模型**：Qwen3.6-Plus（可切換）

### 主要功能

- 自由文字對話輸入（最多 6000 字）
- 深度思考（Deep Think）開關 + 搜尋功能
- 上傳圖片 / 文件附件

### 快捷模板

圖片理解、視頻理解、網頁開發、代碼生成（如俄羅斯方塊遊戲）

### 右側配置面板參數

`system_prompt` / `top_p` / `temperature` / `enable_search` / `enable_thinking` / `thinking_budget` / `result_format`

### Tab 切換

模型體驗 / 模型調試

---

### 1.1 現役文本模型

| 模型系列 | 主干模型 | 動態更新版本 | 快照版本 |
|---|---|---|---|
| Qwen3.6-Plus | Qwen3.6-Plus | — | Qwen3.6-Plus-2026-04-02 |
| Qwen3.6-Max | Qwen3.6-Max-Preview | — | — |
| Qwen3.6-Flash | Qwen3.6-Flash | — | Qwen3.6-Flash-2026-04-16 |
| Qwen3.6 開源 | — | — | Qwen3.6-35B-A3B、Qwen3.6-27B |
| Qwen3.5-Plus | Qwen3.5-Plus | — | Qwen3.5-Plus-2026-04-20、Qwen3.5-Plus-2026-02-15 |
| Qwen3.5-Flash | Qwen3.5-Flash | — | Qwen3.5-Flash-2026-02-23 |
| Qwen3.5 開源 | — | — | Qwen3.5-35B-A3B、Qwen3.5-27B、Qwen3.5-122B-A10B、Qwen3.5-397B-A17B |
| DeepSeek（阿里直供） | — | — | DeepSeek-V3.2 |
| Qwen-MT-Lite | Qwen-MT-Lite | — | — |
| Qwen3-Max | Qwen3-Max、Qwen3-Max-Preview | — | Qwen3-Max-2026-01-23、Qwen3-Max-2025-09-23 |
| Qwen-Plus | Qwen-Plus | Qwen-Plus-Latest | Qwen-Plus-2025-09-11、Qwen-Plus-2025-07-28、Qwen-Plus-2025-07-14、Qwen-Plus-2025-04-28、Qwen-Plus-2025-01-25 |
| Qwen-Flash | 通義千問-Flash | — | Qwen-Flash-2025-07-28 |
| Qwen3-Coder-Plus | Qwen3-Coder-Plus | — | Qwen3-Coder-Plus-2025-09-23、Qwen3-Coder-Plus-2025-07-22 |
| Qwen3-Coder-Flash | Qwen3-Coder-Flash | — | Qwen3-Coder-Flash-2025-07-28 |
| Qwen-MT-Flash | Qwen-MT-Flash | — | — |
| Qwen-MT-Plus | Qwen-MT-Plus | — | — |
| Qwen-Flash-Character | Qwen-Flash-Character | — | — |

---

### 1.2 即將下線文本模型

| 模型系列 | 包含版本 |
|---|---|
| Qwen3 開源（部分下線） | 通義千問3-Coder-Next、Qwen3-Next-80B-A3B-Thinking、Qwen3-Next-80B-A3B-Instruct、Qwen3-Coder-30B-A3B-Instruct、Qwen3-30B-A3B-Thinking-2507、Qwen3-30B-A3B-Instruct-2507、Qwen3-235B-A22B-Thinking-2507、Qwen3-Coder-480B-A35B-Instruct、Qwen3-235B-A22B-Instruct-2507、Qwen3-14B、**Qwen3-4B（5/13 下線）**、Qwen3-30B-A3B、Qwen3-8B、通義千問3-0.6B、Qwen3-32B、Qwen3-235B-A22B、Qwen3-1.7B |
| Qwen-Max | Qwen-Max、Qwen-Max-Latest、Qwen-Max-2025-01-25 |
| Qwen-Turbo | Qwen-Turbo、Qwen-Turbo-Latest、Qwen-Turbo-2025-04-28、Qwen-Turbo-2024-11-01 |
| Qwen-MT-Turbo | Qwen-MT-Turbo |
| Qwen2.5 開源 | Qwen2.5-7B-Instruct-1M、Qwen2.5-14B-Instruct-1M、Qwen2.5-72B-Instruct、Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct、Qwen2.5-32B-Instruct |

---

## 二、語音模型（`/model_experience_center/voice`）

### 2.1 語音識別

| 模型系列 | 主干模型 | 快照版本 |
|---|---|---|
| Fun-ASR 語音識別 | Fun-ASR 語音識別、Fun-ASR 多語言語音識別 | Fun-ASR 多語言語音識別-2025-08-25 |
| Qwen3-ASR-Flash | Qwen3-ASR-Flash | — |

### 2.2 語音合成

三個子類型：
- **語音合成** — 自然流暢，聲情並茂
- **聲音復刻** — 惟妙惟肖，如聞其聲
- **AI 聲音設計** — 聲隨文動，栩栩如生

**音色庫精選合集**：電話客服、陪伴聊天、有聲書、短視頻配音、電商直播、消費電子

**推薦音色**：芊悅、不吃魚、晨煦、甜茶、詹妮弗、卡捷琳娜 等

#### 語音合成模型版本

| 模型 ID | 說明 |
|---|---|
| qwen3-tts-flash | 主干（預設） |
| qwen3-tts-flash-2025-11-27 | 快照 |
| qwen3-tts-flash-2025-09-18 | 快照 |
| qwen3-tts-instruct-flash | 主干 |
| qwen3-tts-instruct-flash-2026-01-26 | 快照 |
| qwen3-tts-vd-2026-01-26 | 聲音設計快照 |
| qwen3-tts-vc-2026-01-22 | 聲音復刻快照 |
| qwen-tts | 舊版主干 |

---

## 三、視覺模型（`/model_experience_center/vision`）

### 3.1 圖片生成

**預設模型**：Qwen-Image-2.0

**功能**：
- 文字描述生成圖片（最多 800 字）
- 參考圖上傳（最多 3 張）
- 尺寸選擇：2688×1536、2368×1728、2048×2048、1728×2368、1536×2688
- 生成張數：1–6 張
- 智能改寫開關

**免費額度**：32/100

#### 圖片生成模型版本

| 模型系列 | 主干模型 | 快照 / 其他版本 |
|---|---|---|
| Qwen-Image-2.0 | Qwen-Image-2.0 | Qwen-Image-2.0-2026-03-03 |
| Qwen-Image-2.0-Pro | Qwen-Image-2.0-Pro | Qwen-Image-2.0-Pro-2026-04-22、Qwen-Image-2.0-Pro-2026-03-03 |
| Qwen-Image-Max | Qwen-Image-Max | Qwen-Image-Max-2025-12-30 |
| Qwen-Image-Edit-Max | Qwen-Image-Edit-Max | Qwen-Image-Edit-Max-2026-01-16 |
| Z-Image-Turbo | — | Z-Image-Turbo |
| Qwen-Image-Plus | Qwen-Image-Plus、Qwen-Image | Qwen-Image-Plus-2026-01-09 |
| Qwen-Image-Edit-Plus | — | Qwen-Image-Edit-Plus-2025-12-15、Qwen-Image-Edit-Plus-2025-10-30、Qwen-Image-Edit-Plus、Qwen-Image-Edit |
| Wan-T2I（文生圖） | Wan2.6-T2I | Wan2.5-T2I-Preview、Wan2.2-T2I-Plus、Wan2.2-T2I-Flash、Wan2.1-T2I-Plus、Wan2.1-T2I-Turbo |

---

### 3.2 視頻生成

**使用模型**：萬相 2.7（右側面板顯示歷史生成記錄，可按類型篩選）

#### 圖生視頻（I2V）

上傳參考圖（jpeg / jpg / png / webp，最大 20MB），輸入提示詞，選擇清晰度（720P / 1080P）和時長（5–16 秒）

| 可用模型 |
|---|
| 萬相2.7-圖生視頻 |
| Wan2.6-I2V-flash |
| Wan2.6-I2V |
| Wan2.5-I2V-Preview |
| Wan2.2-I2V-Plus |
| Wan2.2-I2V-Flash |

#### 文生視頻（T2V）

| 可用模型 |
|---|
| 萬相2.7-文生視頻 |
| Wan2.6-T2V |
| Wan2.5-T2V-Preview |
| Wan2.2-T2V-Plus |

#### 參考生視頻（R2V）

| 可用模型 |
|---|
| 萬相2.7-參考生視頻 |
| Wan2.6-R2V |
| Wan2.6-R2V-Flash |

---

## 四、HappyHorse 系列（2026-04-22 / 2026-04-26 新上線）

| 模型 ID | 類型 | 說明 |
|---|---|---|
| happyhorse-1.0-i2v | I2V | 高度還原動態畫面，保持圖像一致性 |
| happyhorse-1.0-t2v | T2V | 精準理解文本語義，流暢高質量輸出 |
| happyhorse-1.0-r2v | R2V | 支援最多 9 張參考圖，主體/場景參考 |
| happyhorse-1.0-video-edit | 視頻編輯 | 自然語言指令編輯，最多 5 張參考圖 |

---

## 功能架構總覽

| 模塊 | 子功能 | 主要模型 |
|---|---|---|
| 文本模型 | 對話、模型調試 | Qwen3.6-Plus 等 |
| 語音模型 | 語音識別、語音合成 / 復刻 / 設計 | Fun-ASR、Qwen3-ASR-Flash |
| 視覺模型 | 圖片生成、視頻生成（I2V / T2V / R2V） | Qwen-Image-2.0、萬相 2.7 |
