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
| 文字生成 | SSE 串流輸出、Thinking 模式、多模型切換、多輪對話記憶（帶入前面幾輪對話讓模型記得上下文，按「清除對話」重置） |
| 圖片生成 | 文生圖 (T2I) 與圖像編輯 (I2I)，支援多張參考圖、點擊放大預覽 |
| 影片生成 | 文生影片 / 圖生影片 / 參考生影片 / 視頻編輯 / 動作動畫，含即時輪詢進度與配音 |
| NenAI Spicy | Wan 2.7 I2V Spicy、Z-Image Spicy、圖像編輯 Spicy、圖像換臉 |
| 語音模型 | 語音辨識 (ASR，含串流) 與語音合成 (TTS) |
| AI Canvas | 節點式視覺化畫布（`/canvas`），可拖拉連線組合文字／圖片／影片／圖像編輯／MuleAI 節點，串接多個模型呼叫 |

每個分頁的「模型」選單旁邊都會顯示該模型的參考單價（例如 `模型 ($2→$6/1M)`），資料來自 `GET /api/pricing`（後端代理並快取網關自己的 `/api/pricing` 計費表 1 小時）。文字/語音等 token 計費模型顯示「輸入→輸出每百萬 token 美金」，圖片/影片/NenAI Spicy 等按次計費模型顯示「每次呼叫美金」。這只是換算後的參考價格（假設帳號分組倍率為 1），不是精確帳單金額，正式扣款請以 NenAI 後台實際扣款為準。

主測試台 header 另外有一個「本次花費」徽章，累加這次瀏覽器分頁裡所有呼叫的估計花費：文字生成用實際 token 數精確計算，圖片/影片/NenAI Spicy/語音辨識用固定單價 × 成功次數累加（token 計費的圖片模型與語音合成目前不計入）。純瀏覽器端記憶體累加，重新整理頁面會歸零。

---

## 可用模型列表

### 文字生成

| 模型 ID | 名稱 | 分類 | 思考模式 |
|---|---|---|---|
| qwen3.8-max | Qwen3.8 Max | 旗艦 | enable_thinking |
| qwen3.7-max | Qwen3.7 Max | 旗艦 | enable_thinking |
| qwen3.6-max-preview | Qwen3.6 Max | 旗艦 | enable_thinking |
| qwen3.7-plus | Qwen3.7 Plus | 均衡 | enable_thinking |
| qwen3.6-plus | Qwen3.6 Plus | 均衡 | enable_thinking |
| qwen3.5-plus | Qwen3.5 Plus | 均衡 | enable_thinking |
| qwen3.6-flash | Qwen3.6 Flash | 極速 | enable_thinking |
| qwen3.5-flash | Qwen3.5 Flash | 極速 | enable_thinking |
| qwen3-coder-plus | Qwen3 Coder Plus | 代碼 | — |
| qwen3-coder-flash | Qwen3 Coder Flash | 代碼 | — |
| qwen-plus-character | Qwen Plus Character | 角色 | — |
| deepseek-v4-pro | DeepSeek V4 Pro | 第三方 | enable_thinking（預設開啟） |
| deepseek-v4-flash | DeepSeek V4 Flash | 第三方 | enable_thinking（預設開啟） |
| deepseek-v3.2 | DeepSeek V3.2 | 第三方 | enable_thinking（預設開啟） |
| glm-5.1 | GLM 5.1 | 第三方 | enable_thinking（預設開啟） |
| glm-5.2 | GLM 5.2 | 第三方 | —（走 `/v1/messages`，思考關不掉） |
| dola-seed-sc | Seed SC | ByteDance | — |
| dola-seed-2.0-lite | Seed 2.0 Lite | ByteDance | — （無條件思考，關不掉） |
| dola-seed-2.0-pro | Seed 2.0 Pro | ByteDance | — （無條件思考，關不掉） |
| claude-opus-4-8 / 4-7 / 4-6 / 4-5 / 4-1 | Claude Opus 系列 | Claude | — |
| claude-sonnet-5 / 4-6 / 4-5 | Claude Sonnet 系列 | Claude | — |
| claude-haiku-4-5 | Claude Haiku 4.5 | Claude | — |
| claude-fable-5 | Claude Fable 5 | Claude | — |
| gpt-5.6-terra / sol / luna | GPT 5.6 特化系列 | GPT | reasoning_effort |
| gpt-5.5 / 5.4 / 5.4-mini / 5.4-nano / 5.2 / 5-mini | GPT 5.x 系列 | GPT | reasoning_effort |
| gemini-3.1-pro-preview 等 7 個 Gemini 模型 | Gemini 系列 | Gemini | — |

> **思考模式的三種機制，不能混用**：Qwen/DeepSeek/GLM 用布林值 `enable_thinking`（這幾家幾乎都實測預設就是開啟，必須明確送 `enable_thinking:false` 才會關閉並省 token——完全不帶這個欄位並不會關閉思考，後端一律會明確帶上 `true`/`false`，只有 GPT 系列例外不帶）；GPT 系列改用字串 `reasoning_effort`（實測這個網關接受的枚舉是 `none/low/medium/high/xhigh`，跟 OpenAI 官方文件常見的 `minimal/low/medium/high` 不同，帶錯值或帶 `enable_thinking` 給 GPT 都會被直接拒絕）；Claude／Gemini／ByteDance Seed 2.0 系列目前實測皆無法透過這個網關控制（Claude 送了無效、Gemini 3.x 與 Seed 2.0 系列無條件思考、關不掉），因此 UI 不提供對應開關；`qwen3-coder-plus`/`qwen3-coder-flash` 則是實測 `enable_thinking` 完全沒有效果（true/false 都不會有思考過程），同樣不顯示開關。開啟後若上游回傳 `reasoning_content`（Qwen/DeepSeek/GLM 大部分模型，以及無法關閉思考的 Seed 2.0 系列），會在回答上方顯示成可收合的「思考過程」區塊。

> **`glm-5.2` 走的是 Anthropic Messages 格式的 `/v1/messages`**，不是其餘文字模型共用的 OpenAI 相容 `/v1/chat/completions`（見 `_ANTHROPIC_MESSAGES_MODELS`）。後端會把請求轉成 Anthropic 格式（`system` 提到頂層、`stop` → `stop_sequences`、`max_tokens` 必填），並把回應轉回前端既有的協定。實測（2026-08-10，正式網關）這條路徑有兩個差異：
> - **思考過程只有串流模式看得到**：串流時思考會走獨立的 `thinking` content block（`thinking_delta` 事件），後端轉成前端的 `reasoning` 事件，顯示效果與 `reasoning_content` 相同；**非串流回應則完全不含思考過程**，只有 `text` block（token 照樣被消耗）。
> - **思考關不掉**：`thinking.type=disabled`、`enable_thinking:false` 實測都無效（同一題 output token 150／151／111，關不掉）。相對地走 `/v1/chat/completions` 時 `enable_thinking:false` 是真的有效的（同題 completion token 從 139 掉到 1）。因此 `glm-5.2` 的思考模式開關已從 UI 移除，不顯示一個沒有作用的開關。

### 圖片生成

**文生圖 (T2I)**

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| qwen-image-2.0-pro | 千問圖像 2.0 Pro | 千問文生圖 |
| qwen-image-2.0 | 千問圖像 2.0 | 千問文生圖 |
| wan2.7-image-pro | 萬相 2.7 Image Pro | 萬相文生圖 |
| wan2.7-image | 萬相 2.7 Image | 萬相文生圖 |
| wan2.6-t2i | 萬相 2.6 T2I | 萬相文生圖 |
| z-image-turbo | Z-Image Turbo | Z-Image |
| MAI-Image-2.5-Pro | MAI-Image-2.5-Pro | MAI Image |
| MAI-Image-2.5 | MAI-Image-2.5 | MAI Image |
| MAI-Image-2.5-Flash | MAI-Image-2.5-Flash | MAI Image |

> `wan2.7-image-pro`／`wan2.7-image` 額外支援 `enable_sequential`（組圖模式：一次生成一組風格/角色連貫的故事圖組，開啟後 `n` 上限由 4 提高到 12，實際張數由模型決定）；`wan2.7-image-pro` 純文生圖情境下另支援 2K（`2048*2048`）、4K（`4096*4096`）高解析度輸出，其餘情境上游僅支援到 2K。

**圖像編輯 (I2I)**

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| wan2.7-image-pro | 萬相 2.7 Image Pro | 萬相圖像編輯 |
| wan2.7-image | 萬相 2.7 Image | 萬相圖像編輯 |
| wan2.6-image | 萬相 2.6 Image | 萬相圖像編輯 |
| qwen-image-2.0-pro（編輯） | 千問圖像 2.0 Pro | 千問圖像編輯 |
| qwen-image-2.0（編輯） | 千問圖像 2.0 | 千問圖像編輯 |
| MAI-Image-2.5-Pro（編輯） | MAI-Image-2.5-Pro | MAI Image |
| MAI-Image-2.5（編輯） | MAI-Image-2.5 | MAI Image |
| MAI-Image-2.5-Flash（編輯） | MAI-Image-2.5-Flash | MAI Image |
| dola-seedream-5.0-pro（編輯） | Seedream 5.0 Pro | ByteDance Seedream |
| dola-seedream-5.0-lite（編輯） | Seedream 5.0 Lite | ByteDance Seedream |

> qwen-image-2.0 系列為生成與編輯融合模型：最多 3 張參考圖、可一次輸出 1–6 張，並以 `prompt_extend` 取代 `ref_strength` 參數。

> **MAI Image 家族（2.5 / 2.5-Flash / 2.5-Pro）的尺寸不是固定枚舉，而是兩條同時成立的約束**（2026-08-10 對正式網關實測，三個型號行為一致）：**每邊至少 768 像素**，且**總像素不得超過 1,056,768**。違反時分別回 `'width'/'height' must be at least 768 pixels` 與 `Invalid dimensions WxH: total pixel count (N) exceeds the maximum of 1056768`。
>
> 先前這裡列的 `1536x1024` 與 `1024x1536` 都是 1,572,864 像素，**超過上限、一定會被拒**——三個尺寸裡有兩個從來就不能用。現在列的五個（`1024x1024`／`1366x768`／`768x1366`／`1152x896`／`896x1152`）都逐一實測確認可用，定義在 `_MAI_IMAGE_SIZES`。三個型號共用同一組尺寸，也都不支援 `ref_strength`。

**GPT Image（文生圖 + 圖像編輯，尺寸格式為 `WIDTHxHEIGHT`）**

| 模型 ID | 名稱 |
|---|---|
| gpt-image-2 | GPT Image 2（OpenAI 旗艦圖像模型） |
| gpt-image-1.5 | GPT Image 1.5（OpenAI 前代圖像模型） |

> 額外支援 OpenAI 標準的 `quality`（`auto`/`low`/`medium`/`high`）、`background`（`auto`/`opaque`/`transparent`，透明背景輸出）、`output_format`（`png`/`jpeg`/`webp`）三個參數。

**ByteDance Seedream（文生圖，尺寸格式同 GPT Image 為 `WIDTHxHEIGHT`，也接受 `2k`/`3k`/`4k` 預設值）**

| 模型 ID | 名稱 |
|---|---|
| dola-seedream-5.0-pro | Seedream 5.0 Pro（旗艦） |
| dola-seedream-5.0-lite | Seedream 5.0 Lite（輕量，畫面較大，最小約 2K／369 萬像素，太小的尺寸會被上游拒絕） |

> 兩個模型都同時支援圖像編輯 (I2I)，實測 `ref_strength` 參數有效、不會被拒絕，用法跟其餘走 `/v1/images/edits` 的模型一致。

**Gemini Image（文生圖 + 圖像編輯，走 Gemini 原生的 `/v1beta/models/{model}:generateContent`）**

| 模型 ID | 名稱 |
|---|---|
| gemini-3-pro-image | Gemini 3 Pro Image（旗艦，畫質最佳） |
| gemini-3.1-flash-image | Gemini 3.1 Flash Image（速度與品質平衡） |
| gemini-2.5-flash-image | Gemini 2.5 Flash Image（穩定版） |
| gemini-3.1-flash-lite-image | Gemini 3.1 Flash Lite Image（輕量極速） |

> **輸出尺寸與比例都以結構化參數控制**，文生圖與圖像編輯皆適用：`imageConfig.aspectRatio`（1:1／16:9／9:16／4:3／3:4）與 `imageConfig.imageSize`（1K／2K／4K）。
>
> `imageSize` 的支援度各型號不同（2026-08-10 對正式網關逐一實測，每個型號 × 每個值都實際產圖量過寬高）：
>
> | 模型 | 1K | 2K | 4K |
> |---|---|---|---|
> | `gemini-3-pro-image` | ✓ | ✓ | ✓ |
> | `gemini-3.1-flash-image` | ✓ | ✓ | ✓ |
> | `gemini-2.5-flash-image` | ✓ | 接受參數但靜默忽略，永遠回 1024 | 同左 |
> | `gemini-3.1-flash-lite-image` | ✓ | 回 400 | 回 400 |
>
> 後兩個型號的 `sizes` 因此只列 `1K`——列出來卻做不到的選項比沒有更糟。實際像素由 `imageSize` 與比例一起決定（例如 4K + 16:9 實測得到 5504×3072、2K + 9:16 得到 1536×2752）。
>
> **這裡先前是走 `/v1/chat/completions` + `modalities` 的**，那條路徑上結構化的 `imageConfig` 會被靜默忽略，只能用「在 prompt 文字裡以自然語言要求比例」的權宜做法（該做法實測確實有效：9:16 得到 768×1376），而且**完全無法控制解析度**；圖像編輯模式更是連比例參數都不處理，等於沒有任何輸出尺寸控制（使用者回報「gemini 3 pro image 選不了生成結果大小」即此）。原生端點兩個參數都真的生效，已改用。
>
> 注意原生端點**不接受 `candidateCount`**（送了直接 400），所以一次要多張時是並發打 n 次。另外 `/v1/images/generations` 不支援這些模型（回 `not supported model for image generation, only imagen models are supported`）。網關上 `gemini-3-pro-image-preview`、`gemini-3.1-flash-image-preview` 這兩個 preview 版模型已下線（實測回 404），故不列入清單。

圖片輸出支援點擊放大預覽（lightbox）。

### 影片生成

| 模型 ID | 名稱 | 分類 | 配音 |
|---|---|---|---|
| wan3.0-video | 萬相 3.0（文生／圖生／參考生／視頻編輯） | 萬相 3.0 | 自動 |
| wan2.7-t2v | 萬相 2.7 T2V | 文生影片 | 自動 |
| wan2.6-t2v | 萬相 2.6 T2V | 文生影片 | 自動 |
| wan2.7-i2v | 萬相 2.7 I2V | 圖生影片 | 自動 |
| wan2.6-i2v | 萬相 2.6 I2V | 圖生影片 | 自動 |
| wan2.6-i2v-flash | 萬相 2.6 I2V Flash | 圖生影片 | ✓ |
| wan2.7-r2v | 萬相 2.7 R2V | 參考生影片 | 自動 |
| wan2.6-r2v | 萬相 2.6 R2V | 參考生影片 | 自動 |
| wan2.6-r2v-flash | 萬相 2.6 R2V Flash | 參考生影片 | 自動 |
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
| bytedance-seedance-1.5-pro | Seedance 1.5 Pro | ByteDance Seedance | — |
| dreamina-seedance-2.0 | Seedance 2.0（即夢） | ByteDance Seedance | — |
| dreamina-seedance-2.0-fast | Seedance 2.0 Fast（即夢） | ByteDance Seedance | — |
| bytedance-seedance-1.5-pro（圖生影片） | Seedance 1.5 Pro | ByteDance Seedance | — |
| dreamina-seedance-2.0（圖生影片） | Seedance 2.0（即夢） | ByteDance Seedance | — |
| dreamina-seedance-2.0-fast（圖生影片） | Seedance 2.0 Fast（即夢） | ByteDance Seedance | — |
| bytedance-seedance-1.5-pro（參考生影片） | Seedance 1.5 Pro | ByteDance Seedance | — |
| dreamina-seedance-2.0（參考生影片） | Seedance 2.0（即夢） | ByteDance Seedance | — |
| dreamina-seedance-2.0-fast（參考生影片） | Seedance 2.0 Fast（即夢） | ByteDance Seedance | — |

> 萬相 2.6/2.7 系列 T2V/I2V/R2V 皆支援自動配音（BGM 自動生成或自訂音訊上傳）。
> **萬相 3.0（`wan3.0-video`）是 all-in-one 模型**——同一個模型 id 同時涵蓋文生／圖生／參考生／視頻編輯，UI 上以四個 type 分別呈現。最長 30 秒（其餘萬相家族是 15 秒），解析度 480P／720P／1080P，費率為每秒 $0.05／$0.10／$0.20。`ratio`（畫面比例）與 `resolution` 是兩個互相獨立的參數，預設 `adaptive`（其餘家族預設 16:9）。
>
> 因為模型名沒有 `i2v`／`r2v`／`videoedit` 後綴，上游無法從模型名判斷每個媒體的用途，改以「MIME／副檔名 ＋ 位置」推斷：data URI 取 `data:` 與第一個 `;`／`,` 之間的 MIME 前綴判定，HTTP URL 先切掉 query string 再比副檔名，判不出類型才回退到位置。我們送的 data URI 是判得出來的。
>
> 即使如此，本專案仍改走上游提供的覆寫管道，直接以 `metadata.input.media` 送出自己已標好 `type` 的陣列（`_apply_explicit_media()`），原因有二：**（1）位置推斷表達不了我們實際有的語意**——上游對影片一律推成 `video`（video-edit 的來源影片），永遠不會產出 `first_clip`（影片續寫的起始片段），而我們 i2v 的「影片延伸」模式送的正是 `first_clip`，這兩者在 wan2.7 是不同語意；**（2）除錯價值**——完全不依賴上游推斷，萬一實測失敗就能確定問題出在 `type` 詞彙本身而非推斷邏輯。上游確認這是預期用法。
>
> 官方模型頁已可查證的部分：定價 480P $0.05／720P $0.1／1080P $0.2（每秒）、輸入模態 Audio + Image + Text + Video、輸出 Video、最長 30 秒、「統一支援參考／編輯／複刻／驅動」。UI 的參考單價是從網關計費表讀的，不需要在這裡寫死。
>
> ⚠️ **`wan3.0-video` 尚未實際打過任何一次請求。** 它的官方 API 文檔目前是邀請制、尚未公開（模型頁上「立即體驗」是反灰的，只能「立即申請」）；可公開查證的只有上述那些，不含請求體的細節。`media` 的 `type` 詞彙（`first_frame`／`last_frame`／`driving_audio`／`first_clip`／`reference_image`／`video`）是閘道端實作者從 wan2.7 已公開文件推導的，兩邊都沒有驗證過。上架後必須實測；若上游回報 type 不合法，要把正確清單回報給閘道端校正 `wan3MediaType()`。實測時特別注意影片類請求：`first_clip` 與 `video` 是不同語意，失敗的話兩個值都值得各試一次。
>
> 動作動畫模型：視頻換人（將參考影片角色替換為人物圖片）、圖生動作（將參考影片動作遷移到人物圖片）。
>
> **萬相／HappyHorse 的三個上游限制**（已對照閘道 adaptor 原始碼確認，`app.py` 以 `audio` / `i2v_modes` / `ref_images_only` 三個旗標表示，前端據此收掉對應 UI）：
> 1. **配音只有 `wan2.6-i2v-flash` 有開關。** 阿里的 task adaptor 完全不讀統一請求的頂層 `audio` 欄位，整份程式只有 `wan2.6-i2v-flash` 會去讀 `metadata.audio`（關閉後費用減半）。其餘萬相型號有沒有聲音由上游自行決定，上表標示為「自動」。
> 2. **i2v 只讀首幀。** adaptor 的 i2v 分支只取 `images[0]` 當 `first_frame`，尾幀／驅動音訊／影片延伸片段都會被靜默丟棄，因此這些模型只開放「首幀生成」一種模式。
> 3. **r2v 只接受圖片。** 參考檔案會全部被當成參考圖（wan2.6 走 `reference_urls`、wan2.7/HappyHorse 走 `media` 的 `reference_image`），混入影片檔會被上游拒絕。
>
> **解析度／時長要三家分別送。** 影片端點是所有廠商共用的，但每家取值的欄位不同，漏送不會報錯、只會靜默用預設值（選 1080P 卻拿到 720P，而且照 720P 計費）：阿里讀頂層 `size`；Veo 的頂層 `size` 只認小寫 `x` 分隔的 `WIDTHxHEIGHT`、但 `metadata.resolution` 優先權最高；Seedance/Dreamina 完全不讀頂層 `size` 與 `duration`，只吃 `metadata.resolution` 與頂層 `seconds`（字串）。因此 `_apply_res_and_duration()` 一律把三種形式都送出去。畫面比例同理：Seedance 吃 `metadata.ratio`、Veo 吃 `metadata.aspectRatio`。
>
> **上傳的音訊必須是 URL。** 上游只接受 `audio_url`，base64 data URI 不會被解析，所以使用者上傳的配樂/驅動音訊會先經 `_cloud_put()` 放到雲端物件儲存再帶簽名網址過去；沒有設定任何雲端儲存後端時會直接回報錯誤，而不是送出一個註定無聲的請求。
> ByteDance Seedance 系列同時支援 t2v/i2v/r2v，走跟萬相系列共用的 `media`/`image`/`images` 三欄位注入機制；i2v 在 `bytedance-seedance-1.5-pro`、r2v 在 `dreamina-seedance-2.0-fast` 上實測完整跑到 `completed`，其餘模型 × 模式組合基於同一套機制推斷同樣可用，未逐一窮舉。**不支援**視頻編輯（vedit）——實測直接被上游拒絕（`image_url` 參數不合法）。`min_dur`/`max_dur` 沿用其他家族的常見範圍（2–15 秒），未測邊界值。
>
> **Seedance 各型號的解析度支援度不一樣**（2026-08-10 對正式網關逐一實測）。下表記錄的是**送出時上游是否接受這個參數值**（提交回 200 vs 回 `InvalidParameter`）：
>
> | 模型 | 480P | 720P | 1080P | 4K |
> |---|---|---|---|---|
> | `dreamina-seedance-2.0-fast` | 接受 | 接受 | **拒絕** | **拒絕** |
> | `dreamina-seedance-2.0` | 接受 | 接受 | 接受＊ | 接受 |
> | `bytedance-seedance-1.5-pro` | 接受 | 接受 | 接受 | **拒絕** |
>
> 被拒時回 `InvalidParameter`（`the parameter resolution ... is not valid`），`fast` 版的 t2v/i2v/r2v 三種模式都一樣。這個限制先前看不出來——解析度根本沒送到上游，一律當 720p 跑；是把解析度真的送達之後才浮現的。MODELS 以 `resolutions` 欄位限制住 `fast` 版的可選項，前端據此收掉選單裡的 1080P。
>
> ＊ **「接受」不等於「產得出來」**：`dreamina-seedance-2.0` @ 1080P 提交回 200，但該任務跑了約 15 分鐘後以 `failed` / `Unknown error` 收場，沒有拿到影片。只跑過這一次，無法區分是偶發失敗還是這個組合實際上不可用——**尚未端到端驗證**。相對地，`wan2.7-t2v` 與 `veo-3.1-fast-generate-001` 的 1080P 都已經下載成品用 `ffprobe` 量到實際 1920x1080。
>
> 4K 目前不在 UI 的選項裡，`dreamina-seedance-2.0` 的 4K 能力尚未開放給使用者選擇（也同樣只驗到「提交被接受」）。

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
| gemini-2.5-pro-tts | Gemini 2.5 Pro TTS | 語音合成 (Gemini) | Google 旗艦語音合成 |
| gemini-2.5-flash-tts | Gemini 2.5 Flash TTS | 語音合成 (Gemini) | Google 極速語音合成 |
| gemini-3.1-flash-tts-preview | Gemini 3.1 Flash TTS Preview | 語音合成 (Gemini) | Google 新一代極速語音合成（預覽版）|

TTS 的音色 (`voice`) 在主測試台與 AI Canvas 都是下拉選單、依選到的模型動態重建（`app.py` 的 `MODELS["voice"]["tts"][*].voices`）：
`qwen-audio-3.0-tts-plus`／`qwen-audio-3.0-tts-flash` 各自只支援自己專屬的官方音色（詳見 [Qwen-Audio-TTS 音色列表](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list)），不可混用；3 個 `gemini-*-tts` 模型則共用 [Gemini 官方 30 個音色](https://ai.google.dev/gemini-api/docs/speech-generation)。全部選單都可留空使用上游預設音色。

> ASR 走 NenAI 網關 OpenAI 相容的 `/v1/audio/transcriptions`；`qwen-audio-3.0-tts-*` 走 DashScope 風格的 `/v1/services/audio/tts/SpeechSynthesizer`（回傳 JSON，音檔網址在 `output.audio.url`），支援選填的 `voice`（CosyVoice v3 音色 id，例如 `longanlingxin`、`loongjohn`）、`instructions`（語氣風格描述）與 `sample_rate`/`volume`/`language_hints`（`metadata` 子欄位）；`gemini-*-tts*` 則是走 OpenAI 相容的 `/v1/audio/speech`（不是 Google 原生的 `/v1/text:synthesize`，那個路徑在這個網關上會直接回錯），只吃 `model`/`input`/`voice` 三個欄位（`voice` 例如 `Kore`），不支援 `instructions`（帶了會 400），`response_format` 也會被忽略、固定回傳 WAV。

---

## AI Canvas（`/canvas`）

節點式視覺化畫布，以拖拉連線的方式組合平台上的模型呼叫（類似 ComfyUI），基於 [litegraph.js](https://github.com/jagenjo/litegraph.js)。

| 節點 | 說明 |
|---|---|
| 文字 Text | 手動輸入 prompt，或選模型做真正的文字生成；若連接圖片輸入則可改用「分析圖片」 |
| 圖片 Image | 文生圖 (t2i)；若連接參考圖輸入則自動切換為圖像生成 (i2i)。跟主測試台圖片分頁同步支援：Gemini 系列的「圖片比例」選項、萬相 2.7 的組圖模式（`enable_sequential`，一次生成最多 12 張連貫圖組，多張結果以網格顯示）與 2K/4K 解析度、GPT Image 的 quality/background/output_format 三個參數 |
| 影片 Video | 依連接的圖片組合自動切換 t2v / i2v（首尾幀）/ r2v（最多 6 張參考圖）。原本規劃的「影片延伸」（接上一段影片繼續生成）功能因正式環境尚未接上持久化雲端儲存、容器多實例間抓不到剛生成的檔案而暫時關閉，程式碼保留在 `static/js/canvas.js` 的 `VIDEO_EXTEND_ENABLED` 開關後面，等正式環境接上雲端儲存後再開放 |
| 圖像編輯 Editing | 圖像編輯 (i2i)，需連接一張來源圖片；GPT Image 系列同樣支援 quality/background/output_format |
| 語音 TTS | 呼叫語音模型分頁同一套 `/api/voice/tts`，可接文字節點輸出或手動輸入；依選擇的模型（qwen-audio-3.0-tts-* / gemini-*-tts）動態顯示或隱藏 CosyVoice 專屬的進階參數 |
| MuleAI Spicy | 對應 NenAI Spicy 四個模型，依選擇的模型動態切換必填輸入與輸出型別（image/video） |

節點之間可用連線傳遞文字/圖片/影片輸出，設定面板僅在節點被選取時以固定大小浮層顯示於節點下方。

工具列的「範本」按鈕內建 5 個常見節點組合，一鍵套用可省去手動拉線：文字→圖片生成、圖片生成→圖像編輯、文字腳本→語音配音、文字→圖片+語音雙路輸出、上傳圖片→圖生影片。

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
