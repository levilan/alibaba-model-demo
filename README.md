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
cd nenai-playground
docker compose up -d --build
```

瀏覽器開啟 `http://localhost:5050`，輸入 NenAI API Key 登入。

**常用指令**

```bash
# 查看日誌
docker logs -f nenai-playground-ai-model-tester-1

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
| AI Canvas | 節點式視覺化畫布（`/canvas`），可拖拉連線組合文字／圖片／影片／圖像編輯／語音／MuleAI 節點，串接多個模型呼叫；支援一鍵依順序執行整張圖 |

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
| glm-5.1 | GLM 5.1 | 第三方 | enable_thinking + reasoning_effort（6 段） |
| glm-5.2 | GLM 5.2 | 第三方 | enable_thinking + reasoning_effort（7 段） |
| dola-seed-sc | Seed SC | ByteDance | — |
| dola-seed-2.0-lite | Seed 2.0 Lite | ByteDance | — （無條件思考，關不掉） |
| dola-seed-2.0-pro | Seed 2.0 Pro | ByteDance | — （無條件思考，關不掉） |
| claude-opus-5 | Claude Opus 5 | Claude | — |
| claude-opus-4-8 / 4-7 / 4-6 / 4-5 / 4-1 | Claude Opus 系列 | Claude | — |
| claude-sonnet-5 / 4-6 / 4-5 | Claude Sonnet 系列 | Claude | — |
| claude-haiku-4-5 | Claude Haiku 4.5 | Claude | — |
| claude-fable-5 | Claude Fable 5 | Claude | — |
| gpt-5.6-terra / sol / luna | GPT 5.6 特化系列 | GPT | reasoning_effort |
| gpt-5.5 / 5.4 / 5.4-mini / 5.4-nano / 5.2 / 5-mini | GPT 5.x 系列 | GPT | reasoning_effort |
| gemini-3.7-flash 等 9 個 Gemini 模型 | Gemini 系列 | Gemini | thinkingConfig（走原生 API） |
| kimi/kimi-k3 | Kimi K3 | 月之暗面 | 純思考，關不掉；回 reasoning_content（支援圖片輸入） |
| grok-4.3 | Grok 4.3 | xAI Grok | reasoning_effort（5 段，支援看圖） |
| grok-4-20-reasoning / -non-reasoning | xAI Grok | xAI Grok | reasoning_effort（僅 -20-reasoning 有效） |
| grok-4-1-fast-reasoning / -non-reasoning | xAI Grok | xAI Grok | — |
| qwen3-vl-plus / qwen3-vl-flash | 視覺語言 | 視覺語言 | —（支援圖片輸入） |

> **思考模式的三種機制，不能混用**：Qwen/DeepSeek/GLM 用布林值 `enable_thinking`（這幾家幾乎都實測預設就是開啟，必須明確送 `enable_thinking:false` 才會關閉並省 token——完全不帶這個欄位並不會關閉思考，後端一律會明確帶上 `true`/`false`，只有 GPT 系列例外不帶）；GPT 系列改用字串 `reasoning_effort`（實測這個網關接受的枚舉是 `none/low/medium/high/xhigh`，跟 OpenAI 官方文件常見的 `minimal/low/medium/high` 不同，帶錯值或帶 `enable_thinking` 給 GPT 都會被直接拒絕）；**GLM 5.x 是唯一兩種都支援的家族**，詳見下方；Claude 與 ByteDance Seed 2.0 系列目前實測無法透過這個網關控制（Claude 送了無效、Seed 2.0 系列無條件思考關不掉），因此 UI 不提供對應開關；**Gemini 系列改走原生 API 後已經可以控制**，詳見下方；`qwen3-coder-plus`/`qwen3-coder-flash` 則是實測 `enable_thinking` 完全沒有效果（true/false 都不會有思考過程），同樣不顯示開關。開啟後若上游回傳 `reasoning_content`（Qwen/DeepSeek/GLM 大部分模型，以及無法關閉思考的 Seed 2.0 系列），會在回答上方顯示成可收合的「思考過程」區塊。

> **Gemini 文字模型走 Gemini 原生 API**：`/v1beta/models/{model}:generateContent`，串流是 `:streamGenerateContent?alt=sse`（見 `_GEMINI_NATIVE_TEXT_MODELS`）。後端把請求轉成原生格式（`system_prompt` → 頂層 `systemInstruction`、`max_tokens` → `generationConfig.maxOutputTokens`、`stop` → `stopSequences`、`top_p`/`top_k` → `topP`/`topK`），並把回應轉回前端既有的協定。
>
> **改走原生 API 換到的能力**：先前走 OpenAI 相容端點時，Gemini 的思考過程既看不到也關不掉（本文件一度據此寫「Gemini 3.x 無條件思考、關不掉」）。原生端點兩件事都做得到——`thinkingConfig.includeThoughts` 會回傳帶 `"thought": true` 的 content part（就是思考全文，串流時也逐段送），`thinkingConfig.thinkingBudget: 0` 能真的關掉思考。實測同一題關閉後 completion token 從 170～210 降到 **1**，省下來的是實際費用。
>
> 但支援度**各型號不同**，送錯會直接 400，所以有兩個例外（2026-08-10 逐一實測）：
>
> | 模型 | 顯示思考過程 | 關閉思考 | UI 開關 |
> |---|---|---|---|
> | `gemini-3.1-pro-preview`、`3.6-flash`、`3.5-flash`、`3-flash-preview`、`2.5-flash` | ✓ | ✓ | 有 |
> | `gemini-2.5-pro` | ✓ | ✗ `The model does not support setting thinking_budget to 0` | 無（關不掉，一律顯示過程） |
> | `gemini-3.7-flash` | ✓ | △ **大幅降低但不歸零** | 有 |
> | `gemini-2.5-flash-lite` | ✗ `Thinking_config.include_thoughts is not supported` | ✓ | 有（但不顯示思考區塊） |
>
> 兩個 `flash-lite` 還有一個共同陷阱：它們的思考**預設是關的**，不帶 `thinkingConfig` 時 `thoughtsTokenCount` 欄位根本不出現。開關要送 `thinkingBudget: -1`（動態預算）才會真的啟動思考，否則那個開關會完全沒有作用（實測開/關都是 1 個 token）。這也是 `_GEMINI_THINKING_OFF_BY_DEFAULT` 這個集合存在的原因。
>
> **但兩者對固定的 `thinkingBudget` 反應完全不同**——這格一定要重複取樣才看得出來（每格 8 次，統計「有思考的次數」）：
>
> **`gemini-3.5-flash-lite` 的門檻是 101**（`100` → 0/8 不思考、`101` → 會思考；≤100 是**合法但靜默不思考**，不報錯。送超大值的錯誤訊息會吐出可接受範圍 `integers from 1 to 32768`）。**`gemini-2.5-flash-lite` 的範圍是 `512 to 24576`**，低於 512 直接 400、達到就穩定思考。門檻值與範圍由本專案與文檔站兩邊獨立實測一致。
>
> ⚠️ **但「超過門檻之後會不會真的思考」取決於 prompt，不是取決於預算大小。** 模型自己判斷這一題需不需要推理。同一輪背對背、每格 8 次：
>
> | 提示詞類型 | `101` | `128` | `512` | `-1` |
> |---|---|---|---|---|
> | 需多步算術 | 8/8 | 8/8 | 8/8 | 8/8 |
> | 短句陷阱題 | 3～7/8 ⚠️ | 4～6/8 ⚠️ | 4～7/8 ⚠️ | 8/8 |
> | `2+2` | **0/8** | **0/8** | **0/8** | 8/8 |
> | 單句改寫 | **0/8** | **0/8** | **0/8** | 8/8 |
>
> （前兩列是本專案與文檔站各自實測，陷阱題的比率兩邊不同但同一個現象；後兩列由文檔站實測，📄 轉述。）
>
> **最後兩列是決定性的**：`2+2` 與單句改寫在**任何固定預算下都是乾淨的 0/8**，預算給多大都一樣。所以 `thinkingBudget` 給一個數字的語意是「**允許**思考，要不要用由模型看題目決定」，不是「啟用思考」。邊界題目（陷阱題）不穩定只是這個機制的副作用，比率本身不必當成數字看。
>
> `thoughtsTokenCount` 也有系統性差異：多步推導的題目 210～369、陷阱題 144～180。
>
> **只有 `thinkingBudget: -1` 是可靠的，而且它不等於「一個很大的固定預算」**——實測連 `2+2` 這種一定不需要推理的題目，`-1` 也是 8/8 都思考（本專案量到 106～128 tokens、文檔站 94～123，一致），而**固定預算 `512` 在同一題上是 0/8**。兩者的語意根本不同：`-1` 是真的把動態思考打開，固定數字只是給一個額度。
>
> 所以：**如果需要「每次回應都經過思考」（例如據此保證推導品質、或用有無思考來做分支），一定要送 `-1`，不要給固定數字。** 我們平台的「思考開」送的就是 `-1`，不受這個問題影響。
>
> `thinkingBudget` 同樣**不是上限而是傾向**：有思考時的 `thoughtsTokenCount` 落在 100～370，跟給 128 還是 2048 沒有關係，跟 Grok 的 `reasoning_effort` 是同一回事。
>
> 📌 這張表改過兩次，兩次都是取樣方式的問題：
> 1. 最初寫成「`128` 靜默忽略、實務門檻約 512」——那是**每格只取樣一次**的產物，兩格剛好都抽到不思考的那次，湊出一個不存在的門檻。
> 2. 改成 8 次取樣後寫成「超過門檻後**逐次隨機**、機率約四成、跟預算無關」——次數夠了，但**只用了一個 prompt**，把 prompt 相依的行為寫成了模型的固有屬性。文檔站用另一個 prompt 得到 8/8，兩邊背對背重跑才定位到真正的變因。
>
> 教訓：樣本數解決雜訊，解決不了「變因沒有被控制」。詳見 `memory.md` 4d。
>
> 計費上要注意 `thoughtsTokenCount` 也是實際收費的輸出 token，後端的 `usage.completion_tokens` 是 `candidatesTokenCount + thoughtsTokenCount`，否則「本次花費」會嚴重低估。

> **xAI Grok 的 reasoning／non-reasoning 是兩個獨立型號，不是同一模型的參數。** 2026-08-11 對正式環境實測四個型號：
>
> | 型號 | 預設是否推理 | `reasoning_effort` | `enable_thinking` |
> |---|---|---|---|
> | `grok-4-20-reasoning` | 會（`reasoning_tokens` 203） | ✅ `none` → 0 | ✗ 無效（送 false 反而 363） |
> | `grok-4-1-fast-reasoning` | 會 | ✗ 無效（各 3 次中位數 176 vs 245） | ✗ 無效 |
> | `grok-4-20-non-reasoning` | 不會（恆為 0） | ✗ 回 400 | ✗ 無效 |
> | `grok-4-1-fast-non-reasoning` | 不會 | ✗ 回 400 | ✗ 無效 |
>
> **`grok-4.3`（2026-08-11 新增）是唯一有完整強度分段、而且支援看圖的 Grok**：`reasoning_effort` 枚舉是 `none`／`minimal`／`low`／`medium`／`high`（`xhigh` 與 `max` 回 422）；圖片輸入實測可用（標了 `vision`）。它一樣不回傳 `reasoning_content`。
>
> **`reasoning_effort` 是「傾向」不是預算上限，而且分段之間分不太出來。** 同一題每檔重跑 5 次的 `reasoning_tokens`：
>
> | effort | 5 次實測值 | 中位數 | 答案 |
> |---|---|---|---|
> | `none` | 0, 0, 0, 0, 0 | 0 | 5 次全錯 |
> | `minimal` | 282, 292, 339, 379, 334 | 334 | 5 次全對 |
> | `low` | 205, 353, 272, 232, 277 | 272 | 5 次全對 |
> | `medium` | 359, 528, 527, 615, 608 | 528 | 5 次全對 |
> | `high` | 265, 418, 489, 366, 336 | 366 | 5 次全對 |
>
> 三件事：①`medium` 的整個範圍都高於 `high`，**不是單調遞增**，所以不能把 effort 當 token 預算用；②`minimal`／`low`／`high` 三檔區間大幅重疊，實務上分不出差別；③**同檔內單次差距可達 1.7 倍**（`low` 205～353），任何單次觀測都不足以下結論——這點的教訓寫在 `memory.md` 4d。
>
> `none` 除了把推理 token 歸零，也**穩定改變答案品質**：測試題（17 隻羊、除了 9 隻以外都跑走）正解是 9，`none` 五次全答 8、其餘四檔五次全答 9。省 token 是有代價的。
>
> ⚠️ **Grok 的推理 token 不計入 `completion_tokens`**（實測 `grok-4.3`：prompt 31 + completion 1 + reasoning 844 = total 876），但那些 token 照樣收費。後端的 `_openai_usage()` 改成以 `total - prompt` 反推 completion，對兩種帳法都正確——其餘家族（DeepSeek V4／GLM／Seed 2.0）的推理 token 本來就含在 `completion_tokens` 裡，直接相加會變兩倍。沒有這層處理，一次花 844 個推理 token 的呼叫會被算成 1 個，「本次花費」嚴重低估。
>
> 其餘四個都**不回傳 `reasoning_content`**，所以思考過程一律看不到，`thinking` 旗標全部為 `False`。只有 `grok-4-20-reasoning` 給 `reasoning_effort` 控制項；送非法值只回通用的 `openai_error`、問不出合法枚舉，所以 `reasoning_efforts` 只列實測有效的 `none`。
>
> **以上都是上游 xAI 的行為，不是網關的映射問題**（已由網關端查證）：網關對 `grok-4-*` 一律原樣轉發、不碰 `reasoning_effort`；`reasoning_content` 網關兩種欄位名（`reasoning_content` / `reasoning`）都會解析，是 xAI 對 grok-4 系列不回傳推理過程。錯誤訊息的枚舉也是上游回的（GLM 那句列出合法值的訊息來自智譜），網關只是轉發。所以同家族內行為不一致這件事，是 xAI 的產品決策，我們與網關都無法抹平。

> ⚠️ **影片輸入必須是雲端網址，不能用 base64 data URI。** 送 `data:video/mp4;base64,...` 時**提交會回 200**，但任務輪詢階段才失敗：`InvalidVideo.FileFormat: Invalid video type. Only mp4/mov/avi is supported.`（2026-08-11 用 `wan2.2-animate-move` 對正式環境實測確認）。這跟音訊是同一類限制——圖片可以用 data URI，影片與音訊都必須是上游抓得到的 URL。
>
> 受影響的是**視頻編輯的來源影片、動作動畫的參考影片、i2v 影片延伸的起始片段、r2v 帶入的參考影片**，都已改成先經 `_upload_video_for_url()` 上傳雲端再帶簽名網址。
>
> **但這需要有雲端物件儲存才會生效** —— 而 2026-08-11 實測**正式環境目前沒有設定**（產出的圖片 `local_path` 是 `/outputs/images/...` 本機路徑，不是簽名網址）。在設定好之前，這些影片功能會回一個明確的 400 錯誤而不是靜默失敗，但仍然不能用。
>
> 沒有雲端儲存另外還有兩個既有風險：**（1）** Cloud Run 每個實例的檔案系統獨立，`maxScale` 是 5，產出的圖片存在某個實例上、下次請求被路由到別的實例就會 404（實測當下只有單一實例在服務，問題尚未浮現但風險存在）；**（2）** 容器重啟後產出全部消失。要根治就是把 OSS／S3／GCS 任一個的憑證設進 Cloud Run。

> **realtime（即時語音對話）2026-08-16 起可用**，在「語音模型」分頁的任務類型選「即時語音對話 (Realtime)」。先前不可用的原因（阿里 adaptor 沒有 realtime 分支，握手成功後立刻斷線）已由閘道端修掉。
>
> 實測（正式環境）：握手 34ms、文字輸入到首包音訊 2.1s、產出 24kHz 音訊非靜音、事件序與 OpenAI realtime 相容。上行 PCM **16kHz**、下行 PCM **24kHz**，都是 mono s16le 裸流（下行沒有 wav 檔頭）。
>
> 路徑是 `/v1/realtime`。**不要寫成 `/api-ws/v1/realtime`**——那是 DashScope 上游的內部路徑、不是閘道對外的路由，會回 404。
>
> 瀏覽器走後端的 `/ws/omni` 代理，不直連閘道：WebSocket 建構子不能帶 header，直連只能把金鑰塞進子協定（`openai-insecure-api-key.<key>`），那會讓金鑰出現在前端可見的握手參數裡。
>
> **若日後要加「上傳音檔」這條輸入路徑**：音檔尾端必須有 1～2 秒靜音，否則 `semantic_vad` 不會斷句——逐字稿會正常出、`speech_started` 也會來，但 `speech_stopped` 永遠不來，一路等到逾時。閘道端實測：1.96 秒無尾靜音的檔案卡滿 60 秒，補 2 秒靜音後同一個檔案立刻正常（`ffmpeg -af "apad=pad_dur=2"`）。目前的即時麥克風輸入天然有靜音，走不到這條路徑，所以我們自己沒有複驗過。
>
> **音色 56 個都逐一實測過**（`app.py` 的 `_QWEN35_OMNI_REALTIME_VOICES`）：送 `session.update` 帶音色再 `response.create`，有效的會開始回傳音訊、無效的回 `Voice 'X' is not supported.`。舊清單裡的 `Chelsie` 實測不支援（那是 qwen2.5-omni 的音色）已移除。⚠️ **不要用 `session.update` 的回應來驗**——它對任何字串都回 `session.updated`，連亂編的名字都照收；也不要用音訊位元比對，同音色同輸入重跑兩次的位元並不相同。

> **`qwen3-vl-plus` / `qwen3-vl-flash` 是視覺語言模型**，可在對話中帶入圖片。用標準的 OpenAI `image_url` 格式（實測 data URI 可用），MODELS 以 `vision: True` 標記、前端據此顯示圖片上傳欄位。圖片只附在**當下這一輪**的提問上，不會進入對話歷史——上游對「歷史訊息裡的圖片」的行為未驗證，不主動送。切換到不支援視覺的模型時前端會清掉已選的圖，避免靜默夾帶造成 400。

> **GLM 5.x 有兩層思考控制**：布林的 `enable_thinking` 與字串的 `reasoning_effort`（分段推理強度）。實測各段的 `reasoning_tokens`：`none`／`minimal` → 0、`low` 182、`medium` 198、`high` 202、`xhigh` 239、`max` 208。可用枚舉各型號不同——`glm-5.2` 七段全有，`glm-5.1` **不支援 `max`**（送了回 400 並列出正確清單），故以 MODELS 的 `reasoning_efforts` 標明、前端選項依此動態產生。
>
> 兩者同時送時，**`enable_thinking:false` 的優先權高於 `reasoning_effort`**；但反向不成立——實測 `enable_thinking:true` 配 `reasoning_effort:none` 仍然不會思考，也就是「關」的那一方永遠贏。
>
> **`glm-5.2` 曾短暫改走 Anthropic Messages 格式的 `/v1/messages`，已改回 OpenAI 相容路徑。** 原因是那條路徑上**整個 `thinking` 物件完全無效**：`type=disabled` 不會關閉思考、`budget_tokens` 128 與 2048 的輸出 token 中位數（195 vs 178）完全重疊、`reasoning_effort` 回 200 但被靜默忽略。經 nen-ai-platform 端查證，根因在 `service/convert.go` 的 `ClaudeToOpenAIRequest`——`claudeRequest.Thinking` **只在 `isOpenRouter` 分支被讀取**，其餘所有渠道一律丟棄；`reasoning_effort` 更是連 `dto.ClaudeRequest` 都沒有這個欄位，反序列化階段就消失了。等網關補上渠道閘控的映射後可再評估是否切回。

### 圖片生成

**文生圖 (T2I)**

| 模型 ID | 名稱 | 分類 |
|---|---|---|
| qwen-image-3.0-pro | 千問圖像 3.0 Pro | 千問文生圖 |
| qwen-image-3.0 | 千問圖像 3.0 | 千問文生圖 |
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
| qwen-image-3.0-pro（編輯） | 千問圖像 3.0 Pro | 千問圖像編輯 |
| qwen-image-3.0（編輯） | 千問圖像 3.0 | 千問圖像編輯 |
| qwen-image-2.0-pro（編輯） | 千問圖像 2.0 Pro | 千問圖像編輯 |
| qwen-image-2.0（編輯） | 千問圖像 2.0 | 千問圖像編輯 |
| MAI-Image-2.5-Pro（編輯） | MAI-Image-2.5-Pro | MAI Image |
| MAI-Image-2.5（編輯） | MAI-Image-2.5 | MAI Image |
| MAI-Image-2.5-Flash（編輯） | MAI-Image-2.5-Flash | MAI Image |
| dola-seedream-5.0-pro（編輯） | Seedream 5.0 Pro | ByteDance Seedream |
| dola-seedream-5.0-lite（編輯） | Seedream 5.0 Lite | ByteDance Seedream |

> qwen-image-2.0 系列為生成與編輯融合模型：最多 3 張參考圖、可一次輸出 1–6 張，並以 `prompt_extend` 取代 `ref_strength` 參數。

> **千問 3.0 系列（`qwen-image-3.0` / `-pro`）的尺寸是一條面積約束，不是固定枚舉**（2026-08-11 對正式網關實測，兩個型號一致、上下界都驗過）：**總像素 262,144（512×512）～ 6,553,600（2560×2560）**，格式必須是 `寬*高`。送 `1K`/`2K`/`4K` 這種規格值會被拒（`Expected format: '<width>*<height>'`）——這點跟萬相 2.7 相反，別混用。
>
> 錯誤訊息寫的是「for **t2i** requests」，i2i 不套用這條——實測 i2i 送 `size=10*10` 照樣成功產圖。行為與 2.0 系列一致：一次可輸出 1–6 張、參考圖最多 3 張（上游規定 `input.messages` 只能一則，I2I 的 `content` 是 1～3 個 image ＋**恰好一個** text）、不支援 `ref_strength`，改以 `prompt_extend` 控制。
>
> ⚠️ **計價要注意**：`/api/pricing` 讀到的是**1K 輸出價**（3.0 每次 $0.03、3.0-pro $0.04）。輸出 2K 時網關會依上游回傳的 `usage.output_image_type` 自動補倍率——**`qwen-image-3.0-pro` 的 2K 是 $0.075，接近一倍**，但 UI 顯示的參考單價不會反映這件事。另外輸入圖是**加法**附加費 $0.003/張，不受輸出張數或解析度倍率影響。
>
> `qwen-image-3.0-pro` 的官方 **RPM = 1**，連續呼叫很容易撞到 429，測試時要留重試間隔。

> ⚠️ **`n > 1` 時 OpenAI 相容的 `data[]` 只會回第一張**（千問 3.0 這條 multimodal-generation adaptor 特有；萬相與千問 2.0 走另一條 adaptor，`data[]` 本來就完整）。實測 `n=2`：`data[]` 只有 1 筆，另一張在 `metadata.output.choices[].message.content` 裡（上游確實產了 2 張，`usage.output_image_count` 也是 2）。
>
> 後端已用 `_extract_images_from_metadata()` 從 metadata 補齊，讓使用者看得到全部的產出。
>
> **計費方向要講清楚，不要弄反**：網關是**數 `data[]` 裡的實際圖片數**來計費（`aliImageHandler` 的 `actualImageCount`），不是看 `usage.output_image_count`。所以在網關修好之前，使用者**只被收一張的錢**、沒有被多收——真正吃虧的是平台方（上游按 2 張收，平台只收使用者 1 張）。這一點我最初判斷反了，實際是由網關端查用量日誌釐清的（`quota 15000` = $0.03 × QuotaPerUnit，就是一張的價錢）。
>
> 附帶影響：在網關修好之前，我們補齊後顯示 2 張，但實際只被收 1 張，所以 header 的「本次花費」在這個情境會**高估**。網關端的修復（每張圖產出一筆 `data`）完成部署後，交付與計費都會是 2 張，估算就會與實際一致。

> **圖像編輯情境下 `size` 是不生效的**（2026-08-11 實測，跨兩個家族三個模型確認）：送 `1280*720`（橫向）與 `512*512` 得到的輸出完全相同，連 T2I 的面積約束都不套用（送 `10*10` 也不會被拒）。
>
> **輸出尺寸由輸入圖的長寬比決定，不是固定值**——我最初量到「一律 2048×2048」是因為測試用的輸入圖剛好是正方形；文檔站用 900×506（16:9）的輸入圖複驗，兩種 size 都得到 **2720×1520**（≈16:9，與輸入一致）。**請以回應的 `usage.output_width` / `output_height` 為準，不要假設是固定值。**
>
> 兩種情況都被歸為 `qima_output_2k` 照 2K 計費，所以 `qwen-image-3.0-pro` 在編輯情境下實質一律是 $0.075／張。
>
> | 模型（編輯） | 送 `size=1280*720` 的實際輸出 |
> |---|---|
> | `qwen-image-3.0` | 2048×2048 |
> | `qwen-image-2.0` | 1024×1024（跟隨輸入圖） |
> | `wan2.7-image` | 2048×2048 |
>
> 千問 3.0 的兩個編輯條目已標 `no_size`，UI 不再顯示那個沒有作用的選單。**`qwen-image-2.0` 與萬相編輯系列目前仍然顯示尺寸選單，那是既有問題**——同樣不會生效，但改動會影響現有使用者的介面，尚未處理。

> **多張參考圖必須用「重複的 `image` 欄位」送 multipart，不能用 `image_2`／`image_3` 這種編號欄位名。** 2026-08-10 實測（兩張純色圖 + 要求模型混色，量輸出的平均 RGB）：`image` + `image_2` 得到紅色（**只有第一張生效，第二張被靜默丟棄**）；重複的 `image` 或 `image[]` 得到紫色（兩張都吃）。這是使用者實際回報的問題（「wan2.7 上傳兩張照片，似乎只會吃第一張」）。已改用重複 `image`，並在 `wan2.7-image`／`wan2.6-image`／`qwen-image-2.0`／`gpt-image-2`／`dola-seedream-5.0-pro` 上都驗過。
>
> **各編輯模型的參考圖張數上限差很多**，已逐一實測（2026-08-10，正式網關）。做法是「前 N 張純紅 + 最後一張純藍 + 要求模型輸出所有參考圖的混色」，量輸出圖片的平均 RGB——出現藍色成分就代表最後那張真的被讀進去了。**只驗「送得出去」是不夠的**，上游可能接受請求卻靜默忽略多出來的圖。
>
> | 模型 | 上限 | 超過時的上游訊息 |
> |---|---|---|
> | `wan2.7-image` / `wan2.7-image-pro` | 9（第 9 張實測有效） | — |
> | `wan2.6-image` | **4** | `the last message must contain 1 to 4 images` |
> | `qwen-image-2.0` / `-pro` | **3** | `supports 0~3 image content items` |
> | `MAI-Image-2.5` / `-Flash` / `-Pro` | **1** | `Exactly one image file must be attached for edit requests` |
> | `gpt-image-2` / `-1.5`、Seedream、Gemini | 9（第 9 張實測有效） | — |
>
> 上限以 MODELS 的 `max_ref` 為單一來源，後端 `_EDIT_MAX_REF` 與前端 `imgMaxRef` 讀同一份資料，未標的模型沿用 9 張。

> **注意**：萬相編輯先前在這裡被標成「最多 2 張」，那是依閘道端 `WanImageInput.images (≤2)` 這個 Go struct 推斷的、沒有實測——**實測後確認那條約束不適用於 `/v1/images/edits` 這條路徑**，萬相 2.7 實際上 9 張都會生效。這個錯誤的限制曾讓使用者反映「wan2.7 nen 只能支援兩張上傳，之前阿里的可以上傳到 9 張」。

> **MAI Image 家族（2.5 / 2.5-Flash / 2.5-Pro）的尺寸不是固定枚舉，而是兩條同時成立的約束**（2026-08-10 對正式網關實測，三個型號行為一致）：**每邊至少 768 像素**，且**總像素不得超過 1,056,768**。違反時分別回 `'width'/'height' must be at least 768 pixels` 與 `Invalid dimensions WxH: total pixel count (N) exceeds the maximum of 1056768`。
>
> 兩條限制**互相獨立**，只檢查總像素會誤判：`767x1024` 的總像素只有 785,408、遠低於上限，照樣因為短邊不足被拒。先前這裡列的 `1536x1024` 與 `1024x1536` 都是 1,572,864 像素，**超過上限、一定會被拒**——三個尺寸裡有兩個從來就不能用。
>
> **上游還會把尺寸往下對齊到 16 的倍數。** 實測請求 `1366x768`（Azure 官方文件自己舉的例子）拿回來的是 **1360x768**——這是讀 PNG header 確認的，不是看回應欄位；回應本身不會提示尺寸被改過。所以 `_MAI_IMAGE_SIZES` 直接登記對齊後的 `1360x768`／`768x1360`（實測上游照收，且輸出與請求完全相符），使用者選什麼就拿到什麼。其餘三個尺寸本來就是 16 的倍數，不受影響。
>
> 現在列的五個（`1024x1024`／`1360x768`／`768x1360`／`1152x896`／`896x1152`）都逐一實測確認可用，定義在 `_MAI_IMAGE_SIZES`。三個型號共用同一組尺寸，也都不支援 `ref_strength`。
>
> **MAI 的尺寸不是固定枚舉，而是「滿足約束的任意值」，所以 UI 另外開放自訂寬高。** 實測 `size="1200x800"`（不在清單裡）輸出就是 1200×800；經由平台送 `size="1088x960"` 輸出 1088×960，都完全相符。三個 t2i 型號帶 `custom_size`（`_MAI_CUSTOM_SIZE`：`min_side` 768／`max_pixels` 1056768／`align` 16），前端據此即時驗證並**自動往下對齊到 16 的倍數、明確顯示「實際輸出 W×H」**——是告知後修正，不是靜默改寫。編輯路徑沒有加，因為 `size` 在編輯情境是否生效尚未驗證，不加無效控制項。
>
> **自訂尺寸有兩條路，都實測可用，UI 讓使用者自行選擇**（`custom_size.modes`）：
>
> | 送出方式 | 請求欄位 | 實測（經由平台，讀 PNG header） |
> |---|---|---|
> | `size` 字串 | `size: "1024x976"` | 輸出 1024×976 |
> | `width`／`height` | `width: 1024, height: 976` | 輸出 1024×976 |
>
> 兩者效果相同。**同時帶入時上游以 `width`/`height` 為準**——實測 `size=1024x1024`（合法）配 `width/height=2000x2000`（超限）會回 400 而不是照 `size` 產圖。所以後端是二選一送出：帶了 `width`/`height` 就把 `size` 拿掉，讓送出的請求跟使用者選的機制一致，看日誌時不會誤以為兩個都在生效。**對齊行為兩條路一致**，`width`/`height` 同樣會往下對齊到 16 的倍數（送 1366 得 1360）。
>
> 📌 `width`/`height` 這條路一度是**無效**的：更早實測送 `{"width":2000,"height":2000}`（400 萬像素、若被讀取必定超限報錯）竟正常產出 1024×1024。根因是閘道的 `dto.ImageRequest` 沒有宣告 `width`/`height`，未宣告的欄位會落進 `Extra` map，而 `MarshalJSON` **刻意不把 `Extra` 合併回去**，於是欄位在反序列化階段就消失、全程沒有任何錯誤訊息。閘道端已改成從 `Extra` 取值（且刻意仍**不**宣告在 DTO——那個結構是所有 image 渠道共用的，一宣告就會把 `width`/`height` 透傳給 dall-e 這類只認 `size` 的上游，讓原本能跑的請求變 400），現已部署到正式環境。這個「沒宣告 = 靜默丟棄」的機制記在 `memory.md` 第 5 條。

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

> **輸出尺寸與比例都以結構化參數控制**，文生圖與圖像編輯皆適用：`imageConfig.aspectRatio` 與 `imageConfig.imageSize`。
>
> **這組參數不是直接指定寬高**——`imageSize` 是**總像素預算**、`aspectRatio` 是形狀，上游取「符合該比例、且總像素最接近預算」的一組寬高，兩邊都對齊到 16 的倍數。以 1K 實測：
>
> | 比例 | 輸出 | 總像素 | 比例 | 輸出 | 總像素 |
> |---|---|---|---|---|---|
> | 1:1 | 1024×1024 | 1,048,576 | 21:9 | 1584×672 | 1,064,448 |
> | 4:3 | 1200×896 | 1,075,200 | 3:4 | 896×1200 | 1,075,200 |
> | 3:2 | 1264×848 | 1,071,872 | 2:3 | 848×1264 | 1,071,872 |
> | 5:4 | 1152×928 | 1,069,056 | 4:5 | 928×1152 | 1,069,056 |
> | 16:9 | 1376×768 | 1,056,768 | 9:16 | 768×1376 | 1,056,768 |
>
> 2K 是 1K 的 4 倍像素、4K 是 16 倍。**注意「4K」是像素預算而不是 UHD 解析度**——4K + 16:9 實測是 5504×3072（約 1,690 萬像素），不是 3840×2160。這個 API 沒有直接指定寬高的方式,想要精確尺寸得自己換算比例。
>
> `aspectRatio` 四個型號都支援全部 10 種（1:1／16:9／9:16／4:3／3:4／3:2／2:3／5:4／4:5／21:9，含最偏門的 21:9 都實測過；`gemini-2.5-flash-image` 的量化略有不同，21:9 給的是 1536×672）。非法值會回 400、不會被靜默忽略,但錯誤訊息是通用的 `Request contains an invalid argument.`，看不出是哪個欄位有問題。
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
| bytedance-seedance-1.5-pro | Seedance 1.5 Pro | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.5 | Seedance 2.5（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0 | Seedance 2.0（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0-fast | Seedance 2.0 Fast（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| bytedance-seedance-1.5-pro（圖生影片） | Seedance 1.5 Pro | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0（圖生影片） | Seedance 2.0（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0-fast（圖生影片） | Seedance 2.0 Fast（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| bytedance-seedance-1.5-pro（參考生影片） | Seedance 1.5 Pro | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0（參考生影片） | Seedance 2.0（即夢） | ByteDance Seedance | 可開關 ⚠️ |
| dreamina-seedance-2.0-fast（參考生影片） | Seedance 2.0 Fast（即夢） | ByteDance Seedance | 可開關 ⚠️ |

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
> **Seedance 的配音開關先前被鎖死在關閉**（2026-08-12 修正）。所有 Seedance 型號原本標 `audio: False`，前端不但隱藏開關，還會**強制把勾選狀態設成 `false`**（`onVidModelChange`），於是每一次請求都明確送出 `metadata.generate_audio=false`。而上游這個欄位的預設值是 `true`，等於平台主動把本來會有的聲音關掉，且使用者無從開啟。送出的管線其實早就接好了——`_apply_audio_flag()` 從一開始就把 Seedance 專用的 `generate_audio` 帶上，缺的只是 MODELS 的旗標。現已改成 `audio: True`，**預設仍維持不勾選**（與上游預設相反，但與平台其餘型號一致；無聲對 `bytedance-seedance-1.5-pro` 另有 0.5× 折扣，官方定價有聲 $0.0024／無聲 $0.0012 每 1K tokens，2.0 與 2.5 未註冊折扣）。
>
> **Seedance 不接受 `negative_prompt` 與 `prompt_extend`。** 閘道 doubao adaptor 的請求結構裡沒有這兩個欄位，文字內容只用到 `prompt`；`metadata` 是整包 unmarshal 進結構、對不上的鍵直接丟棄，所以這兩個控制項對 Seedance 是純裝飾。已用 `no_negative_prompt`／`no_prompt_extend` 標記，前端選到時整組隱藏並清空值（`memory.md` 第 6 條：UI 不要顯示沒有作用的控制項）。
>
> **驗證狀態（一半已驗、一半未驗，不要混為一談）**：
>
> - ✅ **「關閉」那半已由閘道端實測**（📄 轉述自 `nen-ai-platform` session，2026-08-12，正式站）：送 `generate_audio: false` 參數確實送達（上游任務物件原樣回吐 `false`），產出的 mp4 用 `ffprobe` 檢查只有一條 h264 stream、**沒有 audio track**。
> - ⚠️ **「開啟後真的會出聲」那半仍未驗證。** 影片生成單價高，依使用者指示暫不測試。
> - ⚠️ 「上游預設為 `true`」這點仍是讀閘道 Go 原始碼推斷的，屬於 `memory.md` 4d 的第④類「拿間接證據當行為證據」——本專案在 `wan2.7` 的 `max_ref` 上正是這樣栽過（依 Go struct 寫成 2、實測是 9）。
>
> **計費：只有 `bytedance-seedance-1.5-pro` 有無聲折扣（×0.5）。** Dreamina Seedance 2.0 與 2.5 在閘道的 `silentVideoRatioMap` 裡是**刻意沒列**的（官方定價沒有音訊維度），所以這兩代不論開不開配音，價錢都一樣。
>
> **上游支援但平台未送出的 Seedance 參數**（同樣來自 adaptor 結構，未實測）：`camera_fixed`（2.5 實測回 `InvalidParameter`）、`frames`、`output_format`（mp4/mov，僅 2.5）、`priority`（0–9）、`return_last_frame`、`draft`（僅 1.5-pro）、`omni_reference_task_type`（2.5 的 auto/reference/edit/extend）、`callback_url`／`service_tier`／`execution_expires_after`／`safety_identifier`／`tools`。另有兩個逃生門：`metadata.content` 可直接覆寫整個 content 陣列、`metadata.image_role` 可指定參考圖角色。
>
> **`dreamina-seedance-2.5` 與 2.0 系列差異很大，UI 需分開處理**（2026-08-11 對**測試網關** `192.168.0.245` 驗證——正式環境的模型清單裡雖然有它，但網關程式碼尚未部署、在那邊叫不動）：
>
> | | 2.5 | 2.0 系列 |
> |---|---|---|
> | 解析度 | **只有 480p / 720p** | 480p ～ 4K |
> | 時長 | `[4, 30]` 或 `-1` | 2 ～ 15 |
> | 參考圖 / 影片 / 音訊 | 30 張 / 10 支 / 10 段 | 9 / 3 / 3 |
> | 純音訊輸入 | ✅ 支援 | ❌ |
> | 每秒單價（720p） | **$0.2311** | $0.1512 |
>
> **比 2.0 貴約 53%**。送 `1080p`／`4k` 實測回 `InvalidParameter`，`duration` 送 3 或 31 也被拒，所以 MODELS 以 `resolutions` 與 `min_dur`/`max_dur` 限制住。不支援 `camera_fixed`／`frames`／`draft`（實測都回 `InvalidParameter`）。
>
> ⚠️ 網關端說 `seed` 也不支援，但**實測沒有被拒**（任務照樣建立），與其規格說明不符——目前以實測為準，沒有特別擋。
>
> **目前只上架了文生影片（t2v）。** 圖生／參考生／視頻編輯需要帶入參考影片，而影片輸入必須是雲端網址（見上方），在雲端物件儲存設定好之前做了也不能用；另外 2.5 專屬的 `omni_reference_task_type`（顯式指定任務類型，避免參數不合要等到非同步階段才報錯）在測試網關上也還沒透傳。這兩件完成後再補。
>
> 計費公式：`tokens = 寬 × 高 × 幀數 / 1024`，fps 固定 24。回傳的 `duration` 是無條件捨去，**計費按真實幀數**（要求 4 秒實際收 4.04 秒）。

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
> **這裡曾經出現過一次觀察衝突，結論值得記下來。** 文檔站那邊測到 `dreamina-seedance-2.0-fast` 送 1080P「沒有被拒、任務跑到 SUCCESS」，與本文的說法相反。2026-08-11 用控制變因重測後釐清——**兩邊的觀察都是真的，只是測到了不同的東西**：
>
> | 送法 | 結果 |
> |---|---|
> | 只送頂層 `size=1080P` | 200，任務建立（但輸出仍是 720p） |
> | 只送 `metadata.resolution=1080p` | **400 InvalidParameter** |
> | 兩者都送（本平台的作法） | **400 InvalidParameter** |
>
> 因為 **doubao 根本不讀頂層 `size`**，第一種送法的 1080p 從未送達上游，任務會成功只是因為上游用了預設的 720p。`metadata.resolution` 才是 doubao 唯一會讀的解析度來源，一送 1080p 就被拒。**結論不變：`fast` 版確實不支援 1080p。**
>
> 這也是一個提醒：**比對兩份互相矛盾的實測結果時，要先確認雙方送出的請求是否等價**，否則會把「測法不同」誤判成「事實不同」。
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
| qwen3.5-omni-plus-realtime | Qwen3.5 Omni Plus Realtime | 即時語音 | WebSocket 雙向串流，可聽可說、支援語意斷句與插話 |
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
| 語音辨識 ASR | `audio` → `text`。可接語音 TTS 節點的輸出，或直接上傳音檔。補上這個節點之前 `audio` 型別在整張畫布上**只有產出端、沒有消費端**；有了它才組得出「文字 → 配音 → 轉回文字」這類閉環，也才能把上傳的錄音接進後面的文字節點。只提供非串流型號——串流版走另一條端點，而在節點圖裡要的是最終完整逐字稿，中間結果沒有用處 |
| MuleAI Spicy | 對應 NenAI Spicy 四個模型，依選擇的模型動態切換必填輸入與輸出型別（image/video） |

節點之間可用連線傳遞文字/圖片/影片/音訊輸出，設定面板僅在節點被選取時以固定大小浮層顯示於節點下方（沒有媒體預覽的節點——文字、語音辨識——則把控制項直接留在節點本體，不走浮層）。

**進階參數（Negative Prompt / Seed / 生成張數）**：圖片、影片、影片編輯、圖像編輯、MuleAI 五種節點共用同一套實作。顯示與否依端點實際會讀的欄位決定，不放沒有作用的控制項——生成張數只在文生圖出現（`/api/image/edit` 沒有 `n` 這個欄位，組圖模式則自己帶 `seq_n`），Negative Prompt 在參考生影片模式下隱藏（`/api/video/r2v` 不讀這個欄位）。Seed 留空存成 `null` 而不是 `0`（`0` 是合法 seed，兩者不能混）。

**一鍵執行整張圖（工具列的 ▶）**：依拓樸順序自動跑完畫布上所有可執行節點。三個刻意的行為——(1) **不執行文字節點**，它的「生成文字」是 opt-in 的，自動呼叫會把使用者寫好的 prompt 換成 LLM 生成的內容；(2) **已有結果的節點略過**，要重跑單一節點請按它自己的按鈕；(3) **上游失敗就跳過下游**，不浪費一次註定失敗的呼叫。送出前會確認節點數與其中有幾個是影片節點（非同步、送出後無法取消）。

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
