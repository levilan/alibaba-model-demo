# NenAI Playground — MCP Tool 設計

> 狀態：設計稿（2026-08-25），尚未實作。
> 前提結論（與 Levi 討論定案）：以 **remote MCP 為主要形式**，skill 之後補「選型智慧」層；
> 文字對話**不做成 tool**——閘道本來就是 OpenAI 相容 API，客戶把 `base_url` 指過來即可，
> MCP 只包「OpenAI 相容協定給不了的東西」：task 制影音、multipart 編輯、各家參數怪癖。

---

## 一、設計原則

1. **動作型工具，不是模型型工具。** 158 個模型絕不做成 158 個 tool——8 個動作工具，
   `model` 當參數。tool schema 是客戶 context 的常駐成本，每一個欄位都要付 token。
2. **靜態 schema 管形狀，動態約束靠 discovery＋伺服器端驗證。** 各模型的尺寸枚舉、
   時長範圍、參考圖上限彼此不同，塞不進一份靜態 schema。解法分三層：
   - `nenai_list_models` 回傳每個模型的完整約束（資料源就是 `MODELS`，單一真實來源）；
   - tool description 寫「先查 list_models 取得該模型合法值」；
   - 伺服器端驗證失敗時，**錯誤訊息必附合法值清單**（沿用平台 422 白名單的做法），
     讓 agent 一次修正、不用試錯輪迴。
3. **沿用 playground 的轉譯層，不重寫。** MCP server 掛在同一個 FastAPI app 上
   （`/mcp` 路徑），工具實作直接呼叫既有的端點函式——上游怪癖已經在那裡被吸收過一輪，
   再寫一份必然漂移。
4. **遵守「客戶內容不上雲」政策。** 生成結果回傳上游 URL（註明時效約 24h）或本站
   暫存 URL，不落地長期儲存；輸入素材走「處理中暫存、任務完即刪」。
5. **計價透明。** 每個生成工具的結果都帶 `pricing` 欄位（單價與計價方式，取自
   `/api/pricing`），讓 agent 能在行動前回答使用者「這要花多少錢」。

## 二、架構

```
AI 客戶端（Claude Code / Desktop / Cursor / 任意 MCP host）
   │  Streamable HTTP + Authorization: Bearer sk-...
   ▼
playground（既有 Cloud Run 服務）
   ├── /api/*   既有 REST（前端用，不動）
   └── /mcp     MCP endpoint（新增；工具實作內部呼叫同一批轉譯函式）
        │
        ▼
   NenAI 閘道（nen.com.tw）
```

- **驗證**：沿用 `get_api_key()`——MCP 請求帶 `Authorization: Bearer sk-...`，
  原樣轉發上游，不驗證不保存，與現行設計一致。統計 middleware 自動涵蓋（uid 雜湊照舊）。
- **零安裝**：客戶在 MCP 設定貼一個 URL＋自己的 key 即可，版本永遠最新。
  同一套程式碼也可日後包 stdio 版（本地檔案輸入方便），非第一階段目標。
- **Spicy 模型獨立開關**：`nenai_spicy_generate` 預設**不在** tools/list 裡，
  需在連線 URL 帶 `?spicy=1` 才註冊——成人內容工具不該出現在預設工具面上。

## 三、工具清單（8＋1 個）

### 1. `nenai_list_models` — 探索與約束查詢（discovery，最重要的工具）

沒有這個工具，其他工具的動態約束都站不住。

```
輸入：
  category?: "image" | "video" | "voice" | "music"   # 省略 = 全部
  model?: string        # 指定單一模型時回完整約束
輸出（每模型）：
  id, name, desc, capabilities(t2i/i2i/t2v/i2v/r2v/vedit/animate/tts/asr)
  constraints: { sizes[], resolutions[], duration:{min,max,step}, max_n,
                 max_ref_images, voices[], ... }   # 直接來自 MODELS，欄位有就給
  pricing: { unit: "per_call" | "per_second" | "per_1k_tokens", price, currency }
```

### 2. `nenai_generate_image` — 文生圖（同步）

```
輸入：
  model: string          # 必填；合法值見 list_models(category="image")
  prompt: string
  size?: string          # 模型各自的枚舉，錯了回 422＋合法清單
  n?: int (1..模型上限)
  negative_prompt?, seed?: int
輸出：
  images: [{ url, expires_hint }], seed_used?, pricing
```

### 3. `nenai_edit_image` — 圖像編輯／多圖融合（同步）

```
輸入：
  model, prompt
  images: [string]       # URL 或 data URI（base64 上限 8MB/張；上限張數依模型）
  size?, negative_prompt?, seed?
輸出：同 generate_image
```

> 輸入素材處理：URL 由伺服器抓取後走既有 multipart 轉譯；data URI 直接解碼。
> 素材為「處理中暫存」，任務結束即刪。

### 4. `nenai_generate_video` — 文生／圖生影片（**非同步**，回 task_id）

```
輸入：
  model, prompt
  image?: string         # 給了就走 i2v（首幀），沒給走 t2v
  resolution?, duration?: int, audio?: bool, negative_prompt?, seed?
輸出：
  { task_id, status: "pending", estimated_seconds, poll_hint, pricing_estimate }
```

### 5. `nenai_video_advanced` — 參考生／影片編輯／動作動畫（非同步）

r2v／vedit／animate 三種模式的輸入形狀差異大，但都屬「進階影片」低頻操作，
合併一個工具用 `mode` 區分，換取工具面精簡：

```
輸入：
  mode: "reference" | "edit" | "animate"
  model, prompt?
  ref_images?: [string]      # reference/edit 用；上限依模型（萬相 9、happyhorse 9…）
  video?: string             # edit/animate 的來源影片（URL）
  person_image?: string      # animate 用
  resolution?, duration?
輸出：同 generate_video
```

### 6. `nenai_task_status` — 任務查詢（配合 4、5）

```
輸入： task_id: string
輸出： { status: "pending"|"running"|"succeeded"|"failed",
        video_url?, expires_hint?, error?, elapsed_seconds }
```

> **為什麼不做成「阻塞到完成」**：影片任務 1～10 分鐘，超過多數 MCP client 的
> 工具逾時；submit＋status 讓 agent 自主決定輪詢節奏，也能在等待時做別的事。
> description 裡建議輪詢間隔（首次 30s、之後 15s），別讓 agent 一秒打一次。

### 7. `nenai_tts` — 語音合成（同步）

```
輸入： model, text, voice?（合法音色見 list_models）, format?: mp3|wav|opus|flac
輸出： { audio_url, duration_seconds, pricing }
```

### 8. `nenai_asr` — 語音辨識（同步）

```
輸入： model, audio: string（URL 或 data URI）
輸出： { text, pricing }
```

### 9. `nenai_spicy_generate`（預設隱藏，`?spicy=1` 才註冊）

muleai 四模型（圖／圖編／換臉／影片）合一，`model` 區分；非同步，共用 `nenai_task_status`。

> 音樂（lyria）第一階段併入 `nenai_tts`？——**不併**。計價與輸出性質不同，
> 但使用頻率低，第一階段可直接**不收**，等有需求再加 `nenai_generate_music`。

## 四、橫切設計

**錯誤格式**（所有工具一致）：
```json
{ "error": "invalid_parameter", "field": "resolution",
  "message": "veo-3.1-generate-001 只接受 720P/1080P",
  "valid_values": ["720P", "1080P"] }
```
配額未開（上游 AccessDenied）明確區分於參數錯誤——這是 probe 腳本踩過的教訓：
分不清會讓 agent 對同一個錯無限重試。

**結果時效**：所有媒體 URL 附 `expires_hint`，並在 description 明講「請立即下載，
連結約 24 小時失效」——配合不保留客戶資料的政策，管理客戶預期。

**統計**：MCP 呼叫走同一個 middleware，`request.state.model` 照設，報表自動涵蓋；
另在統計 endpoint 欄位可辨識（路徑前綴 `/mcp`），未來可分析「MCP vs 網頁」的使用佔比。

**限流**：沿用上游閘道的配額；playground 端不另做（跟現行 REST 一致）。

## 五、工具面 token 成本估算

8 個工具、每個 schema 平均 ~150 tokens ≈ **1,200 tokens 常駐**。可接受。
反例：若按模型拆工具（158 個）會超過 20k tokens——這就是「動作型不是模型型」的量化理由。

## 六、AI Canvas 整合（三層，取所需）

先講清楚一個事實：**Canvas 的執行引擎在瀏覽器裡**（canvas.js 的 litegraph 迴圈逐節點
呼叫 `/api/*`），後端沒有任何「跑一張圖」的能力。所以「把 Canvas 做進 MCP」不是搬現成
的東西，要拆成三層看：

### L0：什麼都不用做（agent 本來就是 Canvas）

Canvas 存在的理由是**人類需要視覺化拉線**；agent 串工具呼叫是原生能力。
「文字 → 生圖 → 拿去生影片」這種 Canvas 工作流，agent 用第三節的 8 個工具
鏈式呼叫就完成了——**Canvas 的 runtime 對 agent 的價值是零**，不需要為此建任何東西。

### L1：圖的互通——「agent 畫圖給人看／人畫圖給 agent 跑」（建議做）

Canvas 真正獨有的價值是**那張圖本身**：可視、可調、可存。互通做法：

- 定義一個**簡化 workflow DSL**（JSON：節點=動作型工具的呼叫、邊=資料流），
  伺服器負責與 litegraph 序列化格式雙向轉換。**不要讓 agent 直接讀寫 litegraph
  原生格式**——那是 UI 內部序列化（座標、插槽索引、refSlots 重建邏輯），對 agent
  極不友善且會跟著 UI 版本漂移；DSL 是穩定的中間層。
- 新工具 `nenai_canvas_compose(workflow_dsl)` → 回傳可匯入 Canvas 的檔案（走既有
  「匯入」功能）：**agent 幫使用者搭好一張圖，人再視覺化微調**——這是「agent 起稿、
  人類精修」的工作流，也是 Canvas 對 MCP 客戶最有感的賣點。
- 反向 `nenai_canvas_parse(exported_json)` → 回傳 DSL：使用者把自己拉好的圖匯出丟給
  agent，agent 讀懂後可以逐節點代跑（用既有 8 工具）、或解釋與除錯這張圖。

儲存政策：**圖不落地**。graph JSON 走工具參數與回傳值傳遞、由使用者自己保管
（跟 Canvas 現行的 localStorage／匯出檔案一致）——節點裡的 prompt 是客戶內容，
伺服器端不建 canvas 儲存，正好避開「客戶內容不上雲」的政策衝突。

### L2：伺服器端 workflow runner（大建設，需求驗證後再做）

`nenai_run_workflow(workflow_dsl, inputs)` ——後端直接執行整張圖（DAG 執行器＋
中間產物暫存＋進度回報），一次呼叫跑完多步驟管線。這等於把 Canvas 從「視覺測試台」
升級成「工作流平台」，是殺手級功能但工程量大；而且 L0 已讓 agent 能逐步執行，
L2 的增量價值是「原子性＋不佔 agent 回合」。**等 L1 驗證出有人在用 DSL 再投資。**
屆時 L1 的 DSL 直接就是 L2 的執行格式——這是先做 L1 的另一個理由。

## 七、新模型上線時 MCP 怎麼同步

**設計目標：日常新模型 = 零 MCP 手工。** 機制靠三件事，全都掛在「新增模型本來就要做的事」上：

1. **`MODELS` 是唯一真實來源**：`list_models`、伺服器端驗證、（若做動態枚舉）工具
   schema 都在執行期讀 `MODELS`——新模型加進去，MCP 面自動長出來。
2. **部署即生效**：加模型本來就要 push 部署 playground；MCP 掛同一個 app，
   同一次部署一起更新。remote MCP 客戶端每次連線重新拉 tools/list，不存在「客戶端
   裝了舊版」的問題（這也是選 remote 而非發行安裝包的理由之一）。
3. **轉譯層共用**：新模型家族的上游怪癖（參數放哪層、欄位名）在既有端點函式裡吸收，
   網頁 UI 與 MCP 吃同一份——修一次兩邊都對。

因此**新增模型的標準工作流只多一步**（併入 CLAUDE.md 六步流程的第 4、5 步之間）：

> 部署後用 MCP 端冒煙：`list_models` 看得到新模型且約束正確 → 打一次最小生成。
> 約兩三分鐘。

**前提紀律（讓上面成立的代價）**：驗證出的約束必須寫成 `MODELS` 的**結構化欄位**
（sizes/resolutions/min_dur/max_dur/max_ref/max_n…），不能只寫在 `desc` 文案或程式
分支裡——MCP 上線後這些欄位從「UI 提示」升格為「機器契約」，寫漏 = agent 端驗證
放行了會被上游拒絕的參數。

**仍需人工動 MCP 的三種情況**（低頻，模態級變化）：

| 情況 | 例子 | 動什麼 |
|---|---|---|
| 新模態 | 音樂、realtime 語音 | 加新工具 |
| 新輸入形狀 | 影片多了「驅動音訊」這類輸入 | 既有工具 schema 加欄位 |
| 新計價型態 | 按解析度分級之外的新算法 | pricing 欄位擴充 |

**守門機制**：在 `tests/test_pure_functions.py` 加一條「欄位白名單」測試——
`MODELS` 出現 MCP 映射表不認識的新欄位名就 fail，強迫在加模型當下決定
「這個新約束要不要進 MCP schema」，而不是等 agent 端出錯才發現。

## 八、實作順序建議（未排程）

1. `list_models` + `generate_image` + `task_status` + `generate_video`（最小可用面）
2. `edit_image` + `tts` + `asr`
3. `video_advanced` + spicy 開關
4. Canvas L1（workflow DSL＋`canvas_compose`／`canvas_parse`）
5. skill（選型智慧層）引用 list_models 的即時資料
6. Canvas L2（伺服器端 runner）——僅在 L1 有實際使用後投資

技術選型：FastMCP（Python）掛進現有 FastAPI app——單一部署、共用轉譯層與統計。
