# memory.md — 三個專案的關係與跨專案協作備忘

這份文件記錄「測試平台（本專案）／閘道／文檔站」三個 repo 之間的關係、各自的職責邊界，以及在這條鏈上反覆踩過的坑。**目的是讓新接手的人不必重新踩一次**——底下每一條「教訓」都是實際造成過錯誤判斷或錯誤設定的。

日常開發規範看 `CLAUDE.md`，逐次變更紀錄看 `update.md`，模型清單與參數細節看 `README.md`。

---

## 一、三個專案

### 1. 測試平台（本專案）

- 路徑：`/Users/levi/program_lab/AI_lab/alibaba/alibaba-model-nenAI`
- 遠端：`github.com/levilan/alibaba-model-demo`，開發分支 `nenai`
- 技術：FastAPI（單檔 `app.py`）＋ 原生 JS，無資料庫、無 build
- 正式站：**https://playground.nen.com.tw**
- 角色：**這條鏈的最末端**。只負責「把參數用上游看得懂的格式送出去、把結果呈現給人看」，不做模型邏輯、不做計費。

### 2. 閘道 nen-ai-platform

- 路徑：`/Users/levi/claude_code/nen_ai_project/nen-ai-platform`
- 遠端：`github.com/HigherCloudMaster/highercloud-ai-platform`，工作分支 `feat/carrothub-channel-support`
- 技術：Go，「New API」那一系的多渠道 AI 閘道
- 正式站：**https://nen.com.tw**（`NENAI_BASE` 預設值）；測試網關 `http://192.168.0.245`
- 角色：**我們與各家原廠之間唯一的中介**。負責渠道路由、請求格式轉換、輪詢非同步任務、計費。

我們送出的任何參數，能不能生效**完全取決於這一層有沒有映射它**。這是本專案絕大多數「參數送了卻沒作用」問題的根源。

### 3. 文檔站 Nen-AI-Docs-V1

- 路徑：`/Users/levi/claude_code/nen_ai_project/Nen-AI-Docs-V1`
- 遠端：`github.com/HigherCloudMaster/Nen-AI-Docs-V1`，分支 `main`
- 角色：對外的 API 文檔站。

### 資料流

```
瀏覽器 → 測試平台 (Cloud Run) → nen 閘道 → 各家原廠 (DashScope / Google / OpenAI / 智譜 / 字節…)
                                    ↓
                              文檔站（描述上面這一切）
```

---

## 二、職責邊界（不要越界）

- **本專案不改閘道的程式碼。** 發現閘道有問題，是回報給 `nen-ai-platform` 的 session，由他們判斷與修改。
- **本專案不改文檔站的檔案。** 新增模型後把資訊送過去，由文檔站的 session 撰寫。
- 三個 repo 各有自己的 Claude session。跨專案溝通用 `ListAgents` 找對象、`SendMessage` 送訊息，**不要直接跑到別人的目錄改檔案**。

`CLAUDE.md` 已把「新增模型後通知文檔 session」定為必做流程；找不到 session 時要**明確提醒使用者啟動**，不要默默略過。

---

## 三、基礎架構（2026-08-11 實查）

GCP 專案 `ai-model-hub-newapi`，**測試平台與閘道都在 `us-east5`**。

| | 測試平台 | 閘道 |
|---|---|---|
| LB IP | `34.128.190.102` | `34.54.144.102` |
| 網路層級 | `PREMIUM`（全球 anycast） | `PREMIUM` |
| 運算 | Cloud Run `nenai-testing-platform`（us-east5，`minScale` 未設＝會冷啟動） | VM instance group（us-east5-c） |
| LB 後端逾時 | 30s（**對 serverless NEG 不生效**，實際受 Cloud Run `timeoutSeconds: 300` 管轄） | 1800s |

**效能結論：平台幾乎不增加延遲。** 同一個 2048×2048 請求，透過平台 44.8s、直接打閘道 44.0s，**只差 0.8 秒**（兩者同區域）。若有人反映「透過平台比較慢」，先確認是不是在比較不同的工作負載——實測 i2i 約 11s、t2i 約 24s，**編輯本來就比文生圖快一倍以上**；而且同一請求的變異可達 40%（26.8s vs 38.5s），單次測量不可靠。

---

## 四、反覆踩過的坑（最重要的部分）

### 1. 「通過驗證」不等於「能用」

閘道的參數驗證比原廠寬鬆，會放行原廠實際不支援的組合：

- `wan2.7-image` 送 `4096*4096`、pro 開組圖送 `4K`，**閘道都放行**，但官方文件明說不支援（閘道只檢查總像素落在最寬鬆的那組範圍）。
- `dreamina-seedance-2.0` @1080P 提交回 200，任務跑 15 分鐘後 `failed`。

**所以「打閘道試得通」不能當作支援的證據。** 尺寸／能力清單以官方文件為準，能端到端產出成品再量測才算驗證。

### 2. 不要從閘道的 Go struct 推斷行為

`WanImageInput.images` 宣告上限 2 張，我據此把萬相編輯的參考圖上限設成 2——**實測是 9 張都生效**（那條約束不適用於 `/v1/images/edits` 這條路徑）。這個錯誤限制正好對應使用者反映的「wan2.7 只能傳兩張，阿里可以傳九張」。

Go struct 描述的是某一條路徑的資料結構，不等於所有路徑的行為。**要驗就實際打。**

### 3. 探測手法不能跨模型家族沿用

「送超範圍的 `n` 當哨兵、靠錯誤訊息判斷 size 是否合法」在萬相家族有效（size 先驗），但**千問 3.0 的 `n` 先驗**，導致連 `10*10` 都回「n 的錯誤」、看起來像尺寸通過了。差點據此把一堆沒驗證過的尺寸寫進清單。

而且這招對 MAI／Seedream **會真的產圖**（它們不拒絕 `n=13`），曾因此一次意外生成上百張圖。用這招前**務必先確認該模型真的會拒絕哨兵參數**，而且保護機制不能在重寫腳本時被拿掉。

### 4. 官方文件會過時、會寫錯

- 千問 3.0 的 `usage` 官方文件寫 `{width, height, image_count}`，**實際是** `output_width` / `output_height` / `output_image_count` / `output_image_type`。
- 千問 3.0 文件說「目前可用的是 pro」，但 base 版實際可用。
- `wan3.0-video` 的 API 文檔至今是邀請制未公開。

### 4b. 分清楚「觀察到的數據」與「從數據推出的因果」

標明「實測」還不夠——實測的是**數據**，從數據推出的**因果**仍然是推論，要分開講。

實例：我看到 `qwen-image-3.0` `n=2` 回 `usage.output_image_count: 2` 而 `data[]` 只有 1 筆（數據，正確），就宣稱「使用者付了 2 張的錢卻只拿到 1 張」（因果，**錯的**）。閘道實際是**數 `data[]` 的圖片數**計費，使用者只被收 1 張——真正吃虧的是平台方。這個錯誤結論已經流進文檔站才被閘道端查用量日誌打回。

跨專案交接時，建議分三層寫：**我實測到的數據** / **我從中推出的結論** / **我轉述自他人的**。

### 4c. 要知道「實際收多少錢」，用用量增幅量，不要推

`GET https://nen.com.tw/v1/dashboard/billing/usage`（帶一般的 `Authorization: Bearer`）回傳這把 key 的累計用量：

```json
{"object":"list","total_usage":155615.3236}
```

做法是**前後各查一次、比對增幅**：先取基準，送出要測的請求，等一下再取一次。平台開了 `BATCH_UPDATE_ENABLED`，用量不是即時刷寫的，**要等約 12 秒**才反映得出來。

這是唯一能直接回答「這個請求實際被收多少」的方法。`usage.output_image_count`、`model_price` 這些都只是線索，不是計費依據——上面 4b 那個錯誤就是把線索當結論。

實例：文檔站用這個方法驗證 `qwen-image-3.0` 的 `n=1` 與 `n=2` 增幅都是 `+3.0`（比值 1.0），獨立確認了「計費依 `data[]` 實際張數」。

### 5. 「參數送了沒作用」幾乎都是閘道沒映射

實例：`glm-5.2` 走 `/v1/messages` 時整個 `thinking` 物件無效——根因是 `service/convert.go` 的 `ClaudeToOpenAIRequest` **只在 OpenRouter 分支讀 `Thinking`**，其餘渠道一律丟棄；`reasoning_effort` 更是連 `dto.ClaudeRequest` 都沒宣告，反序列化階段就消失。

判斷方法：**找一個同模型、不同路徑的對照組**。同一個 `glm-5.2` 在 `/v1/chat/completions` 上 `enable_thinking:false` 能把 token 從 159 降到 19——證明模型做得到，問題在路徑。

### 6. UI 不要顯示沒有作用的控制項

這是本專案的一貫原則，`MODELS` 裡不少旗標就是為此存在：

- `thinking`：只有真的能開關思考的模型才給開關
- `no_size`：圖像編輯情境下 `size` 不生效的模型（千問 3.0 已標；**千問 2.0 與萬相編輯也一樣不生效，但尚未處理**）
- `i2v_modes`、`ref_images_only`、`max_ref`、`resolutions`、`reasoning_efforts`：都是「上游只吃這些」的白名單

---

## 五、跨 session 協作紀錄（進行中的事項）

- **閘道端已修但尚未部署**：`glm-5.2` 的 `thinking` 映射與非串流思考回傳（推到 `feat/carrothub-channel-support`，`main` 未合）。切回 `/v1/messages` 需要：程式碼部署 **＋** 渠道後台開啟「thinking 映射為 enable_thinking」開關（預設關）。目前刻意不切——現在走的 `/v1/chat/completions` 三項能力（開關思考、非串流看得到思考、7 段 `reasoning_effort`）都有，切回去反而少一項。
- **`wan3.0-video`**：已上架但**從未成功呼叫過**，閘道端測試一直是 `AccessDenied`（模型層級未開通）。`input.media` 的 type 詞彙仍未驗證。
- **`glm-5.2-us` / `glm-5.2-fast-preview`**：閘道上沒有，但上架**不需要改程式碼**，後台把模型名加進渠道即可。
- **realtime（即時語音）整條路對阿里模型不通**：`qwen3.5-omni-*-realtime` 在 `/v1/models` 清單裡，但 realtime 中繼只有 OpenAI 系 adaptor 有實作，阿里 adaptor 沒有 realtime 分支。要能用需要在閘道端補一整套（撥上游 WS、DashScope 與 OpenAI realtime 事件格式互轉、usage 計費），屬獨立功能開發，已轉給閘道端的使用者決定是否排程。我們的 `/ws/omni` 代理保留著、路徑已修正，等上游補上就能直接用。
- **`/v1/models` 清單有某個模型 ≠ 那條通道可用。** realtime 這次是最清楚的例子：模型在清單裡，但走 WebSocket 會讓閘道 panic。清單只代表「這個 key 的群組被允許使用這個模型名」，不保證任何特定端點或協定能用。新增模型時**先驗端點再做 UI**，不要反過來。
