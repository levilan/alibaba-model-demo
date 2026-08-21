# update.md

本檔案記錄這個專案的每一次重要更新（新功能、修 bug、重構、設定調整……），依日期分組、由新到舊排列，每筆盡量附上對應的 git commit hash（`git show <hash>` 可查看完整內容）。

依照 `CLAUDE.md` 的規定：之後每完成一次變更、準備 commit 時，都要在最上方新增一筆條目，不要事後補記。

---

## 2026-08-21

- fix：`scripts/update_announcements.py` 憑證路徑修正——寫死的 `NEN_ENV_PATH` 還指向搬家前的 `/Users/levi/claude_code/nen_ai_project/.env`（已不存在），改成實際位置 `/Users/levi/nen_ai_project/.env`。
- feat：新增文字模型 `dola-seed-2.1-turbo`（Seed 2.1 Turbo，正式環境 `nen.com.tw` 實測）。按 token 計價（輸入 ratio 0.25、completion_ratio 5、cache_ratio 0.2）。實測重點：
  - `enable_thinking` 開關**無效**（true/false 各 3 次全部回 `reasoning_content`），跟 seed-2.0 系列同樣關不掉，`thinking` 維持 False 不顯示開關。
  - 但它支援 `reasoning_effort`，七個枚舉（none/minimal/low/medium/high/xhigh/max）閘道全收，且 **`none` 真的能完全關閉思考**（3 次 reasoning 全 0）——是 seed 家族唯一能控制思考的型號。各檔實測（同一數學題各 3 次、reasoning 字元數中位數）：none/minimal 0、low 167、medium 128、high 349、xhigh 202、max 230，實際分三段：關（none/minimal）、輕（low/medium）、深（high/xhigh/max）。比照 GLM 的做法給 `reasoning_efforts` 選單。
  - 與後端固定會送的 `enable_thinking:false` 並存實測不衝突（`none` 照樣關、`high` 照樣思考）。
  - 圖片輸入（OpenAI `image_url` + data URI，跟 app.py 現行形狀相同）實測可用，正確認出測試圖顏色，標 `vision: True`。
  - Streaming 帶 `stream_options.include_usage` 實測會回 usage，前端「本次花費」計算正常。
  - 疑似閘道缺口（已另行回報）：上游 Ark 的 `thinking.type` 可關思考，但閘道的 `enable_thinking` 對此模型沒有映射效果；對照組是 `reasoning_effort:none` 能關，證明模型本身做得到。
- policy/feat：**客戶內容不上雲**（使用者裁示「使用者生成內容先不做儲存，保持不儲存客戶資料」）。新增 `_output_put()` 包裝：生成的圖片/影片/音訊與上傳的參考素材預設不再走 `_cloud_put` 上傳雲端，退回本機 `outputs/` 暫存（Cloud Run 實例回收即消失——這是政策要的效果）；雲端儲存憑證只供統計 jsonl 使用。設 `STORE_OUTPUTS=true` 可恢復上雲行為。共改 8 個呼叫點（`images/`、`videos/`、`audio/`、`uploads/` 前綴），`stats/` 不受影響。已確認 bucket 裡沒有任何已上傳的客戶內容需要清除。README「雲端物件儲存」一節同步改寫。（commit 待補）

- infra：**修正正式環境統計寫入——Cloud Run 服務上原本一個環境變數都沒有**，統計 middleware 一直走降級路徑寫進容器暫存磁碟（`min-instances=0`，實例回收即消失），GCS 三個既有 bucket 的 `stats/` 前綴下均為零筆，等於 8/20 上線的統計功能在正式環境從未留下資料。修正內容（經使用者確認後執行）：
  - 新建 bucket `gs://nenai-playground-prod`（us-east5，uniform bucket-level access + public access prevention）。
  - 新建專用服務帳戶 `nenai-playground-storage@ai-model-hub-newapi.iam.gserviceaccount.com`，僅授予該 bucket 的 `roles/storage.objectAdmin`；金鑰檔放在 repo 外（`/Users/levi/nen_ai_project/nenai-playground-storage-key.json`，chmod 600）。原想走 README 的方式二（`GCS_USE_ADC` + SignBlob），但部署用的 levi-601（editor）沒有 `iam.serviceAccounts.setIamPolicy`，無法設自我模擬綁定，改走方式一（金鑰直接簽章）。
  - Cloud Run 設定 `GCS_BUCKET_NAME` / `GCS_CREDENTIALS_JSON` / `STORAGE_BACKEND=gcs`（revision `nenai-testing-platform-00125-65j`）。⚠️ 這同時讓 `outputs/`（圖片／影片產出）改上傳 GCS 簽名網址——這本來就是 `_cloud_put` 的設計，且 `storage.googleapis.com` 已在 proxy 白名單內。
  - 部署後實測：透過 `playground.nen.com.tw` 打請求（`*.run.app` 因 ingress 限制回 GFE 404，不會進到 app），等 60 秒 flush 窗口後 bucket 出現 `stats/` jsonl，`scripts/usage_stats.py` 成功產出報表。首批資料就抓到外部掃描器在探測 `/api/config`、`/api/aws` 等不存在端點（全 404、anonymous）。
  - 順手修好本機開發環境：venv 是專案還在舊路徑 `claude_code/` 時建的，搬家後所有 script shebang 失效——就地改寫 `venv/bin/` 路徑、補回執行權限、`ensurepip` 裝回 pip；另裝了 gcloud CLI（brew）並以 levi-601 金鑰啟用。

## 2026-08-20

- feat：**使用者統計**（使用者要求「統計網站有哪些使用者連上來使用過」）。每次 `/api/*` 與 `/login` 記一行 `{ts, uid, endpoint, ok, status, ms}`，滿 50 筆或超過 60 秒寫成 jsonl 進雲端物件儲存（沿用 `_cloud_put`，無憑證時降級寫本機 `outputs/stats/`）。報表用 `scripts/usage_stats.py` 產生本機 HTML。
  - **三個安全設計**（回應使用者的兩個顧慮：不能給使用者看到、金鑰不能外洩）：①`uid = SHA256(key + STATS_SALT)` 前 16 碼，**明文金鑰不落地**、不可反推；②**不記 prompt／生成結果／IP**——prompt 是客戶的商業內容，記了就變成我們要保管的東西；③**平台沒有任何查詢入口**，沒有網址、沒有管理密碼、沒有可被猜到的路徑，報表是本機檔案。「後台外洩」在設計上不存在，而不是靠密碼防。使用者裁示：空 salt 可接受、失敗也記錄。
  - **middleware 實作，端點程式碼一行都不用改**，之後新增端點自動納入。⚠️ 但**不記 model**：要記得從 request body 拿，而 `BaseHTTPMiddleware` 讀了 body 端點就讀不到；ContextVar 也不可靠（middleware 與端點在不同 task，set 的值傳不回來）。對「知道有哪些人在用」這個目標 endpoint 已足夠。
  - **實測抓到一個真 bug：同一實例在同一秒內 flush 兩次會撞檔名、後者蓋掉前者。** 檔名原本用 `int(time.time())`（秒級），灌 66 筆只留下 13 筆才發現——前 50 筆觸發自動 flush、剩下 16 筆在關閉時再 flush，兩次同一秒。我防了「多實例互相覆蓋」（`_INSTANCE_ID`）卻沒防「同實例覆蓋自己」，已加隨機碼。**這類降級路徑要用『資料量對不對』驗證，不能只看『有沒有產生檔案』。**
  - 統計失敗一律吞掉不影響請求（middleware 整段包 try、flush 失敗只印一行日誌）。`outputs/stats/` 已加進 `.gitignore`。

- feat：**查看實際請求**（使用者選定的新功能方向）。圖片（t2i／i2i）、影片五條路徑、音樂生成完成後，結果區可展開看到**實際送給網關的請求**：method／endpoint／body（或 form 欄位），並可一鍵複製 cURL。解決的是「在 playground 試好參數之後要自己接 API，得自己猜格式」——這也是文檔站每次寫新模型都要我方手動整理呼叫方式的根源。
  - **安全設計：永遠不顯示金鑰。** 後端 `_debug_req()` 回傳的 `auth` 欄位固定是 `Bearer $NENAI_API_KEY`，複製出來的 cURL 也是。理由：使用者知道自己的 key，平台再顯示一次只會多一條外洩管道（截圖、螢幕分享、錄影、貼給別人問問題）。測試釘住這條（`test_debug_req_never_leaks_api_key`）。
  - base64（上傳的圖／影片／音訊）換成 `<N chars>` 長度摘要——塞爆畫面且無參考價值，但**欄位名與結構保留**，那才是使用者要照抄的部分。同樣有測試釘住。
  - 樣式踩到一個坑：結果區是 grid，面板不設 `grid-column: 1 / -1` 會被擠成一欄、JSON 全被裁掉；另外初版誤用深色系 fallback，本專案是淺色主題（`--bg #F5F6F8`），已改用專案既有變數。Playwright 實測渲染、遮罩與版面三項。
  - **影片分頁補上（使用者回報看不到）**：影片是 task 制，面板改在**送出當下**就掛到任務卡上，不等輪詢完成——影片動輒兩三分鐘，想照抄參數沒理由等它跑完。`savePendingTask` 一併存下摘要，重整頁面恢復任務時也還原得回來。`app.js?v=85`。
  - **補完其餘分頁（`app.js?v=86`）**：文字生成（串流與非串流兩條路都接）、NenAI Spicy、語音 TTS 與 ASR。
    - **文字串流的做法**：SSE 沒有「最後回一包 JSON」的機會，所以請求摘要當成**第一個**事件送出（`data: {"request": ...}`），前端收到先存著、生成結束後掛到訊息尾巴。放最前面而不是最後，是因為使用者可能在生成到一半就想看參數。
    - **task 制的兩處（影片、Spicy）面板在送出當下就掛上**，不等輪詢完成；`savePendingTask` 一併存下摘要，重整頁面恢復任務時也還原。
    - TTS 有兩條上游路徑（OpenAI 相容的 `/v1/audio/speech`、DashScope 風格的 `/v1/services/audio/tts/SpeechSynthesizer`），各自記下自己的端點，後者附註說明為什麼不走前者。
  - **仍未涵蓋：realtime（即時語音）**。那是 WebSocket 事件序列、沒有單一「請求 body」可顯示，要做得另設計（例如顯示 `session.update` 的內容），本輪不做。
  - `app.js?v=84`、`style.css?v=42`，測試 +2（共 37 條全過）。

- fix：**wan3.0 音訊開關補上（擋在閘道部署前的迴歸）**（Levi 裁示預設無聲）。閘道 2026-08-20 的修正會讓 `metadata.audio` 對 wan3.0 生效，而**我方一直在送 `false`**——鏈路是 `MODELS.audio=False` → 前端隱藏開關並強制 `checked=false`（`app.js:1206`）→ 送出 `audio=false` → `_apply_audio_flag` 寫成 `meta["audio"]=False`。閘道以前不讀所以沒事，一開始讀就會把**所有 wan3.0 影片變成無聲**。四筆 `audio` 改成 `True` 讓 UI 有開關；HTML checkbox 無 `checked` 屬性，預設即不勾＝無聲，符合裁示。
  - ⚠️ **閘道部署前這個開關是無效的**（勾了仍有聲，因為上游預設有聲、閘道還沒開始讀）。這違反「不顯示無效開關」的慣例，但必須先改——否則部署當下就是全面無聲。閘道部署後即恢復名實相符。
  - **這也反證了閘道端的推論**：他們原以為「我方送 true 所以有聲」，實際上我方送的是 `false` 卻產出有聲，直接證明他們沒讀該欄位（比「官方預設是 true」的推論更硬）。文檔站那支示範片帶的 `metadata.audio:true` 是他們指定、我原樣轉送的，不走 `_apply_audio_flag`，不在這條鏈路上。
  - **待辦（閘道端指出的設計問題，本次未做）**：`MODELS.audio` 把「模型有沒有音訊能力」與「這次要不要音訊」壓在同一個欄位，wan3.0 送出 false 正是這個混淆的副作用。拆成兩個欄位才是根治，但會影響所有影片模型與前端邏輯，需逐家重驗，另排。

- docs：**`wan3.0-video` 首次成功產出，並查出平台計價設定錯誤**（文檔站 session 委託跑一支效果展示片，使用者裁示同意計費）。t2v 對正式站實測：`{model,prompt,duration:10,size:"720P",metadata:{audio:true}}` → 產出 1280×720／30fps／10.03s／21.7MB，**含 AAC 44.1kHz 立體聲且非靜音**（mean −14.9 dB）；排隊 314s＋生成 456s＝12分50秒（1 樣本）。先前 `memory.md` 記載的「從未成功呼叫過、一直 AccessDenied」已過時，權限現已開通。
  - **⚠️ 我把自己程式正規化後的輸出當成回應原文回報，差點害文檔站改錯六個頁面。** 輪詢程式寫了 `d = rj.get("data", rj)` 相容兩種結構、只印 `d` 的欄位，我據此告訴文檔站「查詢回應是扁平的、沒有 `data` 包一層」並要他們改文檔。對方堅持要 `curl` 直出才肯改——原文確實是 `code`/`message`/`data` 三層，我的結論作廢。**回報「回應長什麼樣」一律貼未加工輸出。**（教訓已寫進 `memory.md`。）
  - 兩個端點的回應形狀確實不同（這條是原文比對過的）：`POST` 回**扁平**、`status` 小寫 `queued`、`progress` 數字 `0`；`GET` 回**三層**、`status` 大寫、`progress` 字串 `"100%"`，另有 `data.data.output.task_status`（上游的 `SUCCEEDED`，與外層平台的 `SUCCESS` 不同字）。`result_url` 簽名約 24 小時過期。
  - **平台計價設定錯誤（已回報並由平台端定位）**：官方是按秒三檔（480P $0.05／720P $0.10／1080P $0.20，國內站 0.3／0.6／1.2 元/秒），平台後台卻把 `0.05` 填進「模型倍率」而非「模型固定價格」欄位。平台端算出 `0.05/2 × 500000 × seconds(10) × resolution-720P(2) = 250000` 與回應中的 `quota` 完全吻合＝**$0.50，官方價 $1.00 的一半**。三檔倍率程式碼本來就在且正確，壞的是後台設定。**第二個後果更嚴重**：`model_price` 未設時 `PriceData.ModelPrice = -1`，使 `AdjustBillingOnComplete` 整段跳過——**完成後依實際秒數的結算永遠不執行**，`duration=0`（保留原長）的 videoedit 會用 1 秒保證金預扣後永不補收。
  - **音訊無差價**：官方 API 參考頁的 `audio` 參數說明寫「开关声音价格相同」，三檔價即含音訊；官方計費規則為「输入不计费，输出按成功生成的视频秒数计费」——後者正是平台那條被跳過的完成後結算該做的事。
  - **我方前端估價是對的**（`_VIDEO_SEC_PRICE` 走自己維護的官方每秒價表，不讀 `/api/pricing` 的 ratio），估 $1.00 而實收 $0.50，方向是**少收**；後台修好前拿前端估算對帳會差一倍，不是前端算錯。修好後我方要再跑一支驗證完整鏈路（預扣→三檔倍率→依實際秒數結算），並可加驗 `duration=0` 路徑。
  - **官方 `input.media` 值域確認（雙方各自獨立抓官方頁面複驗，逐字一致）**：合法值只有 `first_frame`(≤1)／`last_frame`(≤1)／`reference_image`(≤10)／`reference_video`(≤5 段、總時長≤15s)／`reference_audio`(≤5 段、總時長≤15s)／`file`(≤1)／`link`(≤1)，且**參考類與幀類互斥、不可混用**。`video`／`driving_audio`／`first_clip` 三個字串**整份文檔都不存在**——那組詞彙是雙方各自從 wan2.7 推導的，推導錯了。
  - **我方盤點：只有一處非法**。i2v 送 `first_frame`/`last_frame`、r2v 送 `reference_image`/`reference_video` 都合法且不混用；非法的是**影片延伸模式送的 `first_clip`**——wan3.0 值域裡沒有「影片延伸」這個概念，不是換字串就好，是該功能在 wan3.0 上不存在。`video`／`driving_audio` 我方沒送過（那是閘道在「客戶沒送 media 陣列」時自己按副檔名推斷的）。
  - **⚠️ 我方會踩到的實際情境只有一種：t2v 帶背景音樂**。閘道收到 `audio_url` 會追加一筆 `driving_audio`（非法），該段跑在 metadata 覆寫之前；我方對 wan3.0 走覆寫管道的條件是 `media_arr` 非空，所以 i2v/r2v 帶音訊會被覆寫蓋掉（安全），**只有 t2v（media_arr 為空）會原樣送出非法 type**。修法的語意待實測：官方 `reference_audio` 是「參考音訊」，與我方的「驅動音訊／背景音樂」是否同一回事文檔沒寫，**不推測**，等閘道修好後用同段音訊跑對照確認。
  - **`duration: -1`（智能時長）目前對客戶不存在**：官方文檔列為支援參數，但雙邊各斷一處——閘道 `if req.Duration > 0` 把 `-1` 丟掉、我方 UI 的時長選單由 `min_dur`/`max_dur` 決定（wan3.0 是 2~30）根本選不到。**客戶讀官方文檔會以為能用。** 要做得兩邊一起改。
  - `metadata.input.media` 覆寫的兩個隱性契約（閘道端查證後轉知）：Go 的 unmarshal 對已有元素的 slice 是**逐元素合併**，同位置只給 `type` 不給 `url` 會**繼承閘道推斷出來的 url**；陣列較短會截斷、較長會新建。我方目前每筆都寫全 `type`+`url` 等同完全取代，**不要「優化」成只送 type**。
  - **我方修正清單（四處，等閘道四條修正上線後一起改一起驗）**：①拿掉 `first_clip` 模式 ②重寫描述文案（「驅動音訊／影片延伸」是推導來的）③9.9 秒限制改成「輸入＋輸出≤30s」與 `reference_video` 的「≤5 段、≤15s」④t2v 帶 BGM 的 `audio_url` 路徑。閘道四條：`wan3MediaType` 兩個值、`audio_url` 追加的 type、幀類/參考類混用推斷、`-1` 放行。
  - 我方 `MODELS` 的 `audio: False` **不是缺口**：萬相家族一律 False（配音由模型自動決定、UI 不給開關），與這次實測「產出自動含音軌」相符。

## 2026-08-18

- docs：**pro 帶圖補測通過，成果總表送交文檔站**（使用者裁示）。`lyria-3-pro-preview` 帶圖在正式站實測 200／46.5s／MP3 3582453 bytes，**歌詞命中圖中元素**（`in the library of us`、`Dusty covers`、`Light is filtering through the glass`、`Wooden floors begin to creak`，Caption 寫 `a grand, sun-drenched library`），提示詞刻意只寫 `write a song inspired by this image`、零場景描述——**圖片輸入自此確認 lyria-3 兩個都支援**。pro 的段落標記 `[[A0]][[B1]][[C2]]` 比 clip 明顯（clip 單段）。
  - ⚠️ 第一次補測時探測函式只印音訊不印文字，拿不到判斷依據，多花一次呼叫（$0.08）重跑。**探測工具的輸出要涵蓋『判斷結論所需的欄位』，不是只有主產物。**
  - 音色正式站全掃**刻意不做**：文檔站已決定只列音色 id、不宣稱可用性，掃了不會被採用。
  - 送交內容：pro 帶圖實測、Lyria 參數三組最終定案（seed／negative_prompt／sample_count／response_format／多圖）、三模型完整規格表、realtime 狀態、今日全部成果一覽，以及「ffprobe 讀得到 ≠ 瀏覽器放得出來」這條給他們複驗時參考。

- fix：**AI Canvas 影片節點接首/尾幀就報錯**（由另一個 session 在線上版追到並回報）。症狀：`nenai/video` 節點只要 `first_frame` 或 `last_frame` 有接線，按生成就噴「Cannot read properties of undefined (reading 'link')」，請求根本沒送出；純文字生影正常。根因：`VIDEO_EXTEND_ENABLED=false` 時 `clipSlot = -1`（沒有那個輸入孔），而 **LiteGraph 的 `getInputData` 只擋 `slot >= inputs.length`、不擋負數**，`getInputData(-1)` 走到 `this.inputs[-1].link` 直接拋 TypeError；t2v 分支不讀 clip 所以只有 i2v 會炸。修法：新增 `_clipInput()` 一律先判 `clipSlot >= 0`，`_hasClip()` 同樣加判。`canvas.js?v=38`。
  - Playwright 對本機服務實測：`getInputData(-1)` 重現出**與回報者一字不差**的錯誤訊息；修正後 `_clipInput()` 回 `null` 不拋錯；接上 `load_image → first_frame` 後 `_detectMode()` 正確回 `i2v`、讀 clip 回 `null`。
  - 連帶效果：先前「first_frame 接線會爆、改接參考圖 1 就好」的傳聞 workaround 繞的是同一顆，修掉後 `first_last_frame`（首尾同幀無縫循環）模式可正常使用。
  - **線上端到端驗收通過**（由回報端在 `playground.nen.com.tw` 複驗）：版號與 JS 內容都確認為 v=38、節點無任何 runtime 補丁（`getInputData` 是原生原型方法），`load_image → first_frame + last_frame` 接線後按生成一路走完，產出 `vid_20260818_060716_0cab70.mp4`（wan3.0、4 秒、首尾同幀無縫循環），原本必炸的 TypeError 不再出現。

- feat：**Lyria 三個模型上正式環境，部署閘門清空**（使用者指示逐站複驗）。測試網關（`192.168.0.245`）三個全通並確認規格：clip MP3 44.1kHz 立體聲 30.8s／13.7s、002 WAV pcm_s16le **48kHz** 32.8s／30.3s、pro MP3 177.3s／66.9s（`bpm 120 / duration_secs 180 / good_crop` 曲式標記），回應形態照舊（lyria-3 在 `steps[].content[]`、002 在 `outputs[]`）。正式站在同一輪裡**分三段陸續到位**：18:30 UTC 三個全滅（`/v1/models` 130 個無 lyria、直呼 500 `model_price_error`），18:53 只有 clip 通（131 個），19:0x 三個齊全（133 個）並逐一實測成功——002 25.2s WAV 48kHz、pro 66.9s MP3 158.4s。閘門集合（`_DEPLOY_GATED_MODELS`）依 8/17 裁示清空、機制保留給下一批。
  - **`/api/pricing` 有價 ≠ 叫得動**：三個模型在正式站 `/api/pricing` 早就有價（0.04／0.06／0.08、四個 group 全開），relay 前置卻回「倍率或价格未配置」——兩份設定不是同一本帳，之前「pricing 有就差渠道綁定」的推斷不完整。判斷可用性一律以實呼為準。
  - pro 版第一個提示詞（`a calm acoustic guitar melody about a quiet morning`）被上游安全過濾擋成 `content_blocked`，換成 pop 主題即通——無害提示詞也會誤傷，印證當初決定把該錯誤原樣呈現給使用者的做法。**同一句在 clip 上是正常通過的**（同一分鐘、同一網關），所以過濾結果會因型號而異；1 次觀察，不足以當規律。
  - 正式站與測試站的 API key 是分開的兩套（正式那把對 `192.168.0.245` 回 401 `无效的令牌`），共用 env 只有正式站的 `model_apikey`。

- docs：**Lyria 參數缺口驗證（E）與 realtime 正式站複驗（G）**（使用者裁示做 A/E/G；A＝回報閘道 WAV bug）。**三組官方參數全部不可用，原因各不相同**：
  - `lyria-002` 的 `negative_prompt`／`seed`／`sample_count`：**閘道沒映射，靜默丟棄**。同 `seed=12345` 兩次產出 md5 不同（`514da92b…` vs `229d3959…`），再送三個欄位全錯誤型別（`seed:"not-a-number"` 等）仍回 200 照常生成——**錯誤型別毫無反應＝根本沒解析**，不是解析後忽略。對照組：同路徑的 `response_format` **有**透傳（拿得到上游錯誤），證明路徑本身能帶頂層欄位，缺的是 `interactions → :predict` 的欄位映射。已回報閘道 session。
  - Lyria 3 多圖：clip 帶 2 個 image 元素 → 200 正常生成，但**送的是同一張圖兩次，分辨不出第二張有沒有被讀**，維持「一張」的說法不變。手上沒有第二張有語意的圖（`outputs/images` 其餘都是純色測試圖）。
  - Pro 的 `response_format`：**透傳到上游但無可用組合**。7 種形狀全 400 且錯誤來自上游：`{"type":"audio"}` → `bit_rate must be positive`，補上 `bit_rate` → `Audio bit_rate is not supported`，`format`／`encoding`／`audio` 巢狀 → `Unknown parameter`，`{"type":"wav"}` → 上游列出合法值 `number/string/image/text/boolean/integer/video/object/array/audio`（那是 interactions 的**通用輸出 schema**，不是音訊格式選項）。**官方說的「Pro 可用 response_format 取 WAV」複現不出來。**
  - **realtime 四個對正式站複驗全通**（各 1 樣本，文字輸入 → 語音+文字輸出）：事件序四個完全一致且與 OpenAI realtime 相容；音訊輸出一律 PCM16 24kHz、輸入 16kHz。usage 差異明顯——omni 兩個 in 494 tokens、audio-3.0 兩個只有 54。⚠️ `qwen-audio-3.0-realtime-plus` 與 `-flash` 同句同音色回傳的音訊 **bytes 與 RMS 完全相同**（75640／3142）——當時懷疑兩個 id 綁到同一上游模型會造成收錯價，**經 Levi 查證後裁示沒有這個問題、排除結案**；成因未追查（1 樣本，可能是該句下的巧合），文檔站未寫。
  - **後續更正（同日）**：閘道端指出 `seed`／`negative_prompt` 要放在 **`generation_config`** 而不是頂層（頂層欄位不在共用的 `dto.InteractionsRequest` struct 裡，解析時對不進去就丟了——這也解釋了哨兵為何連錯誤型別都不報錯；`response_format` 有反應是因為它**在** struct 裡）。用正確位置重測：**同 seed 兩次 md5 仍不同**（`5aa43351…` vs `927c6d3c…`）。再送 `generation_config.seed:"not-a-number"` 仍回 200——**這個哨兵沒有分辨力**，因為「正式站 image 沒有那段轉換碼」與「有碼但 `json.Unmarshal` 的 error 被忽略」兩個假設都預測 200。已請閘道端確認 image 版本與錯誤處理，並建議他們直接打一次 `:predict` 把「上游到底重不重現」獨立出來。**結論未定，文檔站已告知先不要寫。**
  - **seed 定案（同日第三輪）**：閘道端確認轉換碼在正式站 image 裡（與 adaptor 同一 commit，能生成就代表碼在），單元測試也證明 `seed=42` 會出現在 `instances[0].seed`。我方再跑超範圍哨兵：`seed=-1` 與 `seed=2147483648` **都照常 200 生成**。結論是 **上游寬容**而非沒送到——**閘道有正確送出，但 Lyria 2 上游非法 seed 不報錯、相同 seed 也不產生相同結果**，`seed` 對客戶不具備「重現同一首」的能力（4 次同 seed、2 組不同 seed 全部 md5 不同）。⚠️ 那個哨兵有未驗證前提「上游會拒絕非法 seed」，是我先前只列了兩個假設（版本落後／error 被吞）就開始挑輸入，**漏了「上游寬容」這第三個**——正是 memory 裡「先寫下所有假設再挑能分辨的輸入」那條的反例。
  - `negative_prompt`：**未定**。用「同 prompt 求鼓、B 組加 `negative_prompt: drums, percussion, beat`」對照，低頻(<150Hz) mean 由 −26.2 dB 降到 −27.5 dB，方向符合但**單樣本且沒有自然變異基準**，1.3 dB 很可能只是生成隨機性，不當證據。要下結論需先建變異基準（約 6～8 次呼叫）且仍是主觀判斷，判定不值得——真正能一刀切開的是直接打 `:predict`，那需要 Vertex 憑證（在渠道設定裡，兩邊 session 都看不到）。
  - 閘道端據此修了三處：`json.Unmarshal` 的 error 不再被吞（改回明確錯誤並列出可接受欄位）、`sample_count` 改成明確拒絕並說明按次計費、lyria WAV 加防禦性剝殼（待部署，屆時由我方驗「offset 44 不再是 RIFF」＋瀏覽器實際出聲）。
  - `sample_count` 經閘道端說明是**刻意釘死 1**（按次固定價下放行 N 首等於只收一首的錢），非遺漏；我方不開這個欄位。
  - 探測結果已送文檔站（含各項的驗證環境與樣本數分層）。**未做**：pro 帶圖（不在裁示內）、同題 clip 重跑（文檔站已放棄同題對照）。

- docs：**帶圖路徑補測與官方 schema 查證**（承上，使用者指示補測）。①**clip 帶圖在正式站實測通過**（200／11.6s／MP3 30.720s）——測試設計上刻意讓提示詞不描述場景（全文只有 `write a song inspired by this image`），輸入圖是圖書館走道，回來的歌詞唱出 `Golden light through arched windows`／`Two friends between the endless rows`，證明圖確實被讀進去而不只是「不報錯」。②**002 帶圖被拒**：`convert_request_failed` — `lyria-002 only accepts text input, got input item of type "image"`，2 秒內回、無上游往返，是閘道轉換層擋的；**但回的是 HTTP 500，語意上應為 4xx**（尚未回報閘道）。③查證官方文件，根因比「上游拒絕」更根本：**Lyria 2 走 Vertex 傳統 `predict`（`instances[].prompt/negative_prompt/seed` + `parameters.sample_count`，schema 無影像欄位），Lyria 3 才走 `/v1beta/interactions` 並支援最多 10 張圖**——兩者不是同一套 API。
  - **官方支援但我們沒用到的參數（皆未實測閘道是否透傳，勿當事實引用）**：002 的 `negative_prompt`／`seed`／`sample_count`（seed 與 sample_count 互斥）、lyria-3 的多圖（官方稱上限 10，我們只收 1 張）、pro 的 `response_format:{"type":"audio"}` 取 WAV。
  - **更正：曲式標記（`Mosic` / `BPM`）是 lyria-3 兩個共有，不是 pro 獨有**——由文檔站複驗時指出，先前寫成 pro 特徵是漏看 clip 的 Caption 尾端。BPM 與提示詞情緒吻合：clip「quiet morning」→ 90、pro「upbeat pop」→ 120。
  - `mime_type` 實際值：clip／pro 是 `audio/mpeg`、002 是 `audio/wav`。
  - 文檔站（`Nen-AI-Docs-V1`）已完成 Lyria 頁（繁＋簡，`content/docs/zh-TW/api/gemini/music/`），採用本輪素材與結論；單價、耗時、精確秒數、未驗參數依該站規範不寫入客戶文檔。
  - ⚠️ **測試網關金鑰一度寫在 scratchpad**（`test.key`），而其他 session 讀得到該目錄（文檔站即由此看到音檔）。已刪除並要求對方不得留存；**該把 key 建議更換**。往後臨時金鑰不要落地成檔案。

- docs：**正式環境驗證輪**（使用者反映正式站沒有 Lyria 而觸發）。查證：正式 playground 已跑最新版（v=83）；正式閘道已部署三個 realtime 模型（閘門自動放行、pricing 四檔正確）；**Lyria 在正式站 `/api/pricing` 有、`/v1/models` 沒有**——渠道模型清單／分組綁定未開，已回報閘道 session，綁好會自動出現。realtime 對正式站對帳三輪：`video_tokens` 修正已生效（image 輪 0.0810＝畫面計入的算式）；**文字輸出仍被計費**（audio-flash 0.0366、audio-plus 0.0610，都吻合「文字照收」）——兩本帳修正未進正式 build，正式站客戶現在被系統性多收文字輸出的錢（單輪零點幾毫）、前端估算低於實收，已回報並提醒優先部署。公告與文檔通知續壓。

- feat：**剪貼簿貼上圖片**（使用者要求增加便利性）。主測試台任何分頁按 Ctrl+V／⌘V，圖片自動進當前頁面的圖片欄位。機制：`_PASTE_TARGETS` 路由表（每分頁的輸入按優先序）→ 取第一個可見目標（單檔欄位已滿讓位給下一個空欄，「貼首幀、再貼尾幀」自然成立）→ `DataTransfer` 塞回原本的 `<input type=file>` 再派發 `change`——與點選檔案完全同一條路，預覽/狀態/張數上限都由既有 handler 處理，零改動各分頁邏輯。目標區閃主色光暈＋toast 說明貼到哪。三個守則：焦點在文字欄位且剪貼簿有文字時不攔、頁面沒有可收圖欄位時安靜略過（t2i）、未登入不攔。涵蓋：文字（視覺模型）、圖片 i2i、影片（首幀→尾幀→vedit 參考圖→r2v 參考文件→動作動畫人物圖）、Spicy（來源圖／換臉圖）、即時對話附件（audio-3.0 純語音模型正確不收）、音樂靈感圖。AI Canvas 未納入（獨立頁面，LiteGraph 有自己的節點貼上語意）。Playwright 合成 paste 事件逐分頁驗證 9 條路由含兩個守則。`app.js?v=83`、`style.css?v=41`。

- feat：**Seedance 2.5 補上 i2v 與 r2v**（使用者回報只上了 t2v；對測試網關端到端驗證，另以免費探測確認正式環境的 2.5 路徑也已部署）。當初只上 t2v 的理由「其他模式要雲端網址」查證後只對影片/音訊輸入成立，圖片用 data URI 就通。實測發現三件事並落在程式碼裡：①**上游按圖片張數分類模式**（1 張→i2v、2 張→flf2v 首尾幀、3 張以上→r2v；非法解析度的錯誤訊息會標模式名，免費）——r2v 帶 1~2 張會被靜默當成別的模式跑，後端對 2.5 擋掉少於 3 張並講明；②**i2v/flf2v 不吃 `ratio`**（`InvalidParameter.TaskTypeConstraint`，輸出比例跟著首幀圖走），i2v 路徑對 2.5 不送 ratio，r2v 照送（實測 16:9→854×480）；③端到端（480P/4s）：i2v 一張圖 SUCCEEDED 且影片第一幀就是輸入圖（ffmpeg 抽幀比對）、r2v 三張圖 SUCCEEDED。vedit 維持不列（家族既有結論＋無雲端儲存無從實測）。測試 +1（`test_seedance_25_video_entries`）。

- feat：**部署閘門——測試網關先上、正式環境還沒有的模型自動隱藏**（Levi 裁示：這批上架是測試性質，正式部署前對外 playground 不可讓使用者選到六個新模型）。`/api/models` 回傳前用呼叫者的 key 查上游 `/v1/models`（快取 10 分鐘），`_DEPLOY_GATED_MODELS` 集合裡「上游清單沒有」的就拿掉；查詢失敗一律隱藏（fail closed）。音樂整類被拿掉時前端連任務選項一起藏。正式部署後自動出現、不用改程式碼；全部上線後把集合清空即可（機制保留給之後每一批「測試網關先上」的模型）。雙向實測：預設（正式環境）只剩 `qwen3.5-omni-plus-realtime`、音樂選項隱藏；`NENAI_BASE=測試網關` 時六個全部出現。順帶盤點確認：程式碼與設定**沒有任何寫死的測試機位址或測試 key**（上游一律看 `NENAI_BASE`，預設正式環境；測試 key 只存在 session 暫存目錄，探測腳本的 `--gateway test` 是既有的開發者工具選項）。`app.js?v=82`。

- feat：**新增三個 Lyria 音樂生成模型**（閘道端在測試機逐一實測生成與計費後轉知；本平台對 clip 文字版、clip 帶圖版、lyria-002 走完整 UI 複驗）：`lyria-3-clip-preview`（$0.04/次，30 秒 MP3）、`lyria-3-pro-preview`（$0.08/次，完整歌曲＋曲式說明）、`lyria-002`（$0.06/次，48kHz WAV）。
  - 「語音模型」分頁改名「語音與音樂」，新增第四個任務類型「音樂生成」；後端新增 `POST /api/music/generate`（multipart，走 `/v1beta/interactions` 單輪同步，timeout 180s）。
  - 回應兩種形態都認：lyria-3 音訊在 `steps[].content[]`、lyria-002 在 `outputs[]`；歌詞與曲式文字一併顯示在結果卡。**帶圖時 `input` 是 `[{type:"text"},{type:"image",...}]` 陣列**，與 Omni 影片在同一端點的 `user_input` 包法不同。
  - 圖片生音樂僅 lyria-3 支援（`image_input` 旗標；實測畫夜空月亮圖，歌詞唱出「夜空中，一輪明月」）；lyria-002 帶圖會被上游拒絕，UI 不顯示上傳欄、切換模型時自動清掉已選的圖。安全過濾的 `content_blocked` 錯誤原樣呈現給使用者。
  - 產出實測：clip 44.1kHz 立體聲 MP3 30.8s（afinfo 驗）、002 48kHz 立體聲 WAV 32.8s；花費累計 $0.04+$0.04+$0.06=$0.14 一分不差（固定價從 /api/pricing 讀）。
  - 測試 +1（`test_lyria_music_models` 釘住 image_input 旗標），共 34 條全過。`app.js?v=81`、`style.css?v=40`。

- fix：**語音結果區的卡片會「折欄」跑出畫面**（上架音樂模型時發現的既有問題，音樂卡較高所以特別明顯）：`#voiceResults` 行內把方向改成 column 但沿用了 `.results-area` 類別裡給圖片並排用的 `flex-wrap:wrap`，卡片一多就折到第二欄、出現橫向捲軸。修掉 wrap 之後又暴露第二層：`flex-shrink` 預設 1，高度不夠時卡片被**壓扁**（160px 內容被壓到 83px）而不是讓容器捲動。兩層都修（`flex-wrap:nowrap` + `.voice-result{flex-shrink:0}`），用 6 張高卡片驗證：自然高度、無橫向溢出、縱向捲動正常。

- test：**realtime 計費回歸對帳**（新 build 部署測試機後，穩定輪詢的乾淨單輪視窗）：`video_tokens` 修正**已生效**（omni-flash 圖片輪增幅 0.0852 與「畫面按文字檔計」算式吻合到第四位）；**白名單仍未生效**（兩個 audio-3.0 模型的增幅仍吻合「文字照收」）。回報後閘道端查明根因：不是部署落差，是**兩本帳**——實際扣款路徑（PreWssConsumeQuota）組 QuotaInfo 時漏帶白名單旗標，白名單只在寫日誌那條路生效；我們讀的 `/v1/dashboard/billing/usage` 是扣款側所以抓到真實多收。閘道端已修，等重建後再對一輪（預期 flash 0.03243、plus 0.04992）。

## 2026-08-16

- docs：**MAI 的 `max_n=1` 從「暫時止血」改成「長期維持」**（下一筆的後續；結論 📄 轉述自閘道端）。閘道端查出根因並修掉多收錢的部分（MAI 的 `num_output_tokens` 本來就按實際產出回報，是閘道的兜底把請求的 n 當乘數再乘一遍；現改以上游回報為準），但 **Azure 上游本來就靜默忽略 `n`、n=3 永遠只回 1 張**——原本寫的解鎖條件「實測 n=3 回 3 張」永遠不會發生。`app.py`／`README.md`／`memory.md`／測試註解已同步改寫。另外兩個 realtime 計費落差的結論：文字輸出照收是**部署落差**（白名單 commit 還沒進測試機的 build），video_tokens 沒計費是**真 bug 已修未部署**；部署後要對三個新模型重跑 basic＋image 對帳。

- fix：**MAI 三個 t2i 的張數鎖成 1**（使用者回報：MAI-Image-2.5-Pro 選 3 張只顯示 1 張、nen ai 平台實際計費 3 張）。對測試網關重現：送 `n=3` 閘道回應的 `data[]` 只有 1 筆 `b64_json`、`usage.num_output_tokens` 只報 1024（一張的量）、無 metadata 可補圖，但用量增幅 $0.3257 與「3 張 × 1024 tokens × model_ratio 2.5 × completion_ratio 21.2」完全吻合——**閘道收 3 張的錢、只回 1 張**，playground 端無從補救。已回報 nen-ai-platform 附完整重現；修好並實測 n=3 回 3 張之前，`MODELS` 的 `max_n` 與後端 `/api/image/generate`（保護 Canvas 與直接呼叫方）都鎖 1，並加測試釘住（`test_mai_image_n_locked_to_one`）。

- feat：**新增三個即時語音模型**（閘道 commit e87398614，對測試網關 192.168.0.245 實測後上架；正式環境尚未部署，公告與文檔通知待部署後補）：`qwen3.5-omni-flash-realtime`（全模態極速版）、`qwen-audio-3.0-realtime-plus`／`qwen-audio-3.0-realtime-flash`（純語音）。快照版 `qwen3.5-omni-flash-realtime-2026-03-15` 經使用者裁示**不上架**（渠道清單沒有它、查不到價）。實測重點（全部走 `scripts/probe_realtime.py`，事件形狀與前端一致）：
  - **兩個家族不能互推，三處實測差異**：①音色——omni-flash 與 plus 同一組 56 個（逐一全掃）；audio-3.0 是**另一組 15 個**（官方文件只列 5 個，完整清單來自非法音色的錯誤訊息、再逐一驗證出聲），互不相通（Tina/Ethan/Serena/Cherry 在 audio-3.0 全被拒）。②`turn_detection`——omni 收 `semantic_vad`/`server_vad`；audio-3.0 送 `semantic_vad` 回 `Unsupported turn_detection.type`，收 `server_vad`/`smart_turn`。③畫面輸入——audio-3.0 對 `input_image_buffer.append` **靜默忽略**（不報錯、usage 無 video_tokens、模型口頭說看不到）。
  - 對應的 MODELS 新欄位：`turn_modes`（前端據此重建「什麼時候算你說完」選單）、`audio_only`（藏附件鈕＋清殘留附件）。連線中切換模型會自動斷線並提示重新開始（先前單一模型不會遇到）。
  - **三模型的文字→語音、語音→語音（macOS `say` 合成 16kHz 真人語音）、omni-flash 的三張對照圖**全部通過，答案跟著輸入變；usage 是複數欄位（`input_tokens_details`），前端本來就相容。
  - **圖片計成 `video_tokens`**（480×480 一張約 225 個，官方費率與文字同檔）——前端 `rtApplyUsage` 先前只算 text+audio，帶畫面的輪次估價偏低（qwen3.5-omni-plus-realtime 既有問題一併修掉），現已把 `video_tokens`/`image_tokens` 併進文字檔計算並顯示「N 畫面」。
  - **用量增幅對帳抓到兩個閘道計費落差**（已回報 nen-ai-platform，前端照官方規則估）：①三個新模型有語音輸出時**文字輸出仍被計費**（audio-3.0-flash 一輪增幅 $0.0003663＝文字照收的算式；官方規則與 `audioOnlyOutputBillingModels` 白名單應為免費）；②omni-flash 的 **`video_tokens` 完全沒計費**（圖片輪增幅 $0.0005951＝「不計 video、文字照收」的算式）。
  - UI 驗證（Playwright 對本機服務 `NENAI_BASE=http://192.168.0.245` 實跑）：四模型下拉、音色與斷句選單依模型重建、audio-3.0 附件鈕隱藏且 semantic_vad 自動退到 server_vad、audio-3.0-plus 連線送文字（回答正確、估價 $0.000499 與費率吻合）、omni-flash 帶畫面提問（綠底黑方答對、用量列顯示「225 畫面」、VAD 自動關閉並還原）、截圖確認版面。`app.js?v=` 已 bump 到 80。
  - 測試釘住：`test_realtime_voices_verified`（四模型、兩組音色、預設音色、audio_only）、`test_realtime_turn_modes_verified`。麥克風收音品質仍屬未驗證（headless 無實體麥克風，與 plus 上架時相同）。

- feat：**新增 `scripts/probe_realtime.py`**——realtime（WebSocket）模型的標準探測工具，比照 `probe_model.py` 的慣例（`--gateway`／`--key-file`），五種測試：`basic`（文字→語音+文字）、`audio`（餵 16kHz wav）、`image`（靜音開門→圖片緩衝→提問）、`voices`（逐一驗音色，先用哨兵音色確認錯誤形狀）、`turn`（逐一試 turn_detection 詞彙）。送出的事件形狀刻意與 `static/js/app.js` 完全一致，探測通過＝前端那條路通。⚠️ 掃音色掃太快會把網關掃出 `thread pool exausted max_workers 100`——那是暫時性錯誤不是音色不支援，8 個被誤判的音色隔幾秒重測全過，判讀一定要看錯誤原文。

- refactor(UX)：**即時對話改成聊天版型**。先前是把 ASR/TTS 的結果卡片直接拿來當對話用，於是有六個問題：
  1. **對話是倒序的**（最新插在最上面）——要由下往上讀
  2. **「你」和「AI」長得一模一樣**，都是滿版卡片，掃視時分不出誰在說話
  3. **要按兩次才能講話**（開始對話 → 開啟麥克風），但這是語音對話，主要動作該是一鍵
  4. **最重要的狀態「尚未連線」是右上角的小灰字**，最不顯眼
  5. **輸入框在最上、對話在下**，與所有聊天介面相反
  6. **上傳畫面在側欄底部**，離輸入框很遠
  - 改法：訊息由上而下追加、最新在底並自動捲動（使用者往上翻看舊訊息時**不**強制捲回底部）；自己說的靠右上色、AI 靠左白底、系統訊息置中虛線；狀態改成左上角的**狀態燈＋文字**（灰／綠／黃／紅）；「開始對話」**連線後自動開麥克風**（權限被拒不當失敗，只提示可以先打字）；輸入列移到底部，附件改成輸入框旁的迴紋針＋縮圖預覽；麥克風開著時按鈕轉紅並呼吸，因為那是唯一會持續收音的狀態。
  - 側欄用詞去技術化：「回應形式」→「AI 怎麼回答你」、「斷句方式」→「什麼時候算你說完」，選項也改成白話（「自動判斷（聽語意）」「我自己按按鈕」）。
  - **踩到一個自己造成的坑**：CSS 我直接用了 `var(--primary)` / `var(--text-main)` / `var(--bg-input)`——這個專案根本沒有這三個變數（實際是 `--ali-orange` / `--text-body` / `--bg-white`）。結果是使用者訊息氣泡與送出鈕的背景**完全沒有套上**，白字配白底幾乎看不見。**不存在的 CSS 變數不會報錯**，只會靜默失效，所以一定要用截圖或 computed style 驗，不能只看程式碼。
  - 驗證：Playwright 實跑連線→送文字→附圖再問，回答正確跟著圖變（綠底黑方 → 「綠色，正方形」），附件送出後自動清空、縮圖留在對話裡、用量與花費照常顯示；淺色與深色主題各截圖確認。

- fix：**用語音提問時圖片沒有被送出**（使用者回報）。上一筆只在「送出文字」那條路徑掛了送影格，麥克風那條完全沒有——所以附了圖之後用說的問「分析一下這個圖片」，模型一路回「我沒看到圖片」，而畫面上沒有任何線索說明為什麼。
  - 修法不只是補上呼叫。實測發現**只要 VAD 開著，影格就不會被帶上**，而且與時機無關：用平台自己合成的真人語音（不是靜音）測「影格在語音前／語音後／語音後再 commit」三種時機，三種都失敗（答案是空白或罐頭的「白色」）。**關掉 `turn_detection` + 說完手動 commit** 才成立——三張對照圖全部答對，逐字稿也證明它確實聽到了問題。
  - 所以改成：**一附上畫面就把那一輪切成手動模式**（自動關閉 VAD、顯示「送出語音」鈕、狀態列明白寫出要按哪顆），答完再還原成使用者原本選的設定。`commitRealtimeAudio()` 也會一併送出影格。
  - 另外依使用者要求，**送出的畫面會以縮圖顯示在對話欄裡**（`.rt-frames`）——先前使用者只看得到自己打的字，無從確認這一輪到底有沒有把圖帶上去。
  - 驗證：用合成語音接進 `getUserMedia` 走完整 UI 路徑，三張對照圖的回答完全跟著圖變。

- fix：**`index.html` 的 `app.js?v=` 版號忘了 bump**，導致上一筆的功能在瀏覽器上看起來「壞掉」——新 HTML（有新選項）配舊 JS（沒有 realtime 函式），選了任務類型卻什麼面板都不出現。已改成 `v=77`。**改 `static/js/app.js` 就要一併 bump 這個版號**，否則使用者拿到的是半新半舊的組合，而且症狀看起來像功能沒做。

- feat：**即時對話補上圖片與影片輸入**（全模態）。上一筆只做了文字與麥克風，但這個模型的輸入模態是 Text／Image／Video／Audio。側欄新增上傳區，影片在瀏覽器端取樣成畫面（最多 8 張）再送。
  - **正確的送法完全是實測出來的**：先 `input_audio_buffer.append` 送 0.6 秒靜音 → 再逐張 `input_image_buffer.append`（裸 base64 JPEG）→ 才送問題。少了第一步會回 `Error append image before append audio.`
  - **⚠️ 而且必須先關掉 `turn_detection`**。開著 `semantic_vad` 時那段靜音會被 VAD 當成「沒有語音」丟掉，圖片緩衝就不會被帶上，**而且不會報錯**——模型回一個編出來的答案。前端改成帶畫面提問時自動關閉、`response.done` 後再還原成使用者選的設定。這個變因是拿 `turn_detection` 開／關兩組各跑三張對照圖才隔離出來的。
  - **四種照直覺推的寫法全都不行**，其中兩種是「收下但看不到」：`input_image` + `image_url`(data URI) → `Invalid video file.`；`input_image` + `image`(裸 base64) → **不報錯但看不到**；`image_url` 物件（chat 格式）→ 解析失敗；`input_video` + 影格陣列 → **不報錯但看不到**。
  - **我在這裡犯了一次「對照組太弱」的錯，記下來**：`input_video` 那個變體我只用一張圖測，它答「紅色」剛好對，我就在給對方的訊息裡寫成「可用，而且它真的看到了」。換成三張底色與形狀都不同的圖再測，三次全部回「白色、圓形」——那是編出來的答案。**單一樣本 + 聽起來合理的答案，跟真的讀到，在資料上長得一模一樣。**
  - 最終驗證（Playwright 走完整 UI 路徑，含 canvas 轉檔）：三張對照圖答案完全跟著圖變（藍/圓、綠/方、黃/三角），VAD 自動關閉並還原。影片路徑用瀏覽器即時錄一段內容已知的 webm 驗證，取樣 3 張、模型看得到三個不同形狀與底色——但**跨影格的順序與配色對應不保證精確**（實測那次形狀與底色的配對有錯位），這是模型能力而非傳輸問題。

- feat：**新增即時語音對話（realtime）**——「語音模型」分頁的第三個任務類型，模型是 `qwen3.5-omni-plus-realtime`。這不只是加一筆 `MODELS`，是把一整個先前被移除的功能做回來（前端 UI 在 `8c012ac` 之後被移除，因為當時閘道端阿里 adaptor 沒有 realtime 分支、握手成功後會立刻斷線）。對正式環境驗證。
  - **通路**：後端既有的 `/ws/omni` 代理現在通了（實測握手 34ms、文字輸入到首包音訊 2.1s、產出 24kHz 非靜音音訊）。維持走代理而不讓瀏覽器直連閘道——WebSocket 建構子不能帶 header，直連只能把金鑰塞進子協定 `openai-insecure-api-key.<key>`，金鑰會出現在前端可見的握手參數裡。
  - **音色 56 個逐一實測**。判準：`session.update` 帶音色 + `response.create`，有效的開始回傳音訊、無效的回 `Voice 'X' is not supported.`（用兩個亂編的名字與有效音色**交錯**跑，確認那個錯誤是音色造成的、不是速率限制）。舊清單裡的 `Chelsie` 實測不支援（qwen2.5-omni 的音色），已移除——直接照抄舊清單就會多一個永遠失敗的選項。
    - ⚠️ 兩個**行不通**的驗法，都試過才排除：`session.update` 的回應對任何字串都回 `session.updated`（連 `NoSuchVoiceZZZ` 都照收）；音訊位元比對也不行——同音色同輸入重跑兩次，長度相同但**位元不同**。
  - **花費估算四檔單價全部讀 `/api/pricing`**，不做人工快照。`audio_ratio` / `audio_completion_ratio` 在閘道是 `*float64` + `omitempty`，只有設定過的模型才會出現——我一度以為 API 不吐這兩個欄位、差點為此做快照，實際上是**我的查詢只印固定幾個欄位、把答案自己濾掉了**。
    - 計費規則有個容易做錯的條件：「同時輸出語音時輸出的文字不計費」的觸發條件是**該次回應真的產出了音訊 token**，不是「開了語音模式」。同一個 session 裡某次只回文字，那次的文字照常收費。估算已依 `audio_tokens > 0` 分支。
    - 端到端對帳：瀏覽器實測一輪（輸入 488 文字、輸出 20 文字＋63 語音）估出 $0.0049308；用閘道端提供的另一筆實際帳單（in 491／audio_out 72，實收 $0.0055）代入同一條公式得 $0.0054951，吻合。
  - **UI 實測項目**（Playwright 實跑，不是只讀程式碼）：連線／送文字／逐字稿逐字更新／用量與花費顯示／切換音色即時重送 `session.update`／切走分頁自動斷線並關閉麥克風／手動送出模式的按鈕顯隱。麥克風上行用合成音源接進 `getUserMedia` 跑完整條鏈（擷取→重取樣→PCM16→base64→上行），2.5 秒送出 9 個封包、關閉後不再送。
    - **待辦（先不要做）**：還有三個 realtime 模型在閘道端排隊——`qwen3.5-omni-flash-realtime`（全模態，另有等價快照版 `-2026-03-15`）、`qwen-audio-3.0-realtime-plus`、`qwen-audio-3.0-realtime-flash`（後兩個是純語音，沒有圖片影片輸入）。**正式環境目前只有 `qwen3.5-omni-plus-realtime` 可用**，那三個尚未部署、後台也還沒上架，所以不放進可選清單——列出來卻叫不動比沒有更糟。它們走同一條 realtime 路徑、計費語意相同（含 `AudioTokens > 0` 那個條件），倍率一樣四檔全在 `/api/pricing`，上架時不需要人工填價。
    - **未驗證**：真實麥克風的收音品質與回授。headless 沒有實體麥克風，`echoCancellation` 是否足夠要在真機上聽才知道。程式碼裡已把 ScriptProcessor 接到 gain=0 的節點避免原音回授（直接接 destination 會把麥克風原音播出來）。

- fix：**影片模型下拉選單的價格基準統一成「含配音」**（`static/js/app.js` 的 `formatPriceSuffix`）。使用者回報 Veo 那三行**一行之內自相矛盾**：文案寫「含原生配音」，價格卻是無配音那一檔（$0.2／$0.08／$0.03，含配音是 $0.4／$0.10／$0.05）。
  - 成因有兩層。第一層是基準不一致：解析度早就刻意固定成 720P 好比較，但配音是讀當下的核取方塊（預設關閉），而且切換配音時選單不會重建，所以選單顯示的幾乎永遠是無配音價。第二層更隱蔽——**各家族表上的基準價含意本來就不同**：萬相 2.6 i2v 的 $0.05 是**含**音訊、Veo 的 $0.2 是**不含**，兩者並排等於在比不同的東西，而畫面上完全看不出來。
  - 改法：`formatPriceSuffix` 有傳入解析度時（＝選單用）連配音也切成固定基準 `audio: true`，並在價格後面標「・含配音」，只對**價格真的會隨配音變動**的模型標（`_withAudio` 或 `_noAudioHalf`；Seedance、HappyHorse、萬相 2.7 配音不影響單價，標了只是雜訊）。沒傳解析度時（＝模型旁的即時提示）維持原本行為，跟著解析度與核取方塊即時變動。
  - 用 node 抽出這幾個函式實跑驗證：選單固定為 Veo `$0.4`／`$0.1`／`$0.05`（720P・含配音）、萬相 2.6 i2v `$0.05`（720P・含配音）、萬相 2.7 `$0.1`（720P，無標註）、動作動畫 `$0.18`（不標解析度）；即時提示在配音開／關與 1080P 下都跟著變。

## 2026-08-15

- test：**把 Veo 的每秒單價表釘住**（`tests/test_pure_functions.py`，+4 條，共 28 條）。`_VIDEO_SEC_PRICE` 在 `static/js/app.js`、是人工維護的顯示用快照，而 `/api/pricing` 只給**基準價**、不給檔次倍率，所以沒有任何 API 能驗證它——先前唯一的防線是註解。新測試從 Python 讀 `app.js`、正則取出三個 veo 型號的無聲／有聲共 18 個數字逐格比對，另加一條鎖住「veo 九個項目的 `resolutions` 必須是 `["720P", "1080P"]`」（480P 不是 Veo 的檔次）。
  - 已做變異驗證：把 Lite 的有聲價改回錯值，測試如預期失敗（`test_veo_per_second_prices[veo-3.1-lite-generate-001]`），改回來後 28 條全過。
  - 動機是閘道端的部署時程未定——這段期間拿正式站帳單對照會發現我們顯示的偏高，**那不是錯**。註解擋得住願意讀註解的人，測試會直接讓改動失敗並把人帶到判斷依據。

- docs：**Veo 3.1 Fast 的過渡期註記移除**（`static/js/app.js`；沒有行為變更）。閘道後台的 `model_price` 已從 0.3 改回 0.1，對方實抓 `https://nen.com.tw/api/pricing` 確認：standard `quota_type=1 price=0.2`、fast `0.1`、lite `0.05`。**我們顯示的官方價與實收現在一致**，下面那筆記的「改完之前顯示 $0.08 而實收 $0.24」的時間差結束。
  - 三筆都寫進後台設定了，所以 `/api/pricing` 的 `quota_type` 現在就正確、不必等後端部署（我們本來就不讀那個欄位）。
  - **還沒生效的是 Lite 的有聲檔次**——後端仍按無聲收（720P 有聲實收 $0.03、1080P 有聲實收 $0.05），要等那批 commit 部署。這段期間**我們顯示的高於實收**；不調回去，因為調回去等部署完又要再改一次，而「顯示得比實收高」不會讓使用者多付錢。
  - **後續（同日）**：閘道端的部署時程未定，所以 Lite「顯示高於實收」會持續不確定長度。已在 `_VIDEO_SEC_PRICE` 的 Lite 那筆加註「拿正式站的實際扣款來對照會發現偏高，那不是錯、不要改回去」——不加的話，下一個比對帳單的人很可能會把正確值改成錯的。另外，後台改動前那段期間（Fast 以 3 倍收費）的帳務經使用者決定不追溯，這條線結案。
  - 對方另外實測確認：送 `generateAudio: false` 上游確實產出無音軌的影片（ffprobe 只有 h264、無 audio stream；對照組不帶該欄位則有 AAC 立體聲）。所以使用者關閉配音時換到無聲檔計價是對的，我們的估算不用調整。先前查到的「veo-3.1 設 false 會報錯」那則論壇回報在這條路徑上不成立。

- docs：**Veo 的兩則註解跟著閘道端的裁示與實測更新**（`static/js/app.js`、`app.py`；沒有行為變更）。
  - **Fast 維持照 Google 官方價顯示，而且這是最終結論、不是暫時措施。** 使用者裁示後台那個 `model_price=0.3` 是**填錯**（不是刻意加成），要改回 $0.10/秒。所以先前記在下面那筆裡的「裁示後要改成 $0.24／$0.30／$0.36／$0.75／$0.90 那組實付數字」**作廢**，程式碼裡對應的註記已改掉。
    - ⚠️ **有時間差**：後台的值要人工到正式站改，改完之前顯示 $0.08 而實收仍是 $0.24。
  - **`generateAudio` 缺漏時是按「有聲」計費，不是無聲**（`_apply_audio_flag` 的註解原本寫反了）。閘道端實測：不帶該欄位時 Veo 回傳的影片帶 AAC 48kHz 立體聲且非靜音（volumedetect mean −24.9 dB），所以缺漏落到上游預設＝有聲。**本專案的行為不受影響**——三個影片提交路徑都無條件帶顯式 bool，錯的只有那句註解。註解已改成說明「為什麼一律帶顯式 bool」，避免日後有人把它改成「只在為真時才帶」。

（以下兩筆同屬 commit `41acafa`）

- fix：**Veo 3.1 Lite 的配音檔次價格漏了**（`static/js/app.js` 的 `_VIDEO_SEC_PRICE`）。原本註明「Lite 不支援配音」，只有無聲一組數字；實際上 Lite 有配音檔次——720P 無聲 $0.03／有聲 $0.05，1080P 無聲 $0.05／有聲 $0.08，等於有聲時的單價提示與「本次花費」**顯示得比實付低**（實付比顯示的高 66%，720P $0.03→$0.05、1080P $0.05→$0.08）。Lite 沒有 4K 檔次、4K 請求會落到 1080P 的價，所以 `4K` 直接沿用 1080P 的兩個數字（UI 目前也沒有 4K 選項）。
  - 來源：`nen-ai-platform` 的 session（後端 `VeoPriceMultiplier` 本來把 Lite 寫死成不支援音訊，同一個 commit 一併修了）。標準版與 Fast 的數字與我們表上的完全一致，不用動。
  - **`/api/pricing` 對 veo 標準版與 lite 回報 `quota_type=0`／`model_ratio=37.5` 是錯的（純顯示產物，實際扣款一直是按秒），但這個專案不受影響**：veo 的每秒單價走自己維護的 `_VIDEO_SEC_PRICE` 官方價表，`formatPriceSuffix` 一定先命中它；`estimateVideoTokenCost` 只認 `_SEEDANCE_DIMS` 裡的 Seedance 模型，veo 查不到會回 null。所以沒有任何地方把 37.5 換算成 $75/1M 顯示出來。
  - **Veo 3.1 Fast 的價格暫時維持顯示官方價**（720P $0.08/秒）。閘道後台配置的基準價是 $0.30/秒，是 Google 官方 Fast 基準價的 3 倍，是刻意加成還是設定填錯尚在確認；結論出來前不動，已在程式碼加註記說明確認後要改成實付數字，否則單價提示與 >$1 的確認框都會低估約 3 倍。
- fix：**veo 九個項目補上 `resolutions: ["720P", "1080P"]`，收掉 480P**（`app.py`）。原本 veo 沒有 `resolutions` 限制，所以解析度選單的 480P 選得到，但 480P 不是 Veo 的檔次。閘道端讀程式碼確認的實際行為：`metadata.resolution` 明確給了就原字串往下走，計費端用 `== "4k"` / `== "720p"` 判斷，**480p 兩個都不中，落進 else 的 1080P 檔**——Fast 變 $0.10/秒（而非 720P 的 $0.08）、Lite 變 $0.05（而非 $0.03），標準版因 720P/1080P 同價所以無差。而請求本身多半會被上游擋掉（Google 只認 720p/1080p，3.1 另有 4k）。所以留著這個選項的後果不只是「查不到價格靜默不顯示」，是**使用者被以較高檔次預扣、又拿不到影片**。4K 檔次不開：UI 選單裡沒有 4K，我們也沒實測過。
  - 依據是閘道端的程式碼判讀（`relay/relay_task.go`／`VeoPriceMultiplier`／`ResolveVeoResolution`），不是實測。收掉一個上游本來就不支援的選項風險夠低，所以直接做；反向的擴充（開 4K）仍照規矩要先實測。
  - 順帶確認了 Fast 低估 3 倍那個推論是對的：實付 = `model_price × 秒數 × video_ratio × group_ratio`，倍率表與後台配置的 `model_price` 完全解耦，所以 `model_price=0.3` 下 Fast 的實付是 720P 無聲 $0.24／有聲 $0.30、1080P $0.30／$0.36、4K $0.75／$0.90。裁示下來要改 `_VIDEO_SEC_PRICE` 時用這組數字。

## 2026-08-14

- chore：**專案搬到 `/Users/levi/claude_code/nen_ai_project/nenai-playground`**，跟閘道（`nen-ai-platform`）、文檔站（`Nen-AI-Docs-V1`）、官網（`website`）並列在同一層。原路徑 `/Users/levi/program_lab/AI_lab/alibaba/alibaba-model-nenAI`。
  - ⚠️ **venv 必須重建**，不是改設定能解決的：`venv/bin` 底下有 **23 個執行檔把舊路徑寫死在 shebang 裡**。已 `rm -rf venv` 重建並重裝 `requirements.txt`（含 pytest），24 條測試通過。
  - **Docker 容器名稱跟著變**：`docker-compose.yml` 沒有指定 `container_name`，專案名是從目錄名衍生的，所以 `alibaba-model-nenai-ai-model-tester-1` → **`nenai-playground-ai-model-tester-1`**。`README.md` 與 `CLAUDE.md` 裡的 `docker logs -f ...` 已更新。
  - 一併更新：`README.md` 的 `cd` 指令、`memory.md` 記錄的本專案路徑。
  - **不需要改的**：`scripts/update_announcements.py` 的 `NEN_ENV_PATH`（指向 `nen_ai_project/.env`，是別人的絕對路徑）、`CLAUDE.md`／`memory.md` 裡指向另外兩個 repo 的路徑、`.github/workflows/deploy-cloud-run.yml`（用 `SERVICE_NAME: nenai-testing-platform`，不依賴目錄名）、git remote。`update.md` 的歷史條目也保留原樣——那是當時的事實。
  - 驗證：容器重建後首頁與 `/canvas` 都回 200、`/api/models` 114 個模型（含新增的 `gemini-3.7-flash` 與 `kimi/kimi-k3`）、兩支腳本（公告、漂移比對）在新位置都正常。

- feat：**新增 `gemini-3.7-flash` 與 `kimi/kimi-k3`**（對正式環境完整驗證）。兩個都已在閘道的 `/v1/models` 與計費表：3.7-flash 是 `model_ratio` 0.75 / `completion_ratio` 5（$1.5→$7.5 每 1M）、kimi 是 1.5 / 5（$3→$15），kimi 的倍率與閘道端先前給的規格完全一致。
  - **⚠️ `gemini-3.7-flash` 的思考開關修正（同日稍晚）：`thinkingBudget: 0` 是有作用的，我一開始判錯了。** 原本寫成「關不掉、靜默忽略」而不給開關——那是**只用一個提示詞**測出來的結論。閘道端的 session 用第三組獨立數據（測試網關：`budget=0` 平均 88、不帶設定 269，區間完全不重疊）提出質疑，我在正式環境用兩種題目背對背重測：**多步算術題 `budget=0` 76～94 vs 不帶設定 180～294（完全不重疊）；短句陷阱題 0～158 vs 121～183（重疊）**。陷阱題的基準思考量本來就低（~167），沒有下降空間，所以看不出差別。改回 `thinking: True` 並移出 `_GEMINI_NO_THINKING_OFF`；端到端量測關閉思考後 `completion_tokens` 中位數 **252 → 96**。這是 `memory.md` 4d「**樣本數解決雜訊，解決不了變因沒被控制**」的第三個實例——每格 5 次很紮實，但只用一個 prompt 就把結論的適用範圍寫錯了。
  - ~~**`gemini-3.7-flash` 的思考關不掉，而且是「靜默」的。**~~（已由上一條推翻，保留原記錄） 送 `thinkingBudget: 0` **收下但照樣思考、完全不報錯**——實測兩種題目各 5 次，需推理題 5/5、簡單題 4/5 仍有 `thoughtsTokenCount`，而且 `budget=0` 的量（86～158）跟不帶設定（139～170）同級。這跟 `gemini-2.5-pro` 歸在同一個 `_GEMINI_NO_THINKING_OFF` 集合，但失敗方式不同：2.5-pro 會直接 400、看得出來。**靜默忽略比報錯危險**——不實測就會以為開關有效，使用者關掉後照樣付思考的錢。所以 `thinking: False`（不給開關），思考過程仍看得到。
  - **`kimi/kimi-k3` 差一點每一次呼叫都失敗。** 它是純思考模型（`thinking: False`），前端因此會送 `enable_thinking: false`，而後端原本對所有非 GPT 模型一律帶這個欄位——實測送 `false` 直接 400。新增 `_NO_ENABLE_THINKING_MODELS` 把這個欄位整個略過（不是改送 `true`）。
    - ⚠️ **那個錯誤訊息會把人帶往錯的方向**：回的是 `invalid temperature: only 0.6 is allowed for this model`，完全沒提到 thinking，而我們根本沒送 temperature（實測 `temperature=0.7` 反而正常）。真正的原因是閘道關閉思考時會改送上游不接受的取樣參數。
    - `reasoning_effort`：`max`／`high`／`medium`／`low`／`minimal` 都收，只有 `none` 會 400（同一個 temperature 錯誤）。但各檔的思考長度**分不出來**（每檔 3 次：`minimal` 43～235、`medium` 122～384，區間大幅重疊），官方也只寫支援 `max`——所以**不給強度選單**，寧可不給也不要給看不出有沒有作用的控制項。
    - **圖片輸入：已標 `vision`。** 一度依閘道端的說法寫成「只接受公網 URL、不接受 Base64」而不標——**那條後來被對方自己複驗推翻**（他們原本那筆失敗是模型服務剛開通時的短暫狀態，被推導成協定層級的限制）。我在**正式環境**自行複驗：三種尺寸各 3 次、**9/9 全成功**且顏色都答對，`prompt_tokens` 隨尺寸增長（64² → 120、256² → 211、512×384 → 377），證明圖確實被處理；經由平台的視覺路徑也正常（300×200 → 答「洋紅色」）。這是「**單一次失敗 ≠ 不支援**」的又一個實例。影片輸入只有公網 URL 被驗過、data URI 未驗，所以不宣稱支援影片。
  - 端到端驗證：兩個模型經由平台呼叫皆正常，思考過程都送得出來（**串流的欄位是 `reasoning`、非串流才是 `reasoning_content`**——我一度以為思考過程遺失，其實是測試腳本找錯欄位）。補了 2 條測試把這些約束鎖住，共 24 條通過。

- fix（工具）：**`scripts/update_announcements.py` 的寫入驗證改成輪詢回讀，不再重送 PUT**。原本是「寫完立刻回讀，不一致就重送一次」——那個判斷是錯的：這個 API 的寫入**不是即時生效**，補 108 則公告那次兩次都判定失敗、實際上早就成功了（幾分鐘後再讀 22 則都在、且沒有重複）。所謂「重送才成功」其實是第二次的回讀剛好等到傳播完成，**跟重送無關**。
  - 改成只 PUT 一次，然後以 0／3／8／15／30 秒的節奏輪詢回讀（最多等 56 秒）。
  - **重送的危險在別處**：這次的操作剛好冪等所以沒出事，但同樣的寫法用在 `--add-models`（附加一則）上就會**重複新增**。失敗訊息也改成明確叫人「不要直接重跑，先用 `--show` 確認」。
  - 用 stub 驗過三種情境：延遲兩輪才生效 → 成功且只讀 3 次；始終不一致 → 回報失敗；立刻一致 → 零等待。

- feat（流程）：**新增模型上架後的第二個必做步驟——更新 nen.com.tw 的系統公告（`Notice`）**，寫進 `CLAUDE.md`。原本只有「通知文檔 session」一項，現在是公告 + 文檔兩項。新增 `scripts/update_notice.py` 與英文草稿 `docs/notice-en.md`。
  - **公告一律用英文**（依使用者指定）。腳本內建中日韓字元防呆，避免把中文草稿送上去。
  - 腳本的三個安全設計，都是這個 API 的形狀逼出來的、不是操作習慣問題：①**整份覆寫、沒有 append**，要加一則得自己讀出現值再合併；②**沒有樂觀鎖**，讀寫之間有人從網頁改過會被靜默蓋掉，所以一律先備份到 `outputs/notice-backups/`（已 gitignore）——**那是唯一的還原方式，這個 API 沒有版本歷史**；③**所有客戶都看得到**，所以預設是預演模式，要寫入必須明確加 `--confirm`。
  - **憑證還沒有。** 需要 `NEN_TOKEN`（系統存取權杖，網頁 → 個人設定 → 生成）與 `NEN_USER_ID`。實測：呼叫模型的 API key（`sk-...`）打 `/api/option/` 會回「access token 无效」，而且**HTTP 狀態碼仍是 200**、只有 body 的 `success` 是 false——腳本因此不看狀態碼、只看 `success`，否則會把失敗當成功。上層目錄的 `.env` 只有 `DASHSCOPE_API_KEY`，不是這個用途。
  - 站上另有 `Announcements`（列表式公告，`GET /api/status` 的 `data.announcements`）與 `Notice` 是不同東西，那個由閘道端管理，本專案不動。
- fix（canvas）：**移除「一鍵執行整張圖」（Run All）**。使用者實際使用後認為用不到——原因可以從設計本身看出來：Run All 刻意不執行文字節點（它的「生成文字」是 opt-in，自動呼叫會把使用者寫好的 prompt 換成 LLM 生成的內容），所以在「文字 → TTS」這種兩節點的圖上實際只跑 1 個節點，跟直接按那個節點自己的按鈕完全一樣、還多一個確認對話框。它要有價值得是四五個節點的長鏈。移除 `canvas.js` 148 行（`runAll`／`_topoSort`／`_upstreamOf`／`_nodeHasResult`／`_runnableNodes`）＋事件綁定＋工具列按鈕，零殘留。需要時可從 commit `4545908` 取回。`canvas.js` v=37。
- docs：`scripts/probe_model.py` 的 `drift_check()` 註記 `/api/pricing` 是公開端點、且有 `pricing_version`（全域一個 hash、每模型各一個），要做定期變更偵測時比 diff 整份 JSON 便宜（📄 轉述自閘道端 session，未自行驗證）。
- docs：`memory.md` 記下 `kimi/kimi-k3` 等待渠道上架，以及**上架時先不要標 `vision`** ——該模型的圖片／影片輸入只接受公網 URL、不接受 Base64，而本平台的視覺輸入送的是 data URI。

## 2026-08-12

- fix：**「本次花費」與送出前確認改用每秒 × 秒數，先前嚴重低估**。兩者現在共用同一個入口 `videoCostFor()`，不再各算各的。
  - 先前「按次計費的就加 `model_price`」是錯的（原因同上一條：`model_price` 其實是每秒基準價）。實測徽章增幅：萬相 2.7 T2V 720P 5 秒 **$0.10 → $0.50**、HappyHorse 1.1 1080P 10 秒 **$0.02 → $1.80**（90 倍）、Veo 3.1 720P 8 秒含配音 **完全沒計入 → $3.20**（它是 token 型又查不到尺寸表，`estimateVideoTokenCost()` 回 null 就被跳過了）。
  - **Seedance 維持用自己驗證過的 token 公式**，不改成「每秒 × 秒數」：它的幀數是 `秒數 × 24 + 1`，多出來那 1 幀是整支影片一次性的開銷，token 公式比較精確（720P 5 秒 $0.7623 vs 每秒換算的 $0.7560）。所以 `videoCostFor()` 是「先試 token 公式，沒有才用每秒單價」。
  - **超過 $1 的送出前確認終於對按次登記的模型生效**。先前 HappyHorse 一支 10 秒 1080P 只被估成 $0.02、永遠不會提醒；現在會先問。實測會跳確認的：HappyHorse 1080P 10 秒、Veo 3.1 含配音 8 秒、動作動畫 mix 6 秒。
  - 配音與服務模式會改變單價，所以三個呼叫點的 `costInfo` 都補上 `audio` 與 `mode`。實測萬相 2.6 i2v Flash 5 秒：含配音 $0.25、無配音 $0.125。
  - ⚠️ 過程中修掉一個我自己引入的 bug：`animateMode` 是宣告在 animate 分支內的 `const`，我在外層的 `addVideoTask()`／`addVideoCost()` 引用它會拋 `ReferenceError`，而那段在 `try` 裡——會讓**所有任務類型**的影片送出都失敗。`node --check` 抓不到這種作用域錯誤，是逐行看程式碼時發現的。改成在呼叫點直接從 DOM 讀。
  - `gemini-omni-flash-preview` 仍不計入：長度由模型自己決定，沒有秒數可乘。`app.js` v=73。

- fix（UI）：**影片模型的參考單價全部統一成「每秒」，改用各廠商的官方定價**。先前的顯示有三種形態且都不對：按次計費登記的模型顯示「/次」（`$0.1/次`、`$0.02/次`）、token 計費的顯示「/1M」（`$75→$75/1M`）、Seedance 顯示「/次（720P 5秒）」。
  - **「/次」是嚴重低估**：查閘道 `relay/channel/task/ali/adaptor.go` 的 `EstimateBilling()` 可見 `OtherRatios["seconds"] = duration`，而 `relay_task.go` 會把所有 ratio 逐一乘進額度——`model_price` 其實是**每秒基準價**。HappyHorse 標 `$0.02/次`、但官方 720P 是 $0.14/秒（基準 ×7），一支 5 秒的片子差 **35 倍**。
  - **改用官方定價而不是從平台倍率反推**（依使用者指示）。倍率是「基準價 × 解析度倍率 × 秒數 × …」層層相乘的結果，要還原每秒單價得先知道基準價對應哪一檔解析度、秒數有沒有被乘進去——那是一連串看不見的假設，錯了也不會有徵兆。官方定價是可以直接對照查證的單一數字。新增 `_VIDEO_SEC_PRICE` 表（萬相全系列、HappyHorse、Veo），Seedance 維持用自己驗證過的 token 公式換算（結果與官方每秒價完全相符）。
  - 處理了三個特例：**配音會改變單價**（Veo 含配音是純影片的兩倍：$0.20→$0.40；萬相 2.6 i2v 關掉配音減半：$0.05→$0.025），切換配音開關會即時重算；**動作動畫依服務模式計價**（wan-std $0.18／wan-pro $0.26）且固定 720P 輸出，標籤不顯示解析度。
  - 順手移除 `onVidAudioToggle()` / `onT2VAudioUpload()` 的**完全重複定義**（同一份程式碼出現兩次，後者靜默蓋掉前者）。
  - `gemini-omni-flash-preview` 維持 token 顯示：它走 chat completions、長度由模型自己決定，沒有「每秒」這個概念。
  - 瀏覽器逐項驗證：t2v/r2v 兩種任務下所有模型都顯示每秒價、配音開關即時改變單價、解析度切換正確（wan2.7 720P $0.10 → 1080P $0.15）、動作動畫依模式切換且不標解析度。`app.js` v=72。

- fix（UI）：**影片模型的參考單價改用「每秒」，不再寫「/次」**。先前 Seedance 系列顯示成「約 $0.2614/次（720P 5秒）」——那個數字會跟著時長 slider 一直跳，而且「/次」讓人以為是固定價；影片本來就按秒計費，每秒單價才是穩定、可跨模型比較的資訊。
  - 換算用**整整 24 幀**，而不是把 `seconds=1` 丟進 `estimateVideoTokenCost()`——那條公式的幀數是 `秒數 × 24 + 1`，多出來的 1 幀是整支影片只算一次的固定開銷，拿去當每秒費率會高估。新增 `estimateVideoPerSecond()`。
  - **這條換算對得上兩個已知的官方每秒單價（720p）**：`dreamina-seedance-2.5` 算出 $0.2311、`dreamina-seedance-2.0` 算出 $0.1512，與 README 記錄的官方數字完全相符，所以不是憑空換算。
  - 順手修掉一個小的不一致：下拉選單裡的價格先前用「建立選單當下」的解析度，等於比較基準隨使用者的操作順序而變。改成固定用 720P 當基準（選單是拿來比較模型的），模型旁的即時提示才跟著目前選到的解析度走。實測選 1080P 時選單仍顯示 720P 基準價、提示顯示 $0.3742/秒（1080P）。
  - ⚠️ **還有三個模型仍顯示無意義的 `/1M`**：`wan3.0-video`（$0.1→$0.1/1M）、`veo-3.1-generate-001` 與 `-lite`（$75→$75/1M）、`gemini-omni-flash-preview`（$1.5→$17.5/1M）。這幾家的「token ↔ 秒」對應關係我們**沒有驗證過**，Seedance 那條公式不適用（wan 在 720P 的官方價是 $0.10/秒，若沿用 Seedance 公式會算成完全不同的數字）。沒有把握就不換算，待決定要用實測或改採官方每秒價。

- feat：**視覺語言模型的圖片上傳前先縮到長邊 2048**。實測 `qwen3-vl-flash` 的圖片 token 在 2048×2048 之後就**封頂不再增加**（512²=285、1024²=1053、1536²=2333、2048²=2529，之後 3072² 與 4096² 都是 2529），也就是更大的圖不會被更精細地看，只是讓使用者多等上傳時間。縮圖在瀏覽器端做，後端與上游都不必改。
  - 兩個實作細節：**本來就在上限內的圖原樣送出、不重新編碼**（重壓一次只會讓畫質變差）；讀不出來的檔案（壞檔或瀏覽器不支援的格式）也原樣送出讓上游回報錯誤——縮圖只是最佳化，不該變成新的失敗點。縮過的圖統一輸出 JPEG 0.92（縮圖後的照片用 PNG 反而更大）。
  - 有縮圖時會提示使用者，說明「超過這個尺寸不會提高辨識精細度」，不是默默改掉他的檔案。
  - 瀏覽器驗證：一次上傳 4096×3072 與 1024×768 兩張，攔請求確認實際送出的是 `2048x1536`（JPEG）與 `1024x768`（原樣 PNG），長寬比都維持。
  - 📌 這個上限是本專案自己實測的。文檔站先前提供過相近的數字（1024²=1,037、2048²=2,513），但送出去的結論要用自己的數據，所以重跑了一次；兩邊的絕對值略有差異（測試圖內容與 prompt 不同），封頂位置一致。

- feat（canvas）：**新增語音辨識（ASR）節點**（`audio` → `text`）。這是唯一整個分類缺失的節點——先前 `audio` 型別在整張畫布上**只有產出端、沒有消費端**（語音 TTS 會吐 audio，但沒有任何節點吃得下）。補上之後才組得出「文字 → 配音 → 轉回文字」這類閉環，也能把上傳的錄音接進後面的文字節點。可接 TTS 節點輸出，或直接上傳音檔。
  - 只提供非串流型號（`qwen-audio-3.0-asr-flash`）：串流版走另一條端點 `/api/voice/asr/stream`，而在節點圖裡要的是最終完整逐字稿再往下游送，中間結果沒有用處。
  - 實作上踩到兩個坑：①**`wireConfigOverlay()` 會把 `.cv-controls` 整個搬到獨立圖層**，那是給「上半部放預覽、設定收進浮層」的節點用的；ASR 沒有媒體預覽，照搬會讓節點本體變空，改成跟文字節點一樣把控制項留在節點本體。②**新增節點類型要同時加 `NODE_TYPE_LABELS` 與 `NODE_MENU_TYPES` 兩張表**（前者給輸出插槽旁的快速新增選單，後者給工具列的「+ 新增節點」），只加一邊的話按鈕會出現但點下去毫無反應——`type` 查不到就直接 return。已在 `NODE_MENU_TYPES` 上方加註解說明。
  - 瀏覽器驗證：節點建立、沒有音訊時擋下並提示、選檔後狀態更新、攔 fetch 確認送出 `model` 與檔名／MIME 正確、回應的逐字稿寫進輸出框並往下游送。`canvas.js` v=36。
- feat（canvas）：**補上 Negative Prompt / Seed / 生成張數**，見上方 `b26471d` 條目。

- fix（canvas）：**影片節點的解析度與時長改由 `MODELS` 決定，不再寫死**。先前時長 slider 寫死 2–15 秒、解析度寫死 480P/720P/1080P，完全沒讀 `min_dur`／`max_dur`／`dur_step`／`no_duration`／`resolutions`，於是每個有特殊限制的模型都有一部分選項送出去就被上游拒絕——而主測試台會擋，**同一個模型在兩個介面行為不一致，且 Canvas 那邊是靜默失敗**：`veo-3.1-*` 只有 4/6/8 三個合法值（step 2）其餘全拒、`dreamina-seedance-2.5` 的 2/3 秒與 1080P 都拒、`dreamina-seedance-2.0-fast` 的 1080P 拒、`happyhorse-*` 的 2 秒拒、`wan3.0-video` 最長 30 秒卻拉不到、`gemini-omni-flash-preview` 兩個參數都不該送。新增 `videoLimits()`／`applyVideoLimits()`，換模型、換模式（連線變動）、還原存檔時都重算，並把超出範圍的舊值**夾回合法區間並對齊到 step 的格子上**。
  - 影片編輯節點先前解析度寫死 `1080P`、連選單都沒有，`duration` 存在 properties 卻**從未送出**：兩者都補上。時長 `0` = 保留來源影片長度（後端看到 0 就不送給上游），`1` ~ `minDur-1` 是非法區間，拉進去會跳到最小合法值。
  - 以瀏覽器逐項驗證：五個模型的限制正確切換；在 1080P + 2 秒的狀態切到 `dreamina-seedance-2.5` 會夾成 480P + 4 秒；veo 落在 4/6/8 上；`gemini-omni` 兩列隱藏；vedit 拉到 1 秒跳成 2。`canvas.js` 版本號跳到 `v=31`。
- fix（docs）：**`gemini-3.5-flash-lite` 的 `thinkingBudget` 表第二次修正——「逐次隨機約四成」是 prompt 相依，不是模型屬性。** 文檔站拿到金鑰複驗後在 `101`／`128`／`512` 全部得到 8/8，與我先前的 3～4/8 直接衝突。他們沒有斷言我錯，而是把完整 request body 貼過來要我比對。我把兩組 body 放進**同一輪背對背**執行，一次就分辨出「請求不等價」與「上游改版」：他們的多步算術題每個 budget 都 8/8（`thoughtsTokenCount` 210～369），我的短句陷阱題 3～4/8（144～180）。**是模型自己判斷這題需不需要推理**，跟預算大小無關。
  - 兩邊獨立確認的部分：門檻是 **101**（`100` → 0/8、`101` → 會思考），≤100 是合法但靜默不思考；3.5-lite 的可接受範圍 `1 to 32768`、2.5-lite 是 `512 to 24576`。
  - **`-1` 不等於「一個很大的固定預算」**——連 `2+2` 這種一定不需推理的題目，`-1` 也是 8/8 都思考，而固定預算在陷阱題上只有 3～4/8。平台的「思考開」送的就是 `-1`，不受影響。
  - `memory.md` 4d 補上第三個維度：**樣本數解決雜訊，解決不了「變因沒被控制」**。這張表改了兩次，第一次是次數不足、第二次是次數夠但只用一個 prompt。判準：下結論前先問「換一組輸入，這個結論還成立嗎？」
- fix（UI）：**圖片空狀態的置中先前沒修好——根因找錯了**。第一次修的時候我只讀到 `.results-area` 的 `display: flex`，就據此推理「是 `align-content: flex-start` 讓單行高度只等於內容高」。實際用瀏覽器量 computed style 才發現 `#imageResults` 被後面一條規則改成 **`display: grid`**（`repeat(auto-fill, minmax(260px, 1fr))`），空狀態被塞進其中一個 260px 的網格欄，所以整套 flex 假設都是無效的，只改 `align-content` 當然沒有用（`grid-auto-rows: min-content` 也不吃 `stretch`）。正確修法是**連 `display` 一起覆寫回 flex**，`.empty-state` 的 `flex:1` 才會吃滿容器。量測確認：容器 920×641、空狀態 880×601、四邊留白都是 20px，水平垂直皆置中。這是 `memory.md` 4d 第④類「拿間接證據當行為證據」的又一次——CSS 原始碼是間接證據，computed style 才是行為。`style.css` 版本號跳到 `v=30`。
- feat：**MAI Image 開放自訂尺寸**。尺寸選單多一個「自訂尺寸…」選項（只有 `MODELS` 有 `custom_size` 的模型才出現，目前是三個 MAI t2i 型號），選了才顯示寬／高兩個輸入框。前端依 `custom_size` 的三條約束即時驗證：每邊 ≥ 768、總像素 ≤ 1,056,768、**自動往下對齊到 16 的倍數並明確顯示「實際輸出 W×H」**——告知後修正，不是靜默改寫。不合法時擋在送出前，省下一次必定失敗的呼叫。
  - **兩條送出方式都可選**（同日稍晚新增）：自訂尺寸區多一個「送出方式」下拉，可選 `size` 字串或 `width`／`height` 欄位，提示文字會即時顯示這次實際會送出的欄位（例如 `實際輸出 1360×768（尺寸會對齊到 16 的倍數）　送出 width: 1360, height: 768`）。後端 `ImageGenerateRequest` 新增 `width`／`height`，帶了就把 `size` 拿掉——**上游本來就以 width/height 為準**（實測 `size=1024x1024` 配 `width/height=2000x2000` 會回 400 而非照 size 產圖），拿掉只是讓送出的請求跟使用者選的機制一致，看日誌時不會誤以為兩個都在生效。對齊行為兩條路一致（`width=1366` 同樣得 1360）。
  - 📌 **`width`／`height` 一度是無效的，這次重測才發現閘道修復已上正式環境。** 先前實測送 `{"width":2000,"height":2000}`（若被讀取必定超限報錯）竟正常產出 1024×1024——根因是閘道的 `dto.ImageRequest` 沒宣告這兩個欄位，未宣告的欄位落進 `Extra` map 而 `MarshalJSON` 刻意不把 `Extra` 合併回去，於是靜默消失。當時那個修復還在 feature branch、只到測試網關，所以第一版只做 `size` 那條路。這次重測同一個請求回 400（`total pixel count (4000000) exceeds...`），確認已生效。
  - 端到端實測（經由平台、讀 PNG header）：`size="1024x976"` → 1024×976；`width=1024, height=976` → 1024×976。UI 以瀏覽器逐項驗過送出方式切換與提示文字。
  - 實測依據：`size="1200x800"`（不在預設清單裡）直接對閘道送 → 輸出 1200×800 完全相符；經由平台送 `size="1088x960"` → 輸出 1088×960 完全相符。UI 行為用瀏覽器逐項驗過：非 MAI 模型不出現自訂選項、預設隱藏輸入框、1366×768 提示「實際輸出 1360×768（尺寸會對齊到 16 的倍數）」、700×1024 擋下「每邊至少 768 像素（目前 688×1024）」、1024×1536 擋下「總像素 1,572,864 超過上限」。
- fix：**MAI Image 的 `1366x768` 從來沒有真的產出 1366x768**。閘道端 session 同步發現「上游會把尺寸往下對齊到 16 的倍數」，我獨立複驗確認：請求 `1366x768`（Azure 官方文件自己舉的例子）實際拿到 **1360x768**，讀 PNG header 確認、回應本身完全不提示尺寸被改過。`_MAI_IMAGE_SIZES` 改成直接登記對齊後的 `1360x768`／`768x1360`（實測上游照收、輸出與請求完全相符），使用者選什麼就拿到什麼。其餘三個尺寸（`1024x1024`／`1152x896`／`896x1152`）本來就是 16 的倍數，不受影響。
- docs：MAI 的兩條尺寸限制補上「**互相獨立**」這點——`767x1024` 總像素只有 785,408、遠低於 1,056,768 上限，照樣因為短邊不足被拒。只檢查總像素會誤判。
- docs：**Seedance 配音的驗證狀態更新成一半已驗**。📄 閘道端 session 在正式站實測過「關閉」那半：`generate_audio: false` 確實送達（上游任務物件原樣回吐 `false`），產出 mp4 用 `ffprobe` 檢查只有一條 h264 stream、沒有 audio track。⚠️「開啟後真的會出聲」那半仍未驗證。另確認 **2.0 / 2.5 不論開不開配音價錢都一樣**（閘道的 `silentVideoRatioMap` 只列了 `bytedance-seedance-1.5-pro`，官方定價沒有音訊維度）。
- fix：**Seedance 的配音開關先前被鎖死在關閉**。所有 Seedance 型號原本標 `audio: False`，前端不但隱藏開關、還會**強制把勾選狀態設成 `false`**（`onVidModelChange`），於是每一次請求都明確送出 `metadata.generate_audio=false`；而上游這個欄位預設是 `true`，等於平台主動把本來會有的聲音關掉，使用者還無從開啟。送出的管線其實早就接好了——`_apply_audio_flag()` 從一開始就把 Seedance 專用的 `generate_audio` 帶上，缺的只是 MODELS 的旗標。改成 `audio: True` 把控制權交還使用者，**預設維持不勾選**（依使用者指定「有需要再開」；無聲對 `bytedance-seedance-1.5-pro` 另有 0.5× 折扣）。
- fix（UI）：**選到 Seedance 時隱藏 Negative Prompt 與 Prompt Extend**。閘道 doubao adaptor 的請求結構裡沒有這兩個欄位，文字內容只用到 `prompt`，`metadata` 對不上的鍵直接丟棄——這兩個控制項對 Seedance 是純裝飾。新增 `no_negative_prompt`／`no_prompt_extend` 兩個 MODELS 旗標，前端據此整組隱藏並清空值，避免隱藏後仍把值送出去（`memory.md` 第 6 條）。
- ⚠️ **上面兩項都是讀閘道 Go 原始碼推斷的，未經實測**——屬於 `memory.md` 4d 第④類「拿間接證據當行為證據」，本專案在 `wan2.7` 的 `max_ref` 上正是這樣栽過。影片單價高，依使用者指示這次不做取樣驗證。**配音實際會不會出聲、關掉是否真的折價，仍待驗證。**
- docs：一併盤點出**上游支援但平台從未送出**的 Seedance 參數（未實測）：`camera_fixed`、`frames`、`output_format`、`priority`、`return_last_frame`、`draft`、`omni_reference_task_type`、`callback_url`／`service_tier`／`execution_expires_after`／`safety_identifier`／`tools`，以及 `metadata.content`／`metadata.image_role` 兩個逃生門。記在 README，未實作。
- feat：**文字生成新增「記住上下文」開關**（預設開啟，維持原本的行為）。關閉後每次送出都不帶歷史訊息、視為全新對話。實作上 `textChatHistory` 一律照常累積，開關只決定「這一輪送什麼給模型」——所以關掉再打開能接著先前的內容繼續，不必重問一次。切換模型時歷史仍然沿用（依使用者指定，不做提示清空）。
  - 動機是成本而非功能偏好：每一輪都必須重送完整歷史，`prompt_tokens` 是**累加**的，聊到第 N 輪要付前 N-1 輪的錢，總花費大致隨輪數平方成長。使用者原本看得到「本次花費」在漲，卻沒有辦法停。
  - 端到端實測（`gemini-3.5-flash-lite`，經由容器）：第 1 輪告知「我的貓叫小黑」；第 2 輪帶歷史 → 答「小黑」、`prompt_tokens=37`，不帶歷史 → 答「我不知道你的貓叫什麼名字」、`prompt_tokens=16`。關閉省下 57% 的輸入 token，且記憶行為確實被切斷。
- fix（UI）：**NenAI Spicy 頁的空狀態提示框一併置中**，與圖片、影片一致。
- fix（UI）：**圖片生成的空狀態提示框改成置中**，與影片生成一致。原因是兩區的排版模式不同——影片區是 `flex-direction: column`，空狀態的 `flex: 1` 能吃滿高度所以會置中；圖片區是 `flex-wrap: wrap` + `align-content: flex-start` 的網格，單行的高度只等於內容高，所以提示框貼在左上角。修法是只在「容器裡只有空狀態」時把該行撐滿高度（`#imageResults:has(> .empty-state) { align-content: stretch; }`），有結果時仍維持靠上的網格排列。`style.css` 版本號跳到 `v=28`。
- **部署**：正式環境補上先前累積的 13 個 commit（`474bc3f` ～ `8d9828e`）。部署後端到端實測確認：`grok-4.3` 的 `reasoning_effort` 各檔生效（`none` 答 8、`medium` 答 9 且 completion=449）、`gemini-3.5-flash-lite` 與 `gemini-2.5-flash-lite` 的思考開關各 3 次一致（關 completion=1、開分別為 170～201 與 113～125），確認平台送的 `thinkingBudget: -1` 在部署後仍穩定啟動思考。
- docs：文檔站複驗了先前送過去的兩項（grok-4.3 的 `reasoning_effort` 支援度、MAI Image 的尺寸限制），結論都與我們一致並已修正三語文檔。他們補充的 MAI 發現值得記著：**「每邊 ≥ 768」與「總像素 ≤ 1,056,768」是兩條獨立限制**——`767x1024` 總像素只有 785,408、遠低於上限卻仍被拒（訊息是 `'width' must be at least 768 pixels`），只看總像素會誤判。`1536x1024` / `1024x1536` 這兩個在別家常見的尺寸在 MAI 上都不可用，我們的 `_MAI_IMAGE_SIZES` 已經沒有列它們。
- docs：**修正先前送給文檔站的 grok-4.3 推理數據——原本每檔只跑一次，樣本不足以支撐結論。** 每檔重跑 5 次後：`none` 0／`minimal` 中位數 334（282–379）／`low` 272（205–353）／`medium` 528（359–615）／`high` 366（265–489）。結論（`reasoning_effort` 是傾向、不是預算上限）**恰好是對的**，`medium` 的整個範圍都高於 `high`、反轉可重現；但 `minimal`／`low`／`high` 三檔區間大幅重疊，原本那組單次數字看起來有穩定差距其實分不出來，而且同檔內單次差距可達 1.7 倍。已通知文檔站把逐檔的具體數字改成範圍或中位數。教訓寫進 `memory.md` 新增的第 4d 條：「有沒有效果」單次夠用，「效果多大／誰比誰大」一定要重複取樣並附範圍。
- docs：同一份實測發現 `grok-4.3` 的 `none` 不只是把推理 token 歸零——測試題（正解 9）在 `none` 下五次全答 8、其餘四檔五次全答 9。這種穩定且離散的差異 n=5 就足以下結論，也比 token 數對使用者更有意義，已寫進 README。
- docs：補齊 `gemini-3.5-flash-lite` 與 `gemini-2.5-flash-lite` 的 `thinkingConfig` 對照（文檔站要的資料）。兩者思考都**預設關閉**，不帶 `thinkingConfig` 或 `budget=0` 時 `thoughtsTokenCount` 欄位不出現。
- **fix（同日稍晚，推翻上一條的一半）**：上面那條原本寫「`thinkingBudget: 128` 在 3.5-lite 被靜默忽略、實務門檻約 512」——**是錯的**，每格只取樣一次的產物。文檔站沒有 key 無法複驗、決定先不寫進文檔，我改用每格 8 次重跑，結果是：3.5-lite 的門檻在 **96～128 之間**（≤96 一律不思考且不報錯），但**超過門檻後思考與否是逐次隨機的**（128 → 4/8、512 → 3/8、600 → 3/8），**且與預算大小無關**。沒有 512 這個門檻，只是 128 與 512 各抽一次剛好一個沒中一個中。2.5-lite 則是乾淨的：<512 一律 400、≥512 穩定 8/8 思考。
- **確認我們平台不受影響**：`thinkingBudget: -1` 在兩個型號、簡單題與需推理的題目上各 8 次**全部都思考**（8/8），是唯一可靠的設定；平台的「思考開」送的正是 `-1`。若當初照錯誤結論改送固定預算，3.5-lite 上的思考開關會變成約四成機率才生效、而且不會有任何錯誤訊息。
- **`CLAUDE.md` 新增「送出實測數據時：標明樣本數與當初的測試目的」**（跨 session 協作那節底下）。核心是「**資料換了用途，證據等級要跟著提高**」——為了 A 目的做的測試不會自動夠格支撐 B 目的的結論，而發佈方無從得知那份數據當初測到什麼程度。配套的另一半是收資料方要講清楚打算怎麼用（「參考」與「發佈給客戶」對送出方的義務完全不同）。另補一條判斷樣本數的順序：先確定要寫的句子，再回頭看需要多強的證據。
- `memory.md` 第 4d 條**判準修正**：原本寫「『有沒有效果』單次夠用」，這次就是被這句話坑的。正確的是——看到**有**效果一次就夠（能做到不會是運氣），看到**沒有**效果**一次遠遠不夠**（「沒觀測到」≠「不存在」）。**否定的結論比肯定的結論需要更多樣本**，而我們最常犯的錯正是拿單次的「沒看到」去斷言「不支援／被忽略／有門檻」。
- docs：`qwen3-vl` 只確認了「吃圖片輸入、走 OpenAI 相容的 `image_url` content part、data URI 可用」，其餘參數（解析度上限、多圖上限、影片輸入）**未測**，已明確告知文檔站不要引用。

## 2026-08-10

- fix：另一個 session 對照 nen-ai-platform（阿里/DashScope 渠道）目前的路由與參數結構，幫忙抓出這個測試平台影片端點的兩個實際不會生效的 bug，修正如下：
  - **影片端點的解析度選擇完全沒送到上游**：`/api/video/{t2v,i2v,r2v,vedit}` 原本會把使用者選的 720P/1080P 用 `_res_to_wh()` 轉成 `width`/`height` 放進 payload 頂層，但上游統一任務 API（`TaskSubmitReq`）根本沒有 width/height 這兩個欄位，會被直接忽略——等於 UI 上選的解析度從未真正送達，一律吃伺服器端每個模型自己的預設值。改成直接把 resolution 字串（"720P"/"1080P" 等）放進 payload 的 `size` 欄位，交由上游依模型判斷要轉成 size 還是 resolution 參數；順手移除已無用的 `_res_to_wh`/`_RESOLUTION_WH`。
  - **`/api/video/animate`（wan2.2-animate-mix/move）沒有送 `images`，人物圖/參考影片實際上沒傳到上游**：此檔案裡 i2v handler 早就在註解寫明「平台 TaskSubmitReq 只認 images（陣列），media/image 會被忽略」，i2v/vedit/r2v 三個 handler 也都確實額外補了 `images` 欄位，但 animate handler 唯獨漏補，只送了會被忽略的 `media` 陣列，等於上游收到的圖片/影片內容是空的。補上 `payload["images"] = [人物圖, 參考影片]`（順序對應 wan2.2-animate 上游規則）。
  - 修完後 `docker compose build --no-cache && docker compose up -d` 重新建置部署。

- fix：接續上一筆，把上一筆改動造成的跨廠商迴歸補起來，並依閘道原始碼收掉一批「送了也沒用」的參數與 UI（commit 待補）：
  - **上一筆改成統一送 `size: "720P"` 會讓 Veo 靜默降級、Seedance 完全吃不到**。四個影片端點是所有廠商共用的，但每家 adaptor 取值的欄位都不一樣：阿里讀頂層 `size`（"720P" 這種字串它三條分支都解析得出來）；Veo 的頂層 `size` 是用**小寫 `x`** 切 `WIDTHxHEIGHT` 的，`"1080P"` 切不開會 fallback 成 720p，而且同一個函式也用於計費——使用者選 1080P 會拿到 720p 的影片、還照 720p 計費，全程不報錯；Seedance/Dreamina 則是連頂層 `size` 和 `duration` 都不讀，只吃 `metadata.resolution` 與頂層 `seconds`（字串）。新增 `_apply_res_and_duration()` 一次把三種形式都送出去（頂層 `size` + `metadata.resolution` 小寫 + 頂層 `duration` int 與 `seconds` 字串），畫面比例同時送 `metadata.ratio`（Seedance）與 `metadata.aspectRatio`（Veo）。三家的 metadata 都是整包 unmarshal 進各自的 payload struct、未知 key 直接忽略，所以重複送是安全的。
  - **拿掉萬相系列無效的配音開關**。阿里的 task adaptor 從頭到尾沒有讀統一請求的頂層 `audio`，整份程式只有 `wan2.6-i2v-flash` 會去讀 `metadata.audio`（bool，關閉後費用減半）。其餘萬相型號有沒有聲音完全由上游決定，UI 上那個開關是純粹的裝飾。MODELS 裡把這些型號的 `audio` 改成 `False`（前端本來就會據此隱藏整列），只留 `wan2.6-i2v-flash`；新增 `_apply_audio_flag()` 把 `metadata.audio`（wan2.6-i2v-flash）、`metadata.generateAudio`（Veo）、`metadata.generate_audio`（Seedance）三個欄位一次帶齊。
  - **i2v 只開放「首幀生成」**。adaptor 的 i2v 分支只取 `images[0]` 當 `first_frame`，尾幀／驅動音訊／影片延伸片段送過去都會被靜默丟棄——使用者只會拿到一支「看起來就是沒照做」的影片，沒有任何錯誤訊息，非常難查。MODELS 加上 `i2v_modes: ["first_frame"]`，前端把其餘模式從選單 hidden 掉，後端再擋一次（直接打 API 的呼叫端會拿到明確的 400）。
  - **r2v 擋掉影片檔**。萬相/HappyHorse 的 r2v 會把收到的每個檔案都當成參考「圖片」（wan2.6 走 `reference_urls`、wan2.7/HappyHorse 走 `media` 的 `reference_image`），混入影片會被上游拒。MODELS 加 `ref_images_only`，前端把 `<input accept>` 收成只剩 `image/*`，後端擋下並提示改用動作動畫或視頻編輯。
  - **上傳的音訊改走 URL**。先前是把音檔轉成 `data:audio/...;base64,...` 塞進 `metadata.audio`（t2v/i2v/r2v 的配樂）或 `media` 的 `driving_audio`，但上游只接受 `audio_url`（一個真的能下載的 URL），等於使用者上傳的音訊**從來沒有生效過**。改成先用 `_cloud_put()` 上傳到雲端物件儲存再帶簽名網址；沒有任何雲端後端可用時直接回 400（本機 `outputs/` 路徑上游一樣抓不到，送出去只是註定無聲）。驅動音訊刻意不放進 `media_arr`——那個陣列會整包變成 `payload["images"]`，混進音訊會被上游當成參考圖。
  - **萬相圖像編輯補 `max_ref: 2`**。上游 `WanImageInput.images` 的硬上限就是 2 張，但前端在 `max_ref` 未設時的 fallback 是 9 張（`app.js:697`），等於放任使用者選到必定被拒的張數；後端 `image_edit` 也同步以 `_WAN_EDIT_MODELS` 收到 2 張。
  - **圖片端點 timeout 120 秒 → 300 秒**。萬相 2.7 系列不在閘道的同步圖片模型清單裡，會走 `text2image/image-synthesis` + `X-DashScope-Async`，由閘道代為輪詢到 `SUCCEEDED` 才回應——客戶端看到的是一次很慢的同步請求，120 秒不夠。
  - 前端改動：`app.js?v=51` → `?v=52`。

- test/fix：對正式網關（`https://nen.com.tw`，透過已部署的 playground.nen.com.tw）實測驗證上面兩筆的修正，並修掉實測才暴露出來的一個限制（commit 待補）：
  - **驗證方式**：不是只看「有沒有成功產出影片」——舊版程式碼一樣會成功，只是解析度悄悄掉回預設值，光看回應永遠查不出來。所以每支都把成品下載回來用 `ffprobe` 量實際寬高才算數。
  - `wan2.7-t2v` @ 1080P → 實際 **1920x1080** ✅（阿里確實讀到頂層 `size`）
  - `veo-3.1-fast-generate-001` @ 1080P → 實際 **1920x1080** ✅（`metadata.resolution` 成功蓋過 `SizeToVeoResolution` 對 `"1080P"` 切不開而 fallback 成 720p 的錯誤推導）
  - **`dreamina-seedance-2.0-fast` 不支援 1080P**，送出直接被上游回 `InvalidParameter`。這不是這次改壞的，而是「解析度終於真的送達」之後才浮現的既有限制——先前上游收不到解析度，一律當 720p 跑，所以看起來一切正常。逐一實測後確定支援範圍：`dreamina-seedance-2.0-fast` 只有 480P/720P；`dreamina-seedance-2.0` 四種都支援（含 4K）；`bytedance-seedance-1.5-pro` 支援到 1080P（4K 不行）。fast 版的 t2v/i2v/r2v 三種模式都一樣被擋。
  - 修法：MODELS 的影片條目新增 `resolutions` 欄位（僅 `dreamina-seedance-2.0-fast` 三個條目設為 `["480P", "720P"]`），前端 `onVidModelChange()` 把解析度選單裡不支援的選項 `hidden` 掉、若當前選取值被隱藏就自動跳到第一個可用值——這段同時把原本寫死的「vedit 隱藏 480P」邏輯合併進同一個判斷。
  - **上面那張支援矩陣記錄的是「提交時上游收不收這個參數值」，不是「產得出來」**——這兩件事實測後確認不能劃等號：`dreamina-seedance-2.0` @ 1080P 提交回 200，但任務跑了約 15 分鐘後以 `failed` / `Unknown error` 收場，沒拿到影片。只跑過一次，無法區分偶發失敗或該組合實際不可用，Seedance 家族因此**尚未有任何端到端（下載成品量寬高）的解析度驗證**；已在 `README.md` 標註。萬相與 Veo 則都已經 ffprobe 量到實際 1920x1080。
  - 前端改動：`app.js?v=52` → `?v=53`。

- feat：上架阿里萬相 3.0（`wan3.0-video`），依另一個 session 在 nen-ai-platform（分支 `feat/carrothub-channel-support`）剛接完的渠道實作規格（commit 待補）。**僅完成程式碼，尚未打過任何一次上游請求**——使用者決定先不測，等確認網關與預算後再驗。
  - all-in-one 模型：同一個模型 id 統一支援文生／圖生／參考生／視頻編輯，MODELS 裡以四個 type 分別呈現。最長 30 秒（其餘萬相家族 15 秒）、解析度 480P/720P/1080P（每秒 $0.05/$0.10/$0.20）。
  - **`ratio` 預設要是 `adaptive` 而不是 16:9**。`ratio` 與 `resolution` 是兩個互相獨立的參數；但本專案的 t2v/i2v/r2v 三個 handler 原本是硬寫 `form.get("ratio", "16:9")`（前端只有 vedit 有比例控制項，其餘模式根本不送這個欄位，所以一律吃到 16:9）。改成 `form.get("ratio") or _default_ratio(model)`，讓預設值依模型家族決定。
  - **改走 metadata 覆寫管道指定媒體用途**：因為模型名沒有 i2v/r2v/videoedit 後綴，上游無法從模型名判斷每個媒體的用途，改以「MIME／副檔名 ＋ 位置」推斷。新增 `_apply_explicit_media()`，把我們自己在 `media_arr` 裡早就標好 type 的陣列直接放進 `metadata.input.media`，完全不依賴上游推斷；只對 `_WAN30_ALLINONE_MODELS` 生效，不影響既有模型。理由有二：（1）**位置推斷表達不了我們實際有的語意**——上游對影片一律推成 `video`（video-edit 的來源影片），永遠不會產出 `first_clip`（影片續寫的起始片段），而我們 i2v 的「影片延伸」模式送的正是 `first_clip`；（2）實測階段的除錯價值，萬一失敗就能確定問題出在 type 詞彙本身而非推斷邏輯。上游確認這是預期用法。
    - **更正一則我一度誤報的事實**：我最初判斷「上游靠副檔名推斷，而我們送的 data URI 沒有副檔名，所以推斷必定落空、所有媒體都會被當成圖片」，並把這個結論寫進了本檔與 README。經閘道端查證後確認**不成立**——他們的比對用的是 `strings.Contains` 且清單裡含 `"video/"`／`"audio/"` 兩個 MIME 片段，`data:video/mp4;base64,...` 會命中 `"video/"` 而正確判定成影片。同一輪他們順帶收斂了兩個真實弱點（base64 內容湊巧含 `video/` 會誤判、`Contains` 掃整個 URL 導致 `pic.png?src=song.mp3` 誤判成音訊），改成結構化判定：data URI 取 MIME 前綴、HTTP URL 先切掉 query string 再比副檔名。上述描述已據此改正。
  - 順手修掉一個因為新模型才浮現的既有寫死規則：前端「vedit 隱藏 480P」原本是為 `wan2.7-videoedit` 寫死的通則，但 480P 正是萬相 3.0 的基準價位、它的視頻編輯確實支援。改成模型有明確 `resolutions` 清單時以清單為準，沒有才套用 vedit 通則。
  - ⚠️ **`media` 的 `type` 詞彙尚未驗證**：`wan3.0-video` 的官方 API 文檔目前是邀請制、尚未公開，可公開查證的只有模型名、端點、`resolution`/`ratio`/`duration` 與定價。`first_frame`／`last_frame`／`driving_audio`／`first_clip`／`reference_image`／`video` 這組取值是閘道端實作者從 wan2.7 已公開文件推導的，兩邊都沒打過真請求。實測後若上游回報 type 不合法，要把正確清單回報給閘道端校正 `wan3MediaType()`。
  - 前端改動：`app.js?v=53` → `?v=54`。
  - **UI 文案不標「尚未實測」**（使用者指示）：這個模型在官方模型頁上已是正式發佈的狀態（定價、輸入輸出模態、能力說明都公開可查，只有 API 文檔是邀請制），所以四個條目的 `desc` 照官方說明寫，不在使用者看得到的地方加註測試狀態。`README.md` 與本檔屬開發者文件，該保留的驗證狀態警告照留。

- feat/fix：新增 `MAI-Image-2.5-Pro`，並修掉整個 MAI Image 家族列了兩個永遠不能用的尺寸的既有 bug（commit 待補）。網關選擇：正式環境 `https://nen.com.tw`（使用者指定），已從 `/v1/models` 確認 `MAI-Image-2.5-Pro` 確實存在。
  - 新增 `MAI-Image-2.5-Pro` 的 t2i 與 i2i 兩個條目，並加進 `_NO_REF_STRENGTH_EDIT_MODELS`（與家族其餘型號一致，不支援 `ref_strength`）。
  - **既有 bug**：`MAI-Image-2.5` 與 `MAI-Image-2.5-Flash` 的四個條目都列了 `1536x1024` 與 `1024x1536`，兩者總像素都是 1,572,864，**超過上游 1,056,768 的上限、一定會被拒**——三個尺寸裡有兩個從來就不能用，使用者選到就是必然失敗。實測確認三個型號行為一致。
  - 尺寸不是固定枚舉而是兩條約束：**每邊 ≥ 768 像素**、**總像素 ≤ 1,056,768**。違反時分別回 `'width'/'height' must be at least 768 pixels` 與 `Invalid dimensions WxH: total pixel count (N) exceeds the maximum of 1056768`。改成共用的 `_MAI_IMAGE_SIZES` 常數，五個尺寸（`1024x1024`／`1366x768`／`768x1366`／`1152x896`／`896x1152`）逐一實測確認可用——這次特別逐一驗證，因為這個 bug 的成因正是「列了沒驗證過的尺寸」。

- feat：`glm-5.2` 的呼叫方式改走 Anthropic Messages 格式的 `/v1/messages`（原本與其餘文字模型共用 OpenAI 相容的 `/v1/chat/completions`）（commit 待補）。
  - 新增 `_ANTHROPIC_MESSAGES_MODELS` 分流集合與三個轉換函式：`_build_anthropic_body()`（`system` 提到頂層、`stop` → `stop_sequences`、`max_tokens` 必填、丟掉不適用的 penalty/enable_thinking/reasoning_effort）、`_anthropic_generate()`（非串流）、`_anthropic_stream()`（串流，把 Anthropic 的 SSE 事件轉成前端既有的 `{reasoning}`／`{content}`／`{done,usage}` 協定）。前端完全不用改。
  - **實測出來的兩個行為差異，動手前先驗過才寫**：
    - **思考過程只有串流看得到**。串流時思考走獨立的 `thinking` content block（`thinking_delta` 事件），拿得到、顯示效果與原本的 `reasoning_content` 相同；**非串流回應完全不含思考過程**，只有 `text` block，但 token 照樣被消耗（實測回一個「2」花掉 159 個 completion token）。
    - **思考關不掉**。`thinking.type=disabled`、`thinking.type=enabled`、額外欄位 `enable_thinking:false` 實測都無效（同一題 output token 150／151／148／111）。相對地 `/v1/chat/completions` 上 `enable_thinking:false` 是真的有效的（同題 completion token 139 → 1）。因此 MODELS 裡把 `glm-5.2` 的 `thinking` 旗標關掉，不顯示一個沒有作用的開關——這與本專案既有的處理慣例一致。
  - 端到端實測（本機服務打正式網關）：非串流拿到 `content='2'` 與 usage、串流拿到 195 字元的思考過程 + `content='一，二，三！'` + usage，兩條路徑都正常。

- fix：修好使用者回報的「gemini 3 pro image 我現在選不了生成結果大小」——Gemini 圖像模型改走 Gemini 原生的 `/v1beta/models/{model}:generateContent`，輸出尺寸與比例都以結構化的 `imageConfig` 控制（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **問題確認**：Gemini 圖像模型在 MODELS 裡標了 `no_size: True`，前端會整個隱藏尺寸選單；而四個 i2i（編輯）條目連 `aspect_ratios` 都沒有，所以編輯模式下比例選單也一起被隱藏——**完全沒有任何輸出尺寸控制**。後端更徹底：`_generate_gemini_chat_image()` 的 `if image_files:` 分支根本沒有用到 `aspect_ratio` 參數，就算補上選單也不會有作用。
  - **原因**：先前走的是 OpenAI 相容的 `/v1/chat/completions` + `modalities`，那條路徑上結構化的 `imageConfig` 會被上游靜默忽略（這次重測仍然如此：送 `imageConfig.aspectRatio=9:16` 得到的還是 1408×768 橫式），所以當初只能用「在 prompt 文字裡以自然語言要求比例」的權宜做法。那個做法本身其實是有效的（實測 9:16 → 768×1376、1:1 → 1024×1024），但**解析度完全沒有辦法控制**。
  - **解法**：實測發現網關支援 Gemini 原生的 `/v1beta/models/{model}:generateContent`，而且 `imageConfig` 在那裡真的生效——`aspectRatio` 文生圖與圖像編輯都吃，`imageSize` 給 1K/2K/4K 得到長邊 1024/2048/4096。改寫成 `_generate_gemini_image()` 走原生端點，MODELS 拿掉 `no_size`、改成各型號實測支援的 `sizes`，四個 i2i 條目補上 `aspect_ratios`，前端在編輯模式也送 `aspect_ratio`。
  - **各型號的 `imageSize` 支援度不同**，逐一實測（每個型號 × 每個值都實際產圖量寬高）：`gemini-3-pro-image` 與 `gemini-3.1-flash-image` 三種都真的生效；`gemini-2.5-flash-image` 接受參數但**靜默忽略**、永遠回 1024；`gemini-3.1-flash-lite-image` 的 2K/4K 直接回 400。後兩個型號的 `sizes` 因此只列 `1K`——列出來卻做不到的選項比沒有更糟。
  - 原生端點**不接受 `candidateCount`**（送了直接 400），所以多張改成並發打 n 次、沿用原本的重試次數補足缺的張數。另外 `/v1/images/generations` 不支援這些模型（回 `only imagen models are supported`）。
  - 加了防呆：只有 `1K`/`2K`/`4K` 會被轉成 `imageSize` 送出，避免瀏覽器快取著舊版前端、送來 `1024*1024` 這種其他家族的尺寸字串導致上游 400。
  - 端到端實測（本機服務打正式網關，下載成品量寬高）：文生圖 1K+1:1 → 1024×1024、2K+1:1 → 2048×2048、4K+16:9 → 5504×3072；圖像編輯 2K+9:16 → 1536×2752。
  - 前端改動：`app.js?v=54` → `?v=55`。

- fix：前端價格顯示不要四捨五入（使用者回報：「wan2.7 阿里是 0.075 但在 nen 顯示是 0.08」）（commit 待補）。
  - 成因在 `app.js` 的 `formatUsd()`：金額 `>= 0.01` 時會 `Math.round(n * 100) / 100`，強制捨入到小數第二位，`0.075` 就變成 `0.08`。後端 `/api/pricing` 本來就刻意保留原始精度（那裡的註解已寫明「不要粗暴 round 到固定小數位」），捨入只發生在前端這一處。
  - **這不只是顯示不精確，是會造成誤解**：使用者看到 nen 標 0.08、阿里標 0.075，會直接得出「這個平台比原廠貴」的結論——實際上單價是一樣的。
  - 改成顯示完整數值：用 `Number(n.toPrecision(12))` 消掉浮點誤差（`0.1+0.2` 那種 `0.30000000000000004`），12 位有效數字遠超過任何實際單價需要的精度、不會改變真正的數值；小於 `1e-6` 時 `String()` 會輸出 `1e-7` 這種科學記號，改用 `toFixed(12)` 展開再去掉尾隨的 0。
  - 驗證過的輸出：`0.075` → `0.075`（原本 `0.08`）、`0.000035` → `0.000035`、`0.1+0.2` → `0.3`、`1e-7` → `0.0000001`、整數與 0 維持原樣。
  - 前端改動：`app.js?v=55` → `?v=56`。

- fix：圖像編輯多張參考圖只有第一張生效（使用者回報：「wan2.7 上傳兩張照片，似乎只會吃第一張」——確認屬實）（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **成因**：`image_edit()` 送 multipart 時把欄位命名成 `image`、`image_2`、`image_3`……，但上游只認**重複的 `image` 欄位**，編號欄位名會被靜默丟棄。沒有任何錯誤訊息，使用者只會覺得「模型沒有參考到第二張圖」。
  - **驗證方式**：兩張純色圖（純紅、純藍）＋ 要求模型輸出「所有參考圖顏色的 50/50 混色」，再量輸出圖片的平均 RGB——只吃第一張會是紅色，兩張都吃會偏紫。這樣不必靠肉眼判斷「有沒有參考到」。結果：
    - `image` + `image_2`（原本的寫法）→ RGB(202,61,69) 紅色，**只有第一張生效**
    - 重複的 `image` → RGB(159,9,247) 紫色，兩張都吃 ✅
    - `image[]` + `image[]` → RGB(165,3,247) 紫色，兩張都吃 ✅
  - 改用重複 `image` 欄位。因為 `/v1/images/edits` 是所有編輯模型共用的，逐一確認沒有弄壞其他家族：`wan2.7-image`／`wan2.6-image`／`qwen-image-2.0`／`gpt-image-2`／`dola-seedream-5.0-pro` 都實測兩張都吃到。單張的情況欄位名跟以前完全一樣，不受影響。
  - **順帶發現 MAI 系列的編輯端點只接受「剛好一張」參考圖**，多送直接 400 `Exactly one image file must be attached for edit requests`。原本 MODELS 沒設 `max_ref`、前端 fallback 是 9 張，等於放任使用者選到必定失敗的張數（在舊的欄位命名下反而是「第二張被丟棄所以不會報錯」，改成重複 image 後就會現形）。三個 MAI 編輯條目補上 `max_ref: 1`，後端 `_MAI_EDIT_MODELS` 同步限制。
  - 端到端實測（本機服務打正式網關，透過 `/api/image/edit` 上傳兩張）：平均 RGB (82,4,250)，兩張都生效。

- test/fix：接續上一筆，把**其餘所有多參考圖的圖片與影片模型**都查過一遍（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **圖片編輯：修正後全數通過。** 沿用同一套可量測的驗法（純紅 + 純藍兩張參考圖、要求輸出「所有參考圖顏色的 50/50 混色」、量輸出的平均 RGB）：`wan2.7-image-pro` RGB(165,3,254)、`qwen-image-2.0-pro` RGB(51,80,163)、`gpt-image-1.5` RGB(105,51,184)、`dola-seedream-5.0-lite` RGB(228,196,181) 全部兩張都吃到。Gemini 走的是另一條程式路徑（原生 `generateContent`，多張 `inlineData`），也另外驗過：`gemini-3-pro-image` RGB(135,31,146)、`gemini-3.1-flash-image` RGB(128,7,155)，同樣正常。
  - **影片參考生影片（r2v）：上游兩條不同路徑都驗過。** 用同樣的混色手法，取產出影片的首幀量平均 RGB——`wan2.6-r2v-flash`（上游走 `input.reference_urls`）RGB(120,34,182)、`wan2.7-r2v`（上游走 `input.media` 的 `reference_image`）RGB(162,193,195)，兩張參考圖都有生效。影片端送的是 JSON 的 `images` 陣列而不是 multipart，沒有上一筆那個欄位命名的問題。
  - **修：參考圖張數上限前後端不一致**（同一類「靜默丟棄」的問題，只是方向相反）。原本前端把視頻編輯的參考圖寫死 `slice(0, 3)`，後端卻是 `5 if "happyhorse" in model else 3`——`happyhorse-1.0-video-edit` 明明支援 5 張、desc 也寫「最多 5 張參考圖」，但使用者最多只能選 3 張，白白少了兩個欄位；r2v 則是前後端都完全沒有上限，使用者可以一次丟 20 張進去。
    - 改成以 MODELS 的 `max_ref` 為單一來源：後端新增 `_VIDEO_MAX_REF` / `_video_max_ref()`，前端新增 `vidRefLimit()`，兩邊都讀同一份資料。
    - 只為**有實據的**模型標上限：`happyhorse-1.x-r2v` 9 張、`happyhorse-1.0-video-edit` 5 張（desc 本來就這樣寫）、`gemini-omni-flash-preview` r2v 3 張（後端本來就 `image_files[:3]`）。其餘沒有實測過上限的模型維持不設限——不硬編一個猜測值進去擋人，這正是先前 MAI 尺寸那個 bug 的成因。
  - 前端改動：`app.js?v=56` → `?v=57`。

- fix：逐一實測所有圖像編輯模型的參考圖張數上限，**並更正我先前設錯的萬相上限**（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **更正一個我造成的迴歸**：前面把萬相編輯的 `max_ref` 設成 2，依據是閘道端 `WanImageInput.images (≤2)` 這個 Go struct——**那是推斷、沒有實測**。實測後確認那條約束不適用於 `/v1/images/edits` 這條路徑：`wan2.7-image` 與 `wan2.7-image-pro` 送 9 張都被接受，而且第 9 張確實有生效。這個錯誤的限制正好對應使用者反映的「wan2.7 nen 只能支援兩張上傳，之前阿里的可以上傳到 9 張」——我當時還回覆說那是上游硬限制、不是我們砍功能，那個說法是錯的。已改回 9。
  - **驗法**：「送得出去」不等於「有用到」，上游可能接受請求卻靜默忽略多出來的圖。所以用「前 N 張純紅 + 最後一張純藍 + 要求模型輸出所有參考圖的混色」，量輸出的平均 RGB——出現藍色成分才代表最後那張真的被讀進去。探測上限時由大往小試，被拒絕不會產圖也就不花錢，只有第一次被接受才有成本。
  - 實測結果：
    - `wan2.7-image` / `wan2.7-image-pro` → 9 張，第 9 張有效（RGB 120,0,236 / 83,1,254）
    - `wan2.6-image` → **上限 4**（第 4 張有效 RGB 197,31,185；送 5 張回 `the last message must contain 1 to 4 images`）——跟同家族的 2.7 不一樣
    - `qwen-image-2.0` / `-pro` → **上限 3**（第 3 張有效 RGB 161,24,115；送 4 張回 `supports 0~3 image content items`）
    - `MAI-Image-2.5` / `-Flash` / `-Pro` → **只接受剛好 1 張**
    - `gpt-image-2` / `-1.5`、`dola-seedream-5.0-pro` / `-lite`、Gemini 四個型號 → 9 張，第 9 張有效
  - 後端把原本 `is_qwen2_edit → 3 / _WAN_EDIT_MODELS → 2 / _MAI_EDIT_MODELS → 1 / 其餘 9` 這串特例判斷收掉，改成從 MODELS 推導的 `_EDIT_MAX_REF`，與前端 `imgMaxRef` 讀同一份資料——先前這類「兩邊各自寫死、各自演化」正是張數不一致的來源。

- feat/docs：依官方文件補上萬相 2.7 `size` 的「方式一：規格值」（1K/2K/4K），並查核所有文生圖模型現有尺寸清單的有效性（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **一個不用產圖就能驗 size 的手法**：同時送一個超出範圍的 `n`（13，高於萬相的上限 12）。上游先驗 size 再驗 n，所以錯誤訊息提到 pixels/dimensions 就是 size 不合法、提到 `parameters.n` 就是 size 通過了，兩種情況都不會產圖。**注意這招只對「`n` 會被拒」的模型免費**——MAI 與 Seedream 實測會接受 `n=13` 而直接產圖（詳見下方的疏失記錄）。
  - **查核結果**：`qwen-image-2.0`/`-pro`、`wan2.7-image`/`-pro`、`wan2.6-t2i`、`z-image-turbo` 目前清單裡的每一個尺寸都通過驗證，沒有無效項。`1K`/`2K`/`4K` 這種規格字串**只有萬相 2.7 兩個型號接受**，qwen、wan2.6-t2i、z-image-turbo 一律回 `Invalid size format`。
  - **補上規格值選項**：萬相 2.7 的 `size` 官方支援兩種寫法且不可混用，我們先前只提供「方式二（明確寬高）」。方式一是官方推薦，且**有圖片輸入時輸出寬高比會跟隨輸入圖**（多圖時取最後一張）再縮放到該規格，這是方式二做不到的。依文件加入：pro 文生圖 1K/2K/4K、標準版文生圖 1K/2K、兩者的編輯情境 1K/2K。
  - **⚠️ 這裡有一個重要教訓：網關的 size 驗證是寬鬆的，擋不住官方文件寫的個別限制。** 實測 `wan2.7-image` 送 `4096*4096`、pro 開組圖送 `4K` 都能通過網關驗證——它只檢查總像素落在 589824~16777216（也就是 pro 文生圖那組最寬鬆的範圍），但文件明說 `wan2.7-image` 不支援 4K、組圖模式只到 2K。**所以「打網關試得通」不能拿來當作支援的證據**，清單以官方文件為準。這跟先前 Seedance「提交被接受但生成失敗」是同一類陷阱。
  - 因為網關不擋，前端得自己擋：新增 `sequential_max_size` 欄位，組圖模式開啟時把 4K/4096*4096 從尺寸選單濾掉、當前選取值若被濾掉就跳到第一個可用值；`onImgSequentialToggle()` 切換時會重算選單。`onImgModelChange()` 在模型不支援組圖時本來就會回頭呼叫 `onImgSequentialToggle()`，所以加了 `_imgSizeRefreshing` 旗標擋無限遞迴（已用 Node 模擬兩函式互相呼叫驗證過，不會爆堆疊）。
  - 順帶修正尺寸標籤：`1K`/`2K`/`4K` 原本寫「長邊約 1024/2048/4096」，那是 Gemini `imageConfig.imageSize` 的語意；萬相的規格值是**總像素**（1K=1024*1024）。兩家算法不同，標籤改成只寫規格本身不寫死換算結果。
  - 邊界確認：萬相總像素下限實測是 589824（768*768），送 `767*767` 回 `Total pixels (588289) must be between 589824 and 16777216`，與文件一致。

- **疏失記錄（自我檢討，非功能變更）**：上面那個 size 掃描的第一版腳本有「回 200 就立刻中止」的保護，我改寫第二版時把它拿掉了、只記錄不中止。結果對 MAI 與 Seedream 家族送出的 9 次請求都回 200，而且每次都帶著 `n=13`——最多可能產生 117 張圖，粗估數美元。成因是這兩個家族不會拒絕 `n=13`（Seedream 的 `2k`/`4k` 本來就是它的合法尺寸，MAI 則是似乎照單全收）。**教訓：這種「用非法參數換免費驗證」的手法，必須先確認該模型真的會拒絕那個哨兵參數，而且保護機制不能在重寫時被拿掉。**

- feat/docs：研究清楚 Gemini 圖像模型的尺寸參數運作方式，並把支援的比例從 5 種補到 10 種（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **機制**：`imageConfig` 這組參數**不是直接指定寬高**——`imageSize` 是**總像素預算**、`aspectRatio` 是形狀，上游取「符合該比例、且總像素最接近預算」的一組寬高，而且**兩邊都對齊到 16 的倍數**。1K 的預算約 105 萬像素、2K 是 4 倍、4K 是 16 倍。
  - 以 `gemini-3.1-flash-lite-image` 在 1K 下實測全部 10 種比例（都有量實際輸出）：1:1 → 1024×1024（1,048,576）、4:3 → 1200×896（1,075,200）、3:2 → 1264×848（1,071,872）、5:4 → 1152×928（1,069,056）、16:9 → 1376×768（1,056,768）、21:9 → 1584×672（1,064,448），直式為對應的轉置。總像素都落在 105～108 萬之間，印證「預算 + 對齊」的模型。
  - **「4K」是像素預算而不是 UHD 解析度**——4K + 16:9 實測是 5504×3072（約 1,690 萬像素），不是 3840×2160。這個 API **沒有直接指定寬高的方式**，要精確尺寸只能自己換算比例；這點跟萬相 2.7 的「方式二（明確寬高像素值）」很不一樣，是這兩家最大的差異。
  - **比例從 5 種補到 10 種**：先前只開放 1:1／16:9／9:16／4:3／3:4，實測確認四個型號都支援 Google 文件列出的全部 10 種，補上 3:2／2:3／5:4／4:5／21:9（8 個 MODELS 條目都更新）。最偏門的 21:9 在四個型號上都個別驗過；`gemini-2.5-flash-image` 的量化略有不同，21:9 給的是 1536×672 而不是 1584×672，但比例有照做。
  - 非法值（`99:1`、`banana`、`8K`）一律回 400、**不會被靜默忽略**，但錯誤訊息是通用的 `Request contains an invalid argument.`，看不出是哪個欄位有問題——所以沒辦法像萬相那樣用「非法哨兵參數」免費列舉，這輪的比例列舉是實際產圖驗的（用最便宜的 flash-lite，共 13 張）。
  - 前端不用改：比例選單是從 `/api/models` 動態產生的，加值即生效。

- feat：Gemini 文字模型（7 個）的路由改走 Gemini 原生 API，`/v1beta/models/{model}:generateContent`（串流 `:streamGenerateContent?alt=sse`）（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **這不只是換端點，是換回兩個先前做不到的能力。** 走 OpenAI 相容端點時 Gemini 的思考過程既看不到也關不掉——`README.md` 與 MODELS 的註解一度據此寫「Gemini 3.x 系列無條件思考、關不掉」、七個模型全部標 `thinking: False`。原生端點實測兩件事都做得到：`thinkingConfig.includeThoughts` 會回傳帶 `"thought": true` 的 content part（思考全文，串流時也逐段送），`thinkingConfig.thinkingBudget: 0` 能真的關掉思考。**同一題關閉思考後 completion token 從 170～210 降到 1**——這是實際的費用差距，先前使用者完全沒有辦法省下來。
  - 新增 `_GEMINI_NATIVE_TEXT_MODELS` 分流與 `_build_gemini_body()` / `_gemini_text_generate()` / `_gemini_text_stream()`。請求轉換：`system_prompt` → 頂層 `systemInstruction`、`max_tokens` → `generationConfig.maxOutputTokens`、`stop` → `stopSequences`、`top_p`/`top_k` → `topP`/`topK`、`presence/frequency_penalty` → `presencePenalty`/`frequencyPenalty`；回應的 `thought` part → 前端既有的 `reasoning` 事件，一般 part → `content`。前端不用改。
  - **支援度各型號不同，送錯直接 400**，所以有兩個例外集合（逐一實測）：
    - `gemini-2.5-pro` 不接受 `thinkingBudget: 0`（`The model does not support setting thinking_budget to 0`）→ 思考關不掉，`thinking` 維持 `False` 不給開關，但改成一律送 `includeThoughts` 把過程顯示出來（先前是白花 token 又看不到）
    - `gemini-2.5-flash-lite` 不接受 `includeThoughts`（`Thinking_config.include_thoughts is not supported`）→ 有開關但不顯示思考區塊
  - **`gemini-2.5-flash-lite` 還有一個差點漏掉的陷阱**：它的思考**預設是關的**（不帶 `thinkingConfig` 時 `thoughtsTokenCount` 是 `None`）。第一版實作對它在「思考開」時什麼都不送，結果開/關都是 1 個 token——那個開關等於沒有作用。實測後改成送 `thinkingBudget: -1`（動態預算）才真的啟動思考（124 vs 1 個 token）。
  - **計費**：`thoughtsTokenCount` 也是實際收費的輸出 token，`usage.completion_tokens` 算成 `candidatesTokenCount + thoughtsTokenCount`，否則 header 的「本次花費」會嚴重低估（思考往往佔了絕大部分）。
  - 端到端實測：7 個模型 × 非串流思考開/關 × 串流,全部通過,行為與上表一致。

## 2026-08-12

- feat：**建立模型測試的標準工作流**（優化 4～6），把這輪反覆手工做的事固化下來。
  - **`scripts/probe_model.py`** — 取代每次重寫的一次性探測腳本。功能：清單存在性、計價方式（按次／按 token）、**權限探測**（送必定違規的參數，用 `AccessDenied` vs `InvalidParameter` 區分「配額未開」與「參數問題」——配額沒開就不必往下測）、尺寸逐一驗證、以及 `--drift` 清單漂移比對。
    - 腳本開頭寫明四條設計原則，都是踩過坑換來的：能不花錢就不花錢；**「送得出去」不等於「能用」**（閘道驗證比原廠寬鬆）；**探測手法不能跨家族沿用**（哨兵模式預設關閉，因為千問是 `n` 先驗、MAI/Seedream 根本不拒絕哨兵會真的產圖）；權限與參數問題要分開判斷。
    - 實測驗證：`wan3.0-video` → 正確報 `AccessDenied`；`qwen-image-3.0` → 正確報參數錯誤（代表權限已開）；`--drift` 正確找出 2 個「我們有、群組看不到」與 13 個「閘道有、我們沒收錄」。
  - **`tests/test_pure_functions.py`** — 22 個測試，**每一條都對應一個實際發生過的 bug**，不是為了覆蓋率：`_openai_usage` 的兩種 token 帳法、`_public_base_url` 不可用內部 http、`_apply_res_and_duration` 三種形式都要送、`_image_usage` 兩種欄位命名、`_EDIT_MAX_REF`（含萬相 2.7 是 9 張那條——曾誤依 Go struct 設成 2）、Gemini 兩個思考例外集合的區別、以及 proxy 白名單要擋偽裝後綴。全數通過。
    - `pytest` 未列入 `requirements.txt`（這個 repo 原本沒有測試套件），需要時 `venv/bin/python -m pip install pytest`。註：`venv/bin/pip` 的 shebang 指向舊路徑已損壞，要用 `python -m pip`。
  - **`CLAUDE.md` 新增「新增／測試模型的標準工作流」** 六步：確認網關 → 跑探測腳本 → **對照原廠文件逐一驗證所有可帶的參數** → 閘道有缺失就回報給 `nen-ai-platform` session（找不到 session 要通知使用者）→ 更新 MODELS 與文件（只寫驗證過的值）→ 補測試 → 通知文檔 session。
    - 第 2 步特別強調「不要只驗自己想用的那幾個參數」，並列出這個平台反覆踩到的三種靜默失敗：閘道沒映射、上游不讀該欄位、欄位放錯層。
    - 文檔通知那節補上「**完整的呼叫方法**」與「**分三層寫**」（實測數據／我的推論／轉述），並記下兩條與文檔站往來的教訓：我方的推論曾經錯誤流入文檔；對方的更正也曾用了測不到目標的方法。

- feat：**進行中的任務會被持久化，重新整理後自動恢復**。先前 `task_id` 只存在記憶體，使用者一重新整理（或分頁被瀏覽器回收）輪詢就永久停止——**任務照樣跑完、照樣計費，但結果再也拿不到**。一支 30 秒的 Seedance 2.5 是 $6.94，這是實打實的損失。
  - 新增 `savePendingTask` / `clearPendingTask` / `resumePendingTasks`，影片與 MuleAI 兩條任務流都接上，**7 個結束出口**（完成／失敗／逾時 × 2 條）都會清除紀錄，避免殘留。
  - 恢復時會把 `costInfo` 一併帶回，所以恢復的任務完成後仍能正確計費。超過 1 小時的視為過期不再嘗試，最多保留 30 筆。
  - 健壯性：`localStorage` 內容損壞回空陣列、寫入失敗（配額滿）吞下例外不影響當次生成——這兩個都驗過。

- feat：**昂貴任務送出前確認**。影片單次最貴到 $6.94，而非同步任務一旦送出就無法取消。超過 $1 時彈出確認並說明換算依據（模型／解析度／秒數），讓使用者能判斷估算是否合理。
  - **算不出價格時不攔**——寧可放行也不要用猜測的數字嚇阻使用者。驗證過五種情境：Seedance 2.5 30 秒（$6.94，攔）、4 秒（$0.42，放行）、萬相按次（$0.1，放行）、Veo（算不出來，放行）。

- fix：**9 個按 token 計費的圖片模型，花費完全沒被計入**（MAI 三個、GPT Image 兩個、Gemini Image 四個）。跟先前影片那筆是同一個病灶：`addFixedCost()` 只認 `type==='fixed'`。
  - 解法比影片那次更可靠：**直接用上游回報的 token 數**，不必在前端維護「解析度→token」的換算表，也不會因為上游改了算法而失準。
  - 兩條路徑的欄位名不同，都實測過：OpenAI 相容端點回 `num_input_text_tokens`／`num_input_image_tokens`／`num_output_tokens`；Gemini 原生端點回 `promptTokenCount`／`candidatesTokenCount`。新增 `_image_usage()` 正規化，Gemini 那條還要**跨多次呼叫累加**（n>1 是並發打 n 次，而且重試那幾次也照樣消耗 token）。
  - 前端新增 `addImageCost()`：按次的用單價×張數、按 token 的用實際 usage、上游沒回 usage 就不計。
  - 端到端驗證三條路徑：MAI `{9, 1024}`、Gemini `{8, 1120}`、按次計費的 z-image 維持 `None`。

## 2026-08-11

- fix：**`/api/proxy/fetch` 的白名單漏了火山引擎（Seedance 的產出網域）**，AI Canvas 把 Seedance 的影片接成下一步輸入會被擋。
  - 全平台健檢時發現：白名單只有 `.aliyuncs.com`／`.amazonaws.com`／`storage.googleapis.com`，但 Seedance 系列的產出網址是 `ark-acg-ap-southeast-1.tos-ap-southeast-1.volces.com`。實測代理對它回 **400 `URL host not allowed`**，對萬相的 aliyuncs 網址則放行。
  - 平常不會踩到——後端會先把產出下載回本機再給前端，Canvas 拿到的是同源的 `/outputs/...`。但**下載失敗時會退回原始網址**（`video_status` 的 `local if local else video_url`），這時 Canvas 的「把上一步結果當下一步輸入」就會走到這支代理而失敗。
  - 已補上 `.volces.com`，並在註解裡寫清楚這份白名單要涵蓋的兩類網域（我們自己的雲端儲存 ＋ 上游直接回傳產出網址的網域）。驗證過偽裝後綴（`volces.com.evil.com`）仍會被擋。
  - 這正是 `CLAUDE.md` 早就警告過的那類問題——先前就因為只寫了 OSS 的網域，正式環境改用 GCS 之後 Canvas 的接續功能全部失效。**每次接入新廠商，都要檢查它的產出網域在不在白名單裡。**

- test：AI Canvas 全功能健檢（正式站，用 Playwright ＋ 直接打端點）。
  - 頁面載入**零 console 錯誤／警告**，10 個自訂節點（text／camera_angle／load_image／image／video／video_edit／video_animate／edit／audio／muleai）全部註冊成功，canvas 元素正常。
  - canvas.js **沒有寫死任何模型 id**，全部從 `/api/models` 動態產生——所以這一輪新增的 14 個模型不需要另外改 canvas。
  - 端點比對：canvas 呼叫的 9 個端點在後端都存在（影片的 t2v/i2v/r2v 是動態組路徑，不是寫死字串）。
  - 實測通過：文字生成、圖片→文字（analyze_image）、TTS、文生圖、圖像編輯（**兩張參考圖都生效**，驗證了本輪的多圖修正在 canvas 路徑上也成立）。
  - TTS 音色抽測：5 個模型的清單各抽 3 個音色，全部可用（沒有重演 MAI 尺寸那種「列了不能用的值」）。
  - **已知阻塞**：影片編輯與動作動畫在正式站仍會失敗，因為 `b598e7c`（沒有雲端儲存時改用本站公開路徑）尚未部署。正式站目前在 `1c002c0`。

- docs：釐清「`dreamina-seedance-2.0-fast` 是否支援 1080P」的觀察衝突——**結論維持原判：不支援**，現有的 `resolutions: ["480P","720P"]` 設定正確、不需更動。
  - 衝突起因：文檔站測到送 1080P「沒有被拒、任務跑到 SUCCESS」，與我們的「回 400 InvalidParameter」相反。
  - 用控制變因重測後，**兩邊的觀察都是真的、只是測到不同的東西**：只送頂層 `size=1080P` → 200（但輸出仍是 720p）；只送 `metadata.resolution=1080p` → **400**；兩者都送（本平台的作法）→ **400**。
  - 因為 **doubao 根本不讀頂層 `size`**，文檔站那次的 1080p 從未送達上游，任務會成功只是上游用了預設的 720p。`metadata.resolution` 才是唯一會被讀取的來源。
  - 教訓已寫進 `README.md`：**比對兩份互相矛盾的實測結果時，要先確認雙方送出的請求是否等價**，否則會把「測法不同」誤判成「事實不同」。
  - 探測代價：三種送法裡「只送 size」那次回 200、建立了一個會計費的任務——我原以為三個都會被拒。

- fix：**影片編輯／動作動畫在沒有雲端儲存時也能用了**（使用者實際在 AI Canvas 上撞到「沒有可用的儲存後端」）。
  - 起因是同日那筆修正：影片不能用 base64 送，必須給可下載的網址；但正式環境沒設定雲端物件儲存，所以功能直接回錯誤。
  - **關鍵發現**：本站的 `/outputs` 是不需驗證就能存取的靜態掛載（實測正式站不帶 `Authorization` 也回 200）。所以沒有雲端儲存時，可以把上傳的檔案寫進 `outputs/videos/`，再給出 `https://<本站網域>/outputs/videos/<name>` 讓模型自己來抓——**不必依賴外部儲存也能運作**。
  - 新增 `_public_base_url()` 推導對外網址：優先讀 `PUBLIC_BASE_URL` 環境變數，否則從 `X-Forwarded-Host` / `Host` 推導；**沒有 `X-Forwarded-Proto` 時預設 `https`**（不能用 `request.url.scheme`，那在 LB 後面是內部的 http，給出 `http://` 的網址有機會被上游拒絕）；推導出 localhost／127.0.0.1 時回 `None`，因為那種網址外部抓不到，寧可報錯也不要送出註定失敗的任務。
  - 影片與音訊共用這套 fallback，四個影片呼叫點與三個音訊呼叫點都接上 `request`。
  - ⚠️ **這條 fallback 有兩個先天限制**，已寫進註解：Cloud Run 每個實例檔案系統獨立（`maxScale` > 1 時模型可能被路由到沒有該檔案的實例）、容器重啟後檔案消失。模型通常幾秒內就抓走，實務上多半可行，但要根治仍是設定雲端儲存。
  - 推導邏輯用 7 種情境驗證過（LB 轉發、只有 host、本機、localhost、無標頭、多值 host、`PUBLIC_BASE_URL` 覆寫）。

- fix：補上上一筆文案清查**漏掉的一處**——`_upload_video_for_url` / `_upload_audio_for_url` 的錯誤訊息含「上游」，而且正是使用者截圖上看到的那句。
  - 漏掉的原因是我上次的掃描只比對 `"error": f"..."` 這種字面形式，而這兩句是以 `return None, ("...")` 的 tuple 形式回傳的。
  - 這次把掃描規則補齊：MODELS 文案、`"error"`/`"detail"`/`"message"` 字面、**`return None, (...)` 形式**、前端的 `data-tip` 與 `toast()`，四類一起掃。**寫檢查時要涵蓋所有「值最終會流到使用者眼前」的路徑，而不只是最常見的那一種寫法。**

- fix：**清掉所有面向客戶的介面上的內部用詞**（使用者指出 Seedance 2.5 的說明寫著「較 2.0 貴約 53%」）。這個平台是面向客戶的服務，那類措辭是工程內部的講法。
  - 全面清查後改了 **16 處 `MODELS` 文案 ＋ 1 處 tooltip ＋ 2 處後端錯誤訊息**，不只使用者圈出的那一句：
    - **價格比較**：「較 2.0 貴約 53%」「1K/2K 同價」「2K 輸出計價較高」「費用減半」——單價由 `/api/pricing` 自動顯示就夠了，文案再去比較等於替客戶做選擇
    - **內部術語**：「上游僅支援 480P/720P」「上游無開關」「上游只讀取首幀圖片…會被靜默丟棄」「上游未回傳音訊」——客戶不需要知道系統內部有幾層
    - **自家型號互比**：「唯一可關閉配音的萬相型號」
    - **抱怨式的限制描述**：「思考過程無法關閉」「不顯示思考過程」「推理無法關閉」——改成正面描述能力，做不到的事就不提
  - 真正需要讓使用者知道的限制（「僅接受圖片」「最多 9 張」「僅首幀」）保留，但用中性語氣。
  - `CLAUDE.md` 新增一節「使用者看得到的文案：不要寫內部觀點」，列出四類不該出現的寫法，並明確區分：工程細節／驗證狀態／上游怪癖／價格比較寫在 `README.md`、`update.md`、`memory.md` 與程式碼註解裡（那些給開發者看，越詳細越好），**兩者不要混**。
  - 第一次掃描用關鍵字清單漏了「唯一可關閉配音的萬相型號」，補了「自家型號互比」這類規則後重掃才抓到——單靠關鍵字比對不夠，要連「這句話是站在誰的視角寫的」一起看。

- feat：新增 `grok-4.3`（正式環境，使用者指定）。它是**唯一有完整推理強度分段、而且支援看圖的 Grok**。
  - `reasoning_effort` 枚舉實測為 `none`／`minimal`／`low`／`medium`／`high`（`xhigh` 與 `max` 回 422）。送非法值只回通用的 `openai_error` 問不出枚舉，是逐一試出來的。`none` 連跑三次都得到 `reasoning_tokens=0`，是穩定有效的（對比 `grok-4-1-fast-reasoning` 的 `none` 完全沒作用）。
  - 圖片輸入實測可用，標了 `vision`，沿用先前為千問 VL 做的圖片上傳 UI。
  - 一樣不回傳 `reasoning_content`，思考過程看不到，`thinking` 維持 `False`。

- fix：**Grok 的推理 token 沒有被計入花費**（peer 先前提醒、這次實測確認）。
  - 實測 `grok-4.3`：`prompt 31 + completion 1 + reasoning 844 = total 876`——**推理 token 不在 `completion_tokens` 裡**，但那些 token 照樣收費。我們的 `addTokenTextCost()` 用的正是 `completion_tokens`，所以一次花 844 個推理 token 的呼叫會被算成 1 個，「本次花費」嚴重低估。
  - 新增 `_openai_usage()`，以 **`total - prompt` 反推 completion**，非串流與串流兩處都改用。這個算法對兩種帳法都正確：Grok 這種「推理不含在 completion」的會補回來，而 DeepSeek V4／GLM／Seed 2.0 這種「推理已含在 completion」的維持原值（若直接把 reasoning 相加會變成兩倍）。沒有 `total` 時退回原值。
  - 單元驗證三種情況：Grok `(31,1,876)` → 845 ✅、GLM `(26,139,165)` → 139 ✅、無 total `(10,20,0)` → 20 ✅。端到端也確認 `usage.completion_tokens` 從 1 變成 540。

- fix：**8 個按 token 計費的影片模型,先前完全沒被計入「本次花費」,而且價格顯示對使用者無意義。**
  - 盤點 27 個影片模型後發現：19 個是 `quota_type=1`（按次固定價），但 **8 個是 `quota_type=0`（按 token）**——Seedance 全系列、`wan3.0-video`、兩個 Veo、`gemini-omni-flash-preview`。
  - 兩個問題：（1）前端把它們顯示成「$10.7→$10.7/1M」，**沒人能心算一支影片是幾個 token**；（2）`addFixedCost()` 只計 `type==='fixed'`，所以這些模型的花費**完全不會累加**到 header 的「本次花費」。
  - 新增 `estimateVideoTokenCost()` 用公式換算：`tokens = 寬 × 高 × 幀數 / 1024`，**幀數 = 要求秒數 × 24 + 1**（fps 固定 24；我端到端實測 480P/4 秒得到 854×480、97 幀，代入 = 38,830.31，與上游回傳的 38,830 吻合）。價格提示改成顯示「約 $X/次（解析度 秒數）」，切換解析度或拖時長會即時重算；新增 `addVideoCost()` 統一處理按次與按 token 兩種，讓花費徽章不再漏算。
  - **兩個容易算錯的地方，都有實測依據**：
    - **480p 的尺寸各世代不同**：Seedance 2.5 是 854×480、2.0 系列與 1.5-pro 是 864×496。共用尺寸表會差 4.5%。
    - **1080p 與 4K 有解析度倍率**（1080p ×1.1、4K ×4/7，基準 720p）。純用像素數算會低估 1080p 10%、高估 4K 75%。我用 peer 的精確總價表反推驗證：純公式 ÷ 精確表 = 1.0997 與 0.5715，與 `constants.go` 的 1.1 和 4/7 吻合。
  - 換算結果與 peer 的精確總價表**逐筆比對 10 組全部吻合**（2.5 的 480P/720P × 4/10/30 秒、2.0 的 480P/720P/1080P/4K）。
  - **算不出來就不計**（沒有該模型的尺寸表時回 `null`）——寧可少算也不要顯示一個編出來的數字。所以 `wan3.0-video`、Veo、`gemini-omni` 目前仍不計入，那些沒有可靠的 token 公式。

- fix：影片失敗時讀不到真正的錯誤原因。`video_status` 只讀 `error.message`，但 **doubao（Seedance）把失敗原因放在頂層的 `fail_reason`**，所以 Seedance 失敗一律顯示「Unknown error」。已補上 `fail_reason` 的讀取。
  - 這在實務上很重要：peer 回報 Seedance 會偶發觸發**版權過濾**（`The request failed because the output audio may be related to copyright restrictions.`），而且是**模型自動配樂**觸發的、跟畫面內容無關——沒有這段訊息幾乎不可能猜到原因（他們踩到的提示詞是「Mist over a lake」）。想避開可以傳 `generate_audio: false`。
  - 順帶確認我們的輪詢**沒有踩到另一個陷阱**：peer 提醒 doubao 的成功與失敗 `progress` 都是 `"100%"`，只看 progress 會把失敗誤判成成功。我們的 `pollVideo()` 判斷的是 `status`（`FAILURE` 也在失敗清單裡），不受影響。

- feat：新增 `dreamina-seedance-2.5`（文生影片）。**網關：測試網關 `192.168.0.245`**（使用者指定並提供金鑰）——正式環境的 `/v1/models` 清單裡雖然有它，但網關端說程式碼尚未部署，那邊叫不動。這又是一個「清單有它 ≠ 通道可用」的例子。
  - 與 2.0 系列差異大到 UI 必須分開處理：**解析度只有 480p/720p**（2.0 到 4K）、時長 `[4,30]` 或 `-1`（2.0 是 2~15）、參考素材上限高很多（30 張圖／10 支影片／10 段音訊）、**支援純音訊輸入**、每秒單價 720p $0.2311 **比 2.0 貴約 53%**。
  - 約束逐一實測（不合法的組合被拒不會產影片，所以探測是免費的）：`1080p`／`4k` 回 `InvalidParameter`；`duration` 送 3 或 31 都被拒；`camera_fixed`／`frames`／`draft` 都回 `InvalidParameter`。MODELS 以 `resolutions: ["480P","720P"]` 與 `min_dur: 4` / `max_dur: 30` 限制住。
  - **一項與網關端規格不符**：他們說 `seed` 也不支援，但實測**沒有被拒**、任務照樣建立（那筆會計費，這是探測法的代價）。目前以實測為準，沒有特別擋，已回報給他們。
  - **只上架 t2v，沒有加 i2v/r2v/vedit**：那些要帶入參考影片，而影片輸入必須是雲端網址（見同日的影片修正），在雲端物件儲存設定好之前做了也不能用；另外 2.5 專屬的 `omni_reference_task_type` 在測試網關上也還沒透傳。兩件完成後再補。
  - 計費公式（網關端反推並驗證）：`tokens = 寬 × 高 × 幀數 / 1024`，fps 固定 24；回傳的 `duration` 無條件捨去但**計費按真實幀數**（要求 4 秒實際收 4.04 秒）。
  - **端到端實測通過**（透過我們的後端打測試網關）：送 480P / 4 秒 / 16:9，實際產出 **854×480、97 幀、4.041667 秒**。代入公式 `854×480×97/1024 = 38,830.31`，與網關端實測上游回傳的 `38,830` 完全吻合——同時也印證了「計費按真實幀數而非回傳的整數秒」這件事。我們既有的參數傳遞（`metadata.resolution` + 頂層 `seconds`）對 2.5 一樣有效，不需另外處理。

- fix：**影片輸入改走雲端網址**——上游不接受 base64 data URI 的影片（由文檔站 session 回報、我實測複驗）。
  - 實測（正式環境，`wan2.2-animate-move`）：送 `data:video/mp4;base64,...` **提交回 200**，但輪詢到最後是 `failed`，錯誤 `InvalidVideo.FileFormat: Invalid video type. Only mp4/mov/avi is supported.`。只看提交結果會誤以為成功。
  - 這代表**視頻編輯與動作動畫從來沒有成功過**。我先前為 animate 補 `images[]` 只是讓它跨過第一個錯誤（`Field required: input.media`），撞上第二個——而我當時沒有端到端驗證，所以沒發現。
  - 新增 `_upload_video_for_url()`（比照既有的 `_upload_audio_for_url()`），四個送影片的地方都改用：`video_animate` 的參考影片、`video_vedit` 的來源影片、`video_i2v` 影片延伸的起始片段、`video_r2v` 帶入的參考影片。
  - **⚠️ 但這需要雲端物件儲存才會生效，而實測正式環境目前沒有設定**（產出圖片的 `local_path` 是 `/outputs/images/...` 本機路徑而非簽名網址）。設定好之前這些功能會回明確的 400 而不是靜默失敗，但仍然不能用。
  - 順帶暴露兩個既有風險：Cloud Run 每個實例檔案系統獨立且 `maxScale: 5`，產出圖片存在某實例、下次被路由到別的實例就會 404（實測當下只有單一實例，問題尚未浮現）；容器重啟後產出全部消失。已寫進 `README.md` 與 `memory.md`。

- chore：確認 `gemini-omni-flash-preview` 在我們這邊是**正常的**，不需處理。文檔站測 `/v1/video/generations`（404）與 `/v1/chat/completions`（400）都不通，但那兩條本來就不是給它用的——錯誤訊息自己說了「only supported in the Interactions API」。我們走的 `/v1beta/interactions` 實測回 200 completed。這是一個「別人測到不通、但測錯路徑」的例子，值得在採信他人回報前先確認對方走的路徑跟我們一樣。

- fix/docs：`qwen3.5-omni-plus-realtime` **決定不加**，並修正 `/ws/omni` 代理寫死的錯誤路徑。
  - 使用者要求加這個模型，我在動手建 UI 前先驗端點，結果打不通——這個順序是對的，不然會做出一個死功能。
  - 實測（正式環境）：`/api-ws/v1/realtime`（我們程式碼寫死的）回 **404**；`/v1/realtime` 則是**握手成功（HTTP 101）後立刻斷線、連 close frame 都沒有**。兩個 realtime 型號行為一致，包括程式碼裡當預設值的 `flash-realtime`。
  - 閘道端查證原始碼給出根因：（1）網關的 WebSocket 路由只有 `/v1/realtime` 與 `/v1/responses`，**`/api-ws/v1/realtime` 從來不是對外路徑**——那是 DashScope 上游給 TTS/ASR 用的內部路徑，寫進我們程式碼應該是誤抄；（2）realtime 中繼**只有 OpenAI 系 adaptor 有實作**，`qwen3.5-omni-*-realtime` 屬阿里渠道，阿里 adaptor 的 WebSocket 只涵蓋 TTS 與 ASR，realtime 會掉進一般 HTTP 路徑導致型別斷言失敗而 panic——正好對應「101 後無聲斷線」。
  - 所以：**模型出現在 `/v1/models` 清單裡，不代表 realtime 通道可用**。已把這條寫進 `memory.md`（新增模型時先驗端點再做 UI，不要反過來）。
  - 把 `/ws/omni` 的目標路徑改成正確的 `/v1/realtime`，並改成從 `NENAI_BASE` 推導而不是寫死 `nen.com.tw`（原本連測試網關都打不到）。代理保留著，等阿里 adaptor 補上 realtime 就能直接用，不必重寫。
  - 也因此確認：前端 realtime UI 在 `8c012ac` 之後被移除是**正確的決定**，那個功能當時就是壞的。要恢復需要閘道端先補完整套 realtime 支援，那是獨立的功能開發，已轉給他們的使用者決定。
  - 順帶把 Grok 的行為歸因寫進 `README.md`：`reasoning_effort` 在四個型號上表現不一、以及都拿不到 `reasoning_content`，**都是上游 xAI 的行為，不是網關的映射問題**（網關對 `grok-4-*` 原樣轉發、不碰該欄位；`reasoning_content` 兩種欄位名都會解析）。錯誤訊息的枚舉也是上游回的，網關只轉發——所以「補上合法枚舉」這件事網關做不到，硬編一份清單反而會隨上游更新而過期。

- feat：新增 8 個模型——`claude-opus-5`、`gemini-3.5-flash-lite`、4 個 xAI Grok、2 個千問 VL，並為視覺語言模型做了前端的圖片輸入 UI。網關：正式環境（使用者指定）。
  - **xAI Grok 是新廠商，四個型號行為各不相同**（逐一實測）：`-reasoning` 版預設就推理、`-non-reasoning` 版完全不推理（`reasoning_tokens` 恆為 0）；四個都**不回傳 `reasoning_content`**，思考過程一律看不到，所以 `thinking` 全部 `False`。`reasoning_effort` 只有 `grok-4-20-reasoning` 有效（`none` → 0），`grok-4-1-fast-reasoning` 送了無效（各 3 次中位數 176 vs 245，沒有下降），兩個 `-non-reasoning` 版直接回 400。送非法值只回通用的 `openai_error`、問不出合法枚舉，所以 `reasoning_efforts` 只列實測有效的 `none`。
  - **`gemini-3.5-flash-lite` 的思考預設是關的**，跟 `gemini-2.5-flash-lite` 一樣要送 `thinkingBudget: -1` 才會啟動（不帶 thinkingConfig 時 `thoughtsTokenCount` 是 `None`）；但它**接受 `includeThoughts`**（2.5 版送了會 400），所以思考過程看得到。原本的判斷式是「在 NO_INCLUDE_THOUGHTS 就送 budget=-1、否則送 includeThoughts」二選一，無法表達這個組合，改成新增 `_GEMINI_THINKING_OFF_BY_DEFAULT` 集合、兩個條件獨立判斷。
  - **千問 VL 的圖片輸入**：`TextGenerateRequest` 新增 `images` 欄位（data URI 或網址），有圖時把 `content` 從字串改成陣列、用標準 OpenAI `image_url` 格式帶入。前端在文字頁籤新增圖片上傳欄位，只有 MODELS 標了 `vision` 的模型才顯示；**切換到不支援視覺的模型時會清掉已選的圖**，否則會靜默夾帶到不吃圖的模型上（那些模型會直接 400）。圖片只附在當下這一輪，不進對話歷史——上游對「歷史訊息裡的圖片」的行為未驗證，不主動送。
  - 端到端實測（透過本機服務打正式環境）：8 個模型都正常；`grok-4-20-reasoning` 的 `effort=none` 生效；`gemini-3.5-flash-lite` 思考 ON 212 tokens／OFF 1 token；兩個 VL model 非串流與串流都正確辨識出圖片顏色。
  - 前端改動：`app.js?v=60` → `?v=61`。
  - **`qwen3.5-omni-plus-realtime` 未加**——我原本說「已經有 WebSocket 代理，加進清單就能用」是**錯的**：`index.html` 裡完全沒有 omni 的 UI（`omniModel` 等 10 個 DOM 元素都不存在），後端 `/ws/omni` 代理與 `app.js` 的 handler 雖然都在，但沒有任何入口能到達。查了歷史，那段 UI 在 `8c012ac` 時存在、之後被移除，移除原因不明。要加這個模型等於要重建整個 realtime UI，已回報使用者決定。

- fix：修掉 `qwen-image-3.0` 在 `n > 1` 時**只顯示一張產出**的問題（由文檔站 session 交叉查核時發現）。
  - 千問 3.0 走的 multimodal-generation adaptor 在 `n>1` 時，OpenAI 相容的 `data[]` **只回第一張**，但 `usage.output_image_count` 是實際張數、也照那個張數計費——其餘的圖在 `metadata.output.choices[].message.content` 裡。實測 `n=2`：`data[]` 1 筆、metadata 2 張、計費 2 張。
  - 我們的取圖只讀 `data[]`，所以使用者選 n=2 只看得到一張。新增 `_extract_images_from_metadata()`，在 `/api/image/generate` 與 `/api/image/edit` 兩處補齊（以 URL 去重，不會重複計入 `data[]` 已有的）。
  - 端到端驗證：修正後 `n=2` 正確回傳 2 張；萬相 `n=2` 維持原本的 2 張，未受影響（它走另一條 adaptor，`data[]` 本來就完整）。
  - 根本解需要網關端把 `data[]` 補齊，已回報，他們已修（`AliOutput.ChoicesToOpenAIImageDate` 原本「每個 choice 組一筆 ImageData」、內層迴圈把 `data.Url` 覆蓋掉，改成每張圖產出一筆並跨 choice 攤平），待部署。
  - **⚠️ 更正一個我判斷反了的結論**：我原本寫「使用者付了 2 張的錢卻只拿到 1 張」，**不成立**。網關是數 `data[]` 裡的實際圖片數計費（`aliImageHandler` 的 `actualImageCount`），不是看 `usage.output_image_count`——所以使用者只被收一張、沒有被多收，真正吃虧的是**平台方**（上游按 2 張收、平台只收使用者 1 張）。這是網關端查用量日誌釐清的（`quota 15000` = $0.03 × QuotaPerUnit＝一張的價錢）。我把上游回報的產出張數當成了計費依據，這個錯誤已經流進文檔站，已請他們一併更正。
  - 附帶影響：網關修好部署之前，我們補齊後顯示 2 張但實際只被收 1 張，header 的「本次花費」在這個情境會**高估**；部署後兩邊就會一致。

- docs：更正我先前對千問 3.0 編輯輸出尺寸的錯誤描述。我寫「送任何 size 都得到 2048×2048」，**實際上輸出尺寸是跟著輸入圖的長寬比走**——我會量到固定值只是因為測試用的輸入圖剛好是正方形。文檔站用 900×506（16:9）複驗，兩種 size 都得到 2720×1520（≈16:9）。`README.md` 已改成「以回應的 `usage.output_width`/`output_height` 為準，不要假設是固定值」。`size` 被忽略這點則複驗成立。
  - 這是本輪第二次「用單一樣本推出通則」的錯誤（前一次是萬相參考圖上限）。`memory.md` 的教訓清單已涵蓋這一類。

- chore：與網關 `/v1/models` 交叉比對後確認，`MODELS` 裡有 **3 個模型在目前這個 key 的群組看不到**：`veo-3.1-generate-001`、`veo-3.1-lite-generate-001`、`wan3.0-video`。`/v1/models` 在 new-api 是依群組回傳的，所以可能只是群組權限差異而非全平台沒有；**尚未從清單移除**，待確認是權限問題還是真的沒上架。反向則有 23 個平台有、我們沒列的模型（xAI Grok 4 個、`qwen3-vl-plus`/`-flash`、`gemini-embedding-*` 等），要不要補進測試平台待決定。

- docs：新增專案根目錄的 **`memory.md`**，記錄「測試平台（本專案）／nen 閘道 `nen-ai-platform`／文檔站 `Nen-AI-Docs-V1`」三個 repo 的關係。`CLAUDE.md` 開頭加了指引把人導過去。
  - 內容包含：三個專案各自的路徑／遠端／分支／正式站與角色定位、資料流、**職責邊界**（不改別人的 repo、跨專案用 `ListAgents` + `SendMessage` 溝通）、基礎架構實查結果（兩者都在 us-east5、LB 都是 PREMIUM 全球 anycast、平台實測只多 0.8 秒）。
  - 最重要的是「**反覆踩過的坑**」一節，每一條都是實際造成過錯誤判斷的：通過閘道驗證不等於真的能用（閘道驗證比原廠寬鬆）、不要從閘道的 Go struct 推斷行為（萬相參考圖上限那次）、探測手法不能跨模型家族沿用（`n` 哨兵在千問失效、在 MAI/Seedream 會真的產圖）、官方文件會過時會寫錯、「參數送了沒作用」幾乎都是閘道沒映射（要找同模型不同路徑的對照組）、UI 不顯示沒有作用的控制項。
  - 末段記錄進行中的跨 session 事項：閘道端 `glm-5.2` 的 thinking 修正已推但未部署且需要後台開關、`wan3.0-video` 卡在 `AccessDenied` 從未成功呼叫過、`glm-5.2-us` 等上架不需改程式碼。

- process：`CLAUDE.md` 新增一條流程規範——**每次把新模型加進 `MODELS` 並驗證完之後，都要通知文檔專案的 session 撰寫對應文檔**（`/Users/levi/claude_code/nen_ai_project/Nen-AI-Docs-V1`）。用 `ListAgents` 找那個 session（名稱通常是 `nen-ai-docs-v1-*`），找到就 `SendMessage` 把模型資訊送過去；**找不到就明確提醒使用者去啟動它**，不要默默略過、也不要自己跑去那個 repo 改檔案。
  - 規範裡明訂要送的內容：模型 id／名稱／分類／支援的 type、**實測過的**參數約束、計價（含 UI 顯示不到的加價規則）、上游端點與請求格式的特殊之處，以及**哪些是實測驗證過、哪些只是照文件或推斷的**——最後這點特別要求寫清楚，因為文檔讀者無法自行分辨，而這個專案已經多次因為把推斷當成事實而出錯（萬相編輯參考圖上限、MAI 尺寸、Gemini 思考能否關閉都是例子）。
  - 已立即套用到剛新增的 `qwen-image-3.0` / `-pro`，資訊已送給 `nen-ai-docs-v1-01`。

- feat：新增 `qwen-image-3.0` 與 `qwen-image-3.0-pro`（文生圖 + 圖像編輯各兩個條目）（commit 待補）。網關：正式環境 `https://nen.com.tw`（使用者指定），已從 `/v1/models` 確認兩個模型都存在。
  - **尺寸是面積約束、不是固定枚舉**：總像素 262,144（512×512）～ 6,553,600（2560×2560），格式必須 `寬*高`。上下界都實測驗過（`511*511`、`2561*2561` 被拒，`1024*1024`／`1280*720`／`2048*2048` 能產圖）。錯誤訊息寫「for **t2i** requests」——實測 i2i 送 `size=10*10` 照樣成功，i2i 不套用這條。
  - **`1K`/`2K`/`4K` 規格值不支援**（回 `Expected format: '<width>*<height>'`）——這跟萬相 2.7 相反（那邊規格值是官方推薦寫法），兩個家族的 size 寫法不能互抄。
  - 行為與 2.0 系列一致：`n` 1–6、參考圖最多 3 張（上游規定 `input.messages` 只能一則，I2I 的 `content` 是 1～3 個 image ＋恰好一個 text）、不支援 `ref_strength`。後端把原本的 `_QWEN2_EDIT_MODELS` 改名為 `_QWEN_FUSION_EDIT_MODELS` 並納入 3.0 兩個型號，`is_qwen2_edit` 一併更名，避免名稱與實際涵蓋範圍脫節。
  - **計價有個 UI 顯示不到的落差**：`/api/pricing` 讀到的是 **1K 輸出價**（3.0 $0.03、3.0-pro $0.04）。輸出 2K 時網關會依上游 `usage.output_image_type` 自動補倍率——**`qwen-image-3.0-pro` 的 2K 是 $0.075，接近一倍**，但 UI 的參考單價不會反映。輸入圖是加法附加費 $0.003/張。已寫進 `README.md`，但 UI 上目前沒有提示。
  - `qwen-image-3.0-pro` 官方 **RPM = 1**，連續呼叫很容易 429，測試需要重試間隔（我用 20 秒）。
  - **探測手法的一個教訓**：先前為萬相設計的「送超範圍 `n` 當哨兵、靠錯誤訊息判斷 size 是否合法」在這裡**完全失效**——千問 3.0 的 `n` 驗證發生在 size 之前，所以連 `10*10` 都會回「n 的錯誤」，看起來像是尺寸通過了。我差點據此把一堆沒驗證過的尺寸寫進清單（正是先前 MAI 那個 bug 的成因）。改用合法的 `n=1` 才拿到真正的面積約束訊息。**同一套探測手法不能跨模型家族沿用**。
  - **編輯情境下 `size` 完全不生效**（實測發現）：送 `1280*720`（橫向）與 `512*512` 都一樣得到 2048×2048，連 T2I 的面積約束都不套用。所以 3.0 的兩個編輯條目標了 `no_size`，UI 不顯示那個沒有作用的選單。
  - **順帶查出這是既有問題、不只影響新模型**：`qwen-image-2.0`（輸出 1024×1024，跟隨輸入圖）與 `wan2.7-image`（輸出 2048×2048）的編輯情境實測也一樣忽略 `size`，分屬兩個不同家族。這兩邊的尺寸選單目前**仍然顯示**——因為改動會影響現有使用者的介面，我只修了自己這次新增的部分，既有模型待決定。已寫進 `README.md`。
  - 順帶把 `qwen-image-3.0-pro` 的實際 `usage` 回報給 nen-ai-platform 端（他們因 RPM=1 打不進去，pro 的 2K 分級是從 base 推斷的）：實測確認 pro 2K 確實回 `qima_output_2k`，推斷正確。另外提醒他們一個細節——`input_image_count: 0` 時 `input_image_type` 仍會回值且跟著輸出解析度變（1K 回 `qima_input_1k`、2K 回 `qima_input_2k`），若有計費路徑單看這個欄位判斷有無輸入圖會誤判。

- feat：圖片生成改用「行內佔位卡」取代全螢幕等待遮罩，生成期間平台可以繼續操作（commit 待補）。
  - 原本按下生成會叫 `showLoading()` 蓋上全螢幕遮罩，而圖片生成動輒 30～45 秒——這段時間整個平台是鎖死的，連切到別的頁籤看先前的結果都不行。
  - 改成比照影片頁籤的作法：立刻在結果區插入一張生成中的卡片（沿用影片那組 `vtc-*` 樣式，兩個頁籤觀感一致），顯示模型名、即時計時、不定量進度條與佔位區塊；完成時**就地把佔位卡換成結果圖**（用 `replaceChild`），所以結果會留在當初送出的位置，不會插到後來完成的任務前面。失敗則把同一張卡轉成紅色的錯誤狀態，不再只是一個會消失的 toast。
  - 送出鈕也不再鎖定，可以同時送多張。卡片會立刻出現、本身就是「已收到」的回饋，不需要靠鎖按鈕來防重複點。
  - 進度條用 CSS 來回滑動的不定量動畫（`.vtc-progress-bar.indeterminate`），而不是假造一條會自己往前爬的進度條——圖片生成是單一同步請求，我們拿不到真實進度，假進度只會誤導。
  - 計時器 interval 存在卡片物件上，完成/失敗時一定清掉，避免分頁長開累積計時器。
  - 語音辨識／合成那兩處的遮罩保留，它們是短請求。
  - 前端改動：`app.js?v=59` → `?v=60`、`style.css?v=26` → `?v=27`。

- fix：把上一筆為萬相 2.7 新增規格值時**不小心改掉的預設尺寸**改回來。加入 `1K`/`2K`/`4K` 時我把 `2K` 放在清單第一位（因為官方文件說 2K 是模型預設），但前端是取清單第一個當預設值——這等於把所有人的預設從 `1024*1024` 換成 2K。實測 2K 比 1K 慢約 1.5 倍（42.5s vs 27.2s）且像素多 4 倍，不該在使用者沒察覺的情況下變更。已改回 `1024*1024` 為首。

- fix/feat：`glm-5.2` 改回走 OpenAI 相容的 `/v1/chat/completions`，恢復思考開關；並為 GLM 5.x 新增分段推理強度 `reasoning_effort`（commit 待補）。網關：正式環境 `https://nen.com.tw`。
  - **這是把 2026-08-10 那筆「glm-5.2 改走 `/v1/messages`」改回來。** 當時改完就發現那條路徑關不掉思考，這次查清楚了根因，確認短期內無解，所以撤回。
  - **根因（由 nen-ai-platform 端查證原始碼確認）**：`service/convert.go` 的 `ClaudeToOpenAIRequest` 裡，`claudeRequest.Thinking` **只在 `isOpenRouter` 分支被讀取**，其餘所有渠道走的 else 分支只做「模型名以 `-thinking` 結尾就補後綴」的處理——所以上游根本沒收到任何思考設定，走自己的預設（思考開啟），網關再把回傳的思考包裝成 Anthropic thinking 區塊。`reasoning_effort` 更早就消失：`dto.ClaudeRequest` 根本沒有這個欄位，JSON 反序列化階段就被丟棄，連 adaptor 都沒看到（這解釋了為什麼送它會「回 200 但完全無效果」）。
  - 這不是只影響 glm 的問題，**除了 OpenRouter 以外的所有渠道都沒映射**。要修得比照 `isOpenRouter` 做渠道類型閘控（`enable_thinking` 是智譜/阿里方言，無條件發給 OpenAI 等嚴格上游會 400），超出對方被交辦的範圍，他們會轉給他們的使用者決定，並建議我們短期先改回 OpenAI 相容路徑。
  - 我這邊的實測證據（各參數跑 3 次取中位數，排除雜訊）：`thinking.type=disabled` 串流區塊照樣是 `['thinking','text']`；`budget_tokens` 128 與 2048 的中位數 195 vs 178 完全重疊；頂層 `enable_thinking:false` 中位數 138。對照組同一模型走 `/v1/chat/completions` 時 `enable_thinking:false` 讓 completion_tokens 從 159 掉到 **19**。
  - **順帶補上一個我們一直沒用到的能力**：GLM 5.x 除了布林開關，還支援字串的 `reasoning_effort` 分段推理強度。實測各段的 `reasoning_tokens`：`none`／`minimal` → 0、`low` 182、`medium` 198、`high` 202、`xhigh` 239、`max` 208。**可用枚舉各型號不同**——`glm-5.2` 七段全有，`glm-5.1` 不支援 `max`（送了回 400 並列出正確清單）。
  - 兩者的優先權關係也實測過：`enable_thinking:false` 高於 `reasoning_effort`（配 `effort=max` 仍然是 1 個 token）；但**反向不成立**——`enable_thinking:true` 配 `effort=none` 仍然不思考，也就是「關」的那一方永遠贏。這點官方文件只寫了前半段。
  - 前端的推理強度選項原本是寫死在 HTML 裡的五個值（`none/low/medium/high/xhigh`），少了 GLM 需要的 `minimal` 與 `max`。改成由 MODELS 的 `reasoning_efforts` 動態產生，並在切換到不支援當前值的模型時自動退回「預設」——避免使用者選到一個會被上游 400 的值。GPT 條目也一併補上各自的枚舉。
  - 移除了 `_ANTHROPIC_MESSAGES_MODELS` 與三個相關函式（共 114 行）。那條路徑的知識保留在 `README.md` 與這裡，需要時可從 commit `a37c75b` 取回。
  - 端到端實測：思考開 185 tokens／關 1 token、七段強度逐一驗過、串流拿到 316 字思考過程、`glm-5.1` 送 `max` 確實被上游擋下（前端已不會提供該選項）。
  - 前端改動：`app.js?v=58` → `?v=59`。
  - 附帶情報：`glm-5.2` 不在任何編譯進去的 ModelList 裡，網關上看到的 glm-5.1/5.2 是管理員手動加進渠道「支援模型」清單的結果。所以要上架 `glm-5.2-us` / `glm-5.2-fast-preview` **不需要改程式碼**，後台加模型名即可（前提是上游帳號有權限）。

## 2026-08-08

- fix：手機版登入頁三個問題（使用者回報「手機看好像燈不見了」）。**燈在手機上被隱藏其實是功能退化，不只是美觀問題**——燈同時是登入頁切換淺色/深色模式的唯一入口，`@media (max-width: 760px) { .login-lamp { display: none } }` 等於讓手機使用者完全沒辦法切主題。改成把燈排到卡片下方（那裡本來就是一大片空白）並縮小到 58%；因為 `transform: scale()` 不會改變版面高度，要用負 margin 把縮放後多出來的空白（236px × 0.42 ≈ 99px）收掉，否則底下會留一段幽靈空隙。順手抓到一個既有 bug：`.login-card` 只寫了 `width: 420px` 沒有 `max-width`，在比 420px 窄的手機上卡片會直接撐破視窗、左右被切掉，補上 `max-width: calc(100vw - 28px)`。另外雲照原尺寸放在窄螢幕上會大到擠住登入卡片，等比例縮到 62%——實作上把雲的寬度改成透過 CSS 變數傳遞而不是 JS 直接設 `style.width`，因為 inline style 的優先權比 media query 高，直接設的話得靠 `!important` 才蓋得掉。已用 Playwright 在 390px 寬（iPhone 14 Pro）實測：無橫向溢出、卡片 362px、用真實觸控座標點燈確認能正常切換 dark→light、燈的底緣與場景底緣一致（無幽靈空隙）。
- feat：登入頁重新設計成「拉燈切換晝夜」的情境畫面。經過幾輪跟使用者來回調整後的最終樣貌：
  - **拉燈＝切換淺色/深色模式**（深色＝燈亮、暗房裡一盞燈；淺色＝燈熄、白天不需要燈），取代原本只是裝飾性的開關燈。拉繩掛在燈罩**右側下緣**（不是正中央），點下去是「往下拉再回彈」的動作。實作上刻意把繩子和珠子拆成兩個元素各自動畫（繩子 `scaleY` 延伸、珠子 `translateY` 下移）——如果把珠子放在繩子裡面直接對繩子 `scaleY`，珠子會被一起垂直拉長變成橢圓；兩者位移量對齊（40px × 1.42 ≈ 17px）才不會拉的時候珠子脫離繩子末端。繩子的 `z-index` 必須比燈罩低，上端才會被燈罩遮住、看起來是從燈罩下面垂出來（一開始設得比燈罩高，線頭就露在燈罩表面上）。
  - **檯燈造型**改走簡約溫馨路線：奶油色燈罩＋木紋燈柱，取代第一版冷灰色的塑膠感；燈亮時燈罩轉暖黃、加上暈光與往下灑的光錐。
  - **天空的太陽/弦月**（依使用者提供的參考圖）放在左上角，用 inline SVG 畫——太陽是 12 道放射光芒（星形多邊形頂點自己算）＋實心盤＋內圈白環，弦月用 mask 把一個位移的圓從大圓挖掉、旁邊三顆四角小星星。CSS 做不出這種乾淨的尖角形狀。動態：光芒 30 秒緩慢自轉、光暈 5 秒呼吸、小星星錯開時間閃爍。注意 SVG 元素旋轉一定要加 `transform-box: fill-box`，否則 `transform-origin` 會落在 SVG 座標系原點（左上角），光芒會繞著畫面外一個點「公轉」而不是自轉。
  - **切換主題的過場**做成絲滑漸變：淺色/深色兩片背景疊在一起用 opacity 交叉淡化（原本直接換 `background-image` 會瞬間跳色），太陽/月亮則同時淡入淡出＋沿弧線位移，像日月升落；卡片內的文字/輸入框/提示框也補上顏色過渡。
  - **兩種模式各自的背景動態**：淺色是飄過的雲＋暖色光塵，深色是閃爍星空＋偶爾劃過的流星。雲的形狀用 SVG（平底長條＋三個高低不同圓弧）畫，純用 `radial-gradient` 疊只會糊成一顆扁橢圓（藥丸狀）怎麼調都不像雲；另外白色的雲畫在接近白色的背景上本來就看不見，所以天空上半部（雲的高度）鋪了一層淡天藍當襯底，下半部維持偏暖色調不破壞溫馨感。雲的高度限制在 4%～23%，再往下會飄到檯燈/卡片的高度跟近景物件打架。
  - 登入卡片圓角加大、陰影改柔和有深度的版本；**logo 加白色圓形底盤**（原本直接放在青色 header 上對比不足、看不清楚）；場景底部加地平線微光讓燈跟卡片像站在同一平面上。
  - 全部只在本機測試（`venv/bin/python app.py`），用 Playwright 逐項驗證：拉繩動畫的繩子/珠子位移量同步、繩子被燈罩遮住的實際座標、光芒真的在自轉（隔時間取樣兩次比對矩陣）、小星星錯開閃爍、主題過場中間值確實是漸變而非瞬間跳、窄螢幕（414px）天體不會被邊界切到或壓到卡片。兩個主題都無 console 錯誤。
- feat：UI 微互動美化。裝了 emilkowalski/skills 這套設計/動畫 skill（`emil-design-eng`、`find-animation-opportunities` 等，只裝在本機 `.agents/`、`.claude/skills/`，不進版控），用它掃了一輪找按壓回饋/進場動畫的缺口再動手實作：
  - 全站按鈕（`.btn`/`.btn-primary`/`.btn-ghost`、主題切換鈕）補上 `:active { transform: scale(0.97/0.92) }` 按壓回饋，之前完全沒有任何按下的視覺反饋。
  - AI Canvas 的工具列按鈕、節點「生成」/「新增參考圖」按鈕、`#addNodeMenu`/`#templatesMenu` 彈出選單同樣補上按壓回饋；彈出選單另外補上 `scale(0.95)+translateY(-4px)` 的進場動畫（原本是 `display:none↔block` 直接瞬間切換，完全沒有過渡）。
  - 主測試台與 Canvas 的 lightbox（點圖放大）內容原本背景會淡入但圖片/影片本身是瞬間出現，補上 `scale(0.95)→1` + 淡入。
  - 全站既有的 `fadeUp` 進場動畫（聊天訊息、圖片卡、主題選單）從瀏覽器內建偏弱的 `ease` 換成自訂 `--ease-out: cubic-bezier(0.23,1,0.32,1)`。
  - 追加使用者要求的「多一點美感」：登入卡片進場加上上浮+縮放、logo 帶一點旋轉彈入（一個 session 只會看到一次，delight tier）；圖片結果卡加上滑鼠移入的浮起效果（`translateY(-3px)` + 品牌色陰影）與更有實體感的進場動畫（`cardReveal`：淡入+輕微縮放，取代純 fadeUp）；影片結果卡成功時加一層淡綠色光暈陰影跟失敗/進行中的卡片區分；header「本次花費」徽章每次金額更新會有 0.6 秒淡橘色閃爍提示。
  - 刻意沒動：Tab 切換／native `<select>`（頻率太高、native 元件難控制）、串流輸出的文字/進度條（使用者在讀的功能性資料，動畫只會干擾）、節點右上角「刪除」/「+新增關聯」小圓按鈕（用 inline `transform: scale()` 做畫布縮放跟隨，加 CSS `:active` transform 會被 inline style 蓋掉不會生效）。
  - 全部只在本機測試（`venv/bin/python app.py`），用 Playwright 以真實滑鼠 down/up（不是模擬事件，避免誤判 `:active` 沒生效）驗證按壓縮放的實際生效時序、選單動畫、卡片浮起效果，兩個頁面都沒有 console 錯誤。

## 2026-08-07

- feat：登入頁新增失敗次數鎖定。`POST /login` 原本完全沒有速率限制，可以無限次嘗試不同金鑰；改成依來源 IP（取 `X-Forwarded-For` 第一段，Cloud Run 前面有 LB/CDN）計數，連續失敗滿 5 次鎖定 5 分鐘，鎖定期間即使帶對的金鑰也會直接被擋（避免有人拿這支 endpoint 當 oracle 試金鑰），成功登入會重置計數。純記憶體實作、沒有資料庫，Cloud Run 服務重啟/縮容到 0/多實例時計數會不見或不共享——只是盡力而為拉高門檻，不是嚴格保證，已在 `CLAUDE.md` 記錄這個限制。前端新增鎖定倒數：鎖定期間登入按鈕會停用並即時倒數顯示剩餘秒數，時間到自動恢復可再次嘗試。已本機實測驗證：連續 5 次錯誤金鑰後第 5 次直接回鎖定訊息，鎖定期間送對的金鑰仍被擋，倒數期間按鈕正確停用並逐秒遞減。
- fix：修好「思考模式」開關無效的 bug 之後，逐一實測了全部 17 個有 `thinking` 旗標的文字模型（`enable_thinking` true/false 各打一次直接比對回應是否有 `reasoning_content`），抓到兩類還沒被前一筆修正處理到的情況：`qwen3-coder-plus`/`qwen3-coder-flash` 這兩個代碼模型實測 `enable_thinking` 完全沒有效果（true/false 都不會有思考過程），`dola-seed-2.0-lite`/`dola-seed-2.0-pro` 則是完全相反——不管送 true 還是 false 都無條件會思考、關不掉，跟 Gemini 3.x 系列同一種「關不掉」的情況。這兩種都不是程式碼的 bug，是上游模型本身的行為，所以不是再修 `/api/text/generate` 的邏輯，而是把 `MODELS["text"]` 裡這 4 個模型的 `thinking` 旗標改成 `False`，讓 UI 不顯示一個實際上沒有作用、會誤導使用者的開關（沿用跟 Claude/Gemini 相同的既有處理方式）；`dola-seed-2.0` 系列即使不顯示開關，思考過程還是會照常在回答上方顯示成「思考過程」收合區塊，因為前端顯示邏輯是看回應裡有沒有 `reasoning_content`，跟這個旗標無關。其餘 13 個模型（Qwen 旗艦/均衡/極速系列、DeepSeek、GLM）都確認開關正常生效。
- fix：文字生成的「思考模式」開關關掉後其實完全沒用——使用者回報 `qwen3.8-max` 把「思考模式」關掉，回應卻還是帶了「思考過程」。查了才發現 `/api/text/generate` 原本的寫法是 `if data.enable_thinking: extra_body["enable_thinking"] = True`，只有勾選開啟時才會帶這個欄位，關閉時就完全不送；直連上游 API 實測發現 `qwen3.5-flash`/`qwen3.6-flash`/`qwen3.8-max`/`deepseek-v4-*`/`glm-5.*` 這些模型預設就是思考模式開啟，完全不帶 `enable_thinking` 欄位並不會關閉思考，只有明確送 `enable_thinking:false` 才有效——也就是說這個開關從一開始關掉就形同沒作用，一直在多花 token、多等時間。改成除了 GPT 系列（送這個參數會直接 400）以外，一律明確帶上 `true`/`false`；也實測確認 Claude/Gemini 帶 `enable_thinking:false` 不會報錯（只是沒效果，跟原本文件記載一致），不會有副作用。
- feat：補上 ByteDance/即夢模型缺的其他生成模式。Seedream 5.0 Pro/Lite 補上圖像編輯 (I2I)——實測走一般 `/v1/images/edits` 流程即可，`ref_strength` 參數有效不會被拒絕；Seedance 三個模型（bytedance-seedance-1.5-pro、dreamina-seedance-2.0、dreamina-seedance-2.0-fast）補上圖生影片 (I2V) 與參考生影片 (R2V)——實測跟萬相系列共用同一套 `media`/`image`/`images` 三欄位注入機制，i2v 在 bytedance-seedance-1.5-pro、r2v 在 dreamina-seedance-2.0-fast 上都實測完整跑到 `completed` 拿到影片網址。另外實測確認 Seedance 系列**不支援**視頻編輯（vedit）——直接送一段影片當輸入會被上游拒絕（`content[0].image_url` 參數不合法），因此沒有新增對應的 MODELS 項目。
- feat：新增 8 個 ByteDance/即夢模型（正式環境 `nen.com.tw` 實測皆可用）。文字：`dola-seed-sc`（一般對話）、`dola-seed-2.0-lite`/`dola-seed-2.0-pro`（預設就會回 `reasoning_content` 思考過程，機制跟 DeepSeek/GLM 的 `enable_thinking` 相同）；圖片：`dola-seedream-5.0-pro`/`dola-seedream-5.0-lite`（走 `/v1/images/generations`，尺寸格式跟 GPT Image 一樣是 `WIDTHxHEIGHT`，也接受 `2k`/`3k`/`4k` 預設值——lite 版實測畫面至少要 ~369 萬像素，`1024x1024` 這種小尺寸會被上游拒絕，只列 2K 起跳的尺寸）；影片：`bytedance-seedance-1.5-pro`/`dreamina-seedance-2.0`/`dreamina-seedance-2.0-fast`（走一般 `/v1/videos` 任務制流程，跟萬相系列相同，不需要新增後端特殊處理，其中 `dreamina-seedance-2.0` 實測完整跑到 `completed` 並拿到影片網址）。三種模態都只需要在 `MODELS` dict 補資料，前端下拉選單、AI Canvas 節點、`/api/pricing` 計費顯示都會自動吃到，沒有另外改程式邏輯。
- fix：暫停 AI Canvas 影片節點的「影片延伸」功能。上一筆的 `/api/proxy/fetch` 白名單修正雖然沒錯，但用 `gcloud run services describe` 查正式環境的 Cloud Run 服務才發現：**正式環境完全沒有設定任何雲端儲存的環境變數**（OSS/S3/GCS 皆無），所有生成結果全部退回寫在容器本機磁碟；而服務設定 `min-instances=0`、`max-instances=5`，本機磁碟不會跨容器實例共享、也不會在縮容重啟後保留，導致任何「接續/重新抓取前一步生成結果」的操作都可能在中途換到別的容器實例時抓不到檔案（404）——這才是「影片延伸」在正式環境上失敗的真正根因，不是網域白名單問題。跟使用者確認後決定先不接雲端儲存，改用 `static/js/canvas.js` 新增的 `VIDEO_EXTEND_ENABLED = false` 開關暫時關閉這個功能（不新增輸入插槽、不顯示上傳片段的 UI），等之後正式環境接上持久化的雲端儲存後端再打開。
- fix：正式環境（GCS 儲存後端）上 AI Canvas 任何「把前一步生成結果抓回來當輸入」的功能全部失敗，錯誤是「無法取得來源檔案」——實際在正式站測試「影片延伸」（接上一段生成好的影片當來源片段）時發現。根因是 `/api/proxy/fetch`（CORS 繞過代理，canvas 節點跨網域抓取上一步生成結果時會走這裡）的網域白名單 `_PROXY_ALLOWED_SUFFIXES` 從一開始就只寫了 `*.aliyuncs.com`（阿里雲 OSS），沒有隨著後來新增的 AWS S3、GCP GCS 儲存後端一起更新；正式環境目前用的是 GCS，簽名網址網域是 `storage.googleapis.com`，完全不在白名單內，直接被拒絕。補上 `*.amazonaws.com`、`storage.googleapis.com` 兩個網域，SSRF 白名單機制本身不動。
- feat：AI Canvas 的「影片 Video」節點新增「來源影片(延伸)」輸入插槽（`video` 型別），可接其他影片節點的輸出、也可直接上傳本機影片檔案。接上後 `_detectMode()` 會判定為 i2v 模式但改走既有的 `first_clip`/`first_clip_last_frame`（`/api/video/i2v` 早就支援、主測試台影片分頁的「影片延伸」選項用的就是這個機制，這裡是重用，後端沒有改動），首幀圖片會被忽略；把一個影片節點的輸出接到下一個影片節點的這個插槽，就能一段接一段串接延長。用「+ 新增關聯節點」的快速選單測過：選 video 型別輸出接新的影片節點，會自動連到「來源影片(延伸)」（因為它是目標節點第一個型別相符的輸入插槽），模型清單也會正確切到 i2v 模型、提示文字變成「（影片延伸，首幀圖片將被忽略）」。

## 2026-08-06

- feat：文字生成新增多輪對話記憶。`TextGenerateRequest` 新增 `history` 欄位（`[{role, content}]`），後端組 `messages` 時插在 system prompt 跟目前這句 prompt 之間一起送給上游；前端用 `textChatHistory` 陣列在瀏覽器端累積，每次發送都帶上目前的完整歷史、拿到回覆後把這輪的 user/assistant 都推進去，按「清除對話」時重置。已實測驗證模型確實會依歷史正確回答（例如先說「我叫小明」，下一輪問「我剛剛說我叫什麼名字」能正確答出）。
- feat：新增即時花費統計。後端 `/api/text/generate` 非串流回應與串流最後一個 SSE 事件都補上 `usage`（`prompt_tokens`/`completion_tokens`，串流版靠帶 `stream_options: {"include_usage": true}` 跟上游要）；前端在 header 新增一個「本次花費」徽章，文字生成用 token usage 乘上 `/api/pricing` 查到的單價精確估算，圖片/影片/MuleAI Spicy/語音辨識則用固定單價 × 成功次數累加——只對 `pricingMap` 裡標記 `type:'fixed'` 的模型計費，token 計費的圖片模型（如 gpt-image 系列）跟語音合成目前沒有對應的精確算法，故先不計入，避免顯示錯誤的數字。純瀏覽器端記憶體累加，重新整理頁面會歸零，僅供參考不是精確帳單。
- feat：AI Canvas 新增「範本」工具列按鈕，內建 5 個常見節點組合範本（文字→圖片生成、圖片生成→圖像編輯、文字腳本→語音配音、文字→圖片+語音雙路輸出、上傳圖片→圖生影片），一鍵套用時用 `LiteGraph.createNode()`/`graph.add()`/`node.connect()` 即時建立節點並依範本定義的插槽索引拉線，不是還原一份手刻的序列化 JSON——這樣範本永遠跟目前節點實作的預設參數、插槽順序同步，不會因為節點程式碼改版就跟著過期失真。每套用一次會往右下偏移擺放，避免疊在前一個範本上面。
- fix：文字生成的「思考模式」實測發現嚴重低估了支援範圍——DeepSeek（v4-pro/v4-flash/v3.2）與智譜 GLM（5.1/5.2）原本在 `MODELS["text"]` 裡標記 `thinking: false`，但實測 `enable_thinking` 對這兩家都真的有效（會回傳獨立的 `reasoning_content` 思考過程），而且 DeepSeek V4 預設就是思考模式開啟、`enable_thinking:false` 才能關閉省 token；改成 `thinking: true`。GPT-5 系列改用完全不同的機制：實測對 GPT 送 `enable_thinking` 會直接 400 "Unknown parameter"，得改送字串 `reasoning_effort`，而且這個網關接受的枚舉是 `none/low/medium/high/xhigh`，跟 OpenAI 官方文件常見的 `minimal/low/medium/high` 不一樣（帶 `minimal` 一樣會被拒絕）——新增 `reasoning_effort` 欄位與對應的「推理強度」下拉選單，跟其他家族的「思考模式」開關互斥顯示。Claude／Gemini 兩種機制都實測過沒有效果（Claude 送了無反應，Gemini 3.x 系列無條件思考、目前找不到任何參數能關掉），因此兩個開關都不顯示，避免讓使用者以為能控制卻其實無效。
- feat：後端新增 `reasoning_content` 擷取與轉發（`/api/text/generate` 串流用 SSE 的 `reasoning` 事件、非串流用 `reasoning_content` 欄位），因為 openai SDK 沒有把這個非標準欄位定義成正式屬性，改用 `getattr(..., "reasoning_content", None)` 並落回 `model_extra` 查找（沿用 `/api/omni/chat` 抓 `audio` 欄位時就已經用過的既有慣例）。前端新增可收合的「思考過程」區塊（`.msg-reasoning`），開始輸出正式回答時自動收合，避免思考過程跟答案混在一起。
- docs：README.md 文字生成模型表格補上原本完全缺失的 Claude（9 個）、GPT（9 個）、Gemini（7 個）共 25 個模型，並新增一段說明三種思考模式機制互斥、各家支援狀況的整理。
- feat：所有分頁的模型選單新增參考單價顯示。發現這個網關本身就有 `/api/pricing` 這支「New API」類閘道專案的標準計費表 API（回傳 111 個模型的計價資料），因此不用自己手動維護一份跨 6 家廠商、上百個模型的價目表——新增 `/api/pricing` 後端 endpoint 代理並快取這份資料 1 小時（避免每次載入頁面都打一次上游），換算成美金的公式已用 `quota_per_unit=500000`、`group_ratio=1` 實測反推確認：`quota_type=1`（圖片/影片/Spicy 等）的 `model_price` 本身就是每次呼叫的美金價；`quota_type=0`（文字/語音等 token 計費）則是「每 1M input token 美金 = model_ratio × 2」、「每 1M output token 美金 = model_ratio × completion_ratio × 2」。前端在每個「模型」選單的 label 旁邊新增一個不會被選單寬度截斷的價格提示（例如「模型 ($2→$6/1M)」或「模型 ($0.075/次)」），切換模型時即時更新；下拉選單展開時每個選項也會帶上同樣的價格字樣方便比較。這只是換算後的參考價格，不是精確帳單金額。
  - 過程中抓到一個真的 bug：價格是背景非同步載入、載入完成後會重新呼叫 `populateSelectors()` 補上價格顯示，但原本 `populateSelect()` 重建選單時完全沒保留使用者已經選好的模型，會被打回第一個選項；補上保留邏輯後又抓到第二層 bug——如果無條件把舊值設回去，遇到新清單根本沒有那個值的情況（例如語音分頁切換 ASR/TTS 兩種完全不同的模型清單），選單會變成完全沒有任何選項被選中（`value` 變空字串），而不是退回選第一項；改成先確認新清單裡真的還有那個值才恢復，否則保持瀏覽器預設行為。
- fix：使用者回報語音辨識模型顯示「$0/次」看起來像免費——查證發現 `qwen-audio-3.0-asr-flash` 的原始定價其實是 $0.000035/次（串流版是 $0.00009/次），根本不是 0，是後端 `round(x, 4)` 把這種極小數值直接捨去顯示成 0，誤導使用者以為免費。改成後端不做固定小數位 round、保留完整精度，前端新增 `formatUsd()` 依數值大小動態決定顯示的小數位數（先抓到第一個非零小數位再多留一位），一般價格（如文字模型 $2/1M）顯示不受影響，極小的按次計費價格也能正確顯示。

## 2026-08-05

- feat：語音合成新增 3 個 Gemini TTS 模型：`gemini-2.5-pro-tts`、`gemini-2.5-flash-tts`、`gemini-3.1-flash-tts-preview`。實測發現這幾個模型雖然要求提供的 curl 範例是 Google 原生的 `/v1/text:synthesize`（Google Cloud TTS 格式），但在這個網關上該路徑會直接回錯（`invalid request type, expected GeneralOpenAIRequest, got AudioRequest`）；改測 OpenAI 相容的 `/v1/audio/speech`（只帶 `model`/`input`/`voice`）反而成功，固定回傳 `audio/wav`，且不支援 `instructions`（帶了會 400）。`/api/voice/tts` 因此依 `model` 是否以 `gemini` 開頭分流到不同上游 endpoint；前端「進階參數」區塊（instructions/sample_rate/volume/language_hints，這些是 CosyVoice v3 專屬）選到 Gemini 模型時會自動隱藏，音色輸入框的 placeholder 也會跟著換成 Gemini 的音色範例（例如 `Kore`）。已用正式環境 API Key 對 `nen.com.tw` 實測三個模型皆可正常生成語音。
- feat：AI Canvas（`/canvas`）新增「語音 TTS」節點，實作原本停用中的佔位節點（`語音 Audio（尚未支援）`），呼叫跟主測試台語音分頁同一套 `/api/voice/tts`。節點有 `text` 輸入插槽（可接文字節點輸出，也可手動輸入）與 `audio` 輸出插槽；表單包含模型/音色/輸出格式，以及 qwen-audio-3.0-tts 專屬的進階參數（語氣風格描述、取樣率、音量、語言提示），選到 Gemini 模型時會自動隱藏這組進階參數（比照主測試台的規則）。新增 `setPreviewAudio()` 共用函式在節點內顯示 `<audio>` 播放器與下載連結，`_restoreGenResult()` 也補上 `audioUrl` 分支讓存檔/還原機制正確保留生成結果。已用 Playwright 對 qwen 與 Gemini 兩種模型分流各自實測生成成功、下載連結有效，並驗證重新整理頁面後（讀取 localStorage 自動存檔）音檔與播放器狀態能正確還原。
- feat：語音合成的音色 (`voice`) 從自由輸入文字改成下拉選單，主測試台與 AI Canvas 的 TTS 節點都比照辦理。後端 `MODELS["voice"]["tts"][*].voices` 新增查證過的官方音色清單：`qwen-audio-3.0-tts-plus` 2 個、`qwen-audio-3.0-tts-flash` 12 個（各自專屬、不可混用，來源 [Qwen-Audio-TTS 音色列表](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list)），3 個 `gemini-*-tts` 模型共用同一組 30 個官方音色（來源 [Gemini TTS 官方文件](https://ai.google.dev/gemini-api/docs/speech-generation)）。前端選單依選到的模型動態重建選項，切換模型時若目前選的音色不屬於新模型會自動重置為「留空 = 預設音色」。已用 Playwright 分別在主測試台（Qwen Plus/Flash/Gemini 三種模型）與 Canvas TTS 節點（Gemini）實測指定音色（`Puck`、`Fenrir`）生成成功。
- feat：新增 `static/robots.txt`（掛在 `/robots.txt`，因為 `/static` 掛載路徑本身抓不到根目錄）與兩個模板 `<head>` 裡的 `<meta name="robots" content="noai, noimageai">`，明確表態退出 AI 訓練/內容擷取。這個專案的頁面都要先登入 API Key 才能用、沒有公開的文字內容頁面，所以 robots.txt 只簡化擋最主要的幾個訓練型 UA（GPTBot、ClaudeBot、CCBot、Google-Extended），其餘一律放行；沒有做逐頁 canary 浮水印追查機制（那是為多頁 docs 文件站設計的，這個專案沒有對應的公開內容頁面可埋標記，做了也沒有實質意義）。
- feat：圖片生成補上 4 項先前研究發現的參數缺口，都已對照官方文件並實測驗證過。
  1. 確認 `gemini-3-pro-image-preview`、`gemini-3.1-flash-image-preview` 這兩個網關列出但實際已下線的 preview 模型（實測呼叫回 404 "model not found"）不需要加入，現有的非 preview id 已是正確版本。
  2. Gemini 圖片模型新增「圖片比例」選項（1:1/16:9/9:16/4:3/3:4）。實測過官方的結構化 `imageConfig.aspectRatio`（含 top-level `aspect_ratio`、`extra_body.imageConfig`、`generationConfig.imageConfig` 等各種巢狀寫法）在這個網關的 `/v1/chat/completions` 上一律被靜默忽略、輸出永遠是預設比例；但直接在 prompt 文字裡用自然語言要求比例（例如 `Generate an image with aspect ratio 16:9 depicting: ...`）卻真的有效，已對 1:1/16:9/9:16 三種比例驗證輸出尺寸正確，因此改用這個權宜做法實作，`_generate_gemini_chat_image()` 新增 `aspect_ratio` 參數。
  3. GPT Image（`gpt-image-2`/`gpt-image-1.5`）新增 `quality`（auto/low/medium/high）、`background`（auto/opaque/transparent，透明背景）、`output_format`（png/jpeg/webp）三個 OpenAI 標準參數，T2I 與 I2I 皆支援；已實測確認 `background=transparent` 真的能輸出 RGBA 透明背景圖。
  4. 萬相 2.7（`wan2.7-image-pro`/`wan2.7-image`）新增「組圖模式」（`enable_sequential`）開關，開啟後一次生成一組風格/角色連貫的故事圖組、`n` 上限由 4 提高到 12（實際張數由模型決定）；`wan2.7-image-pro` 純文生圖情境下另外新增 2K（`2048*2048`）、4K（`4096*4096`）解析度選項。已實測確認 4K 輸出真的是 4096×4096、組圖模式一次能拿回 4 張連貫圖片。
- feat：AI Canvas（`/canvas`）的「圖片 Image」與「圖像編輯 Editing」節點同步補上上面這 4 項圖片生成更新，跟主測試台行為一致。
  - 圖片節點新增「圖片比例」（僅 Gemini T2I）、「組圖模式」開關 + 最大張數滑桿（僅萬相 2.7 T2I，開啟後 `n` 上限變成 12）、GPT Image 的 quality/background/output_format 三個參數（T2I 皆適用）；圖像編輯節點也補上後面這組 GPT Image 參數。尺寸選單依模型清單裡的 `sizes` 動態產生，萬相 2.7 Pro 的 2K/4K 選項不用額外改前端就自動出現。
  - 新增 `setPreviewImageGallery()` 共用函式，組圖模式回傳多張圖片時用兩欄網格顯示（每張各自可放大/下載），不再只顯示第一張；`_restoreGenResult()` 補上 `cv.imageUrls` 陣列分支，重新整理頁面後多圖結果也能正確還原。
  - 過程中抓到一個真的 bug：組圖模式勾選框的變更事件處理器原本用建構子當時捕捉的 `panel` 變數去查詢要顯示/隱藏的區塊，但 `wireConfigOverlay()` 執行後那個區塊已經搬到 `this._configOverlay`、不再是 `panel` 的子節點，導致點擊時 `Cannot read properties of null` 直接壞掉——改成一律用 `this._configOverlay || this._domPanel` 當下真正的容器去查詢，這也是這幾個節點原本就有的既有慣例，只是這裡一開始沒跟上。
  - 順手把 Canvas 圖片節點也補上主測試台原本就有、但 Canvas 端一直沒做的「Gemini 模型隱藏尺寸選單」邏輯（Gemini 不支援 `size` 參數，兩個欄位同時顯示會讓人誤以為尺寸也有效）。
  - 已用 Playwright 對萬相組圖模式（4 張連貫橘貓四季圖）、Gemini 圖片比例＋尺寸選單互斥、GPT Image 透明背景（`background=transparent`）三種情境分別實測到端到端生成成功。

## 2026-08-04

- fix：語音合成 (TTS) 改走正確的上游 endpoint `/v1/services/audio/tts/SpeechSynthesizer`（原本用的 OpenAI 相容 `/v1/audio/speech` 一帶 `voice` 就整個失敗，實測發現這個新 endpoint 才是 `qwen-audio-3.0-tts-*` 系列真正支援 `voice` 的地方，回應是 DashScope 風格的 JSON，音檔要從 `output.audio.url` 下載）。新增 `instructions`（語氣/情緒風格描述，CosyVoice v3 專屬）、`sample_rate`、`volume`、`language_hints` 四個選填參數，前端「語音設定」分頁補上對應輸入欄位。已用測試網關（`192.168.0.245`）帶入官方範例的 `longanlingxin`/`loongjohn` 音色與完整參數實測成功，也驗證了不帶任何選填參數時仍可正常運作。
- fix：`language_hints` 依官方文件說明「目前版本僅處理陣列第一個元素」，前端由中/英文兩個 checkbox 改成單選下拉選單，補齊全部 16 種官方支援語言（zh/en/fr/de/ja/ko/ru/pt/th/id/vi/es/it/ms/fil/ar）；後端也加上防呆，即使收到多個值也只會取第一個送給上游。
- feat：主測試台（`templates/index.html`）新增淺色/深色/自動三種外觀模式，比照 NenAI Platform 控制台的做法：右上角新增一個圓形圖示按鈕＋下拉選單（淺色模式／深色模式／自動模式，自動模式跟隨系統 `prefers-color-scheme`）。實作方式是把 `style.css` 裡原本零散的少數幾個寫死色碼（登入頁背景漸層、滾動條 hover、toggle 開關底色、toast/狀態徽章的淺色底、loading 遮罩等）收斂成 CSS 變數，再用 `:root[data-theme="dark"]` 整組覆寫；因為全站絕大多數樣式本來就是走 `var(--bg)`/`var(--text-body)` 這類變數，所以幾乎不用逐一改元件就能套用深色模式。選擇存在 `localStorage`（`nenai_theme_pref`），並在 `<head>` 內嵌一小段同步 script 在 CSS 套用前就先決定好 `data-theme`，避免登入畫面閃一下錯的主題再跳回來。範圍只含主測試台，AI Canvas（`/canvas`）因為節點畫面是 litegraph.js 直接畫在 `<canvas>` 上、顏色寫在 JS 裡而非 CSS，這次不列入範圍。

## 2026-08-03

- feat：文字生成新增 `qwen3.8-max` 模型（新一代旗艦，取代 `qwen3.7-max` 成為預設排序最前的旗艦選項；`qwen3.7-max` 說明文字改為「前代旗艦」）。已用正式環境 API Key 對 `nen.com.tw` 實測 `/v1/models` 確認存在、並實際呼叫 `/api/text/generate` 驗證可正常對話。

- feat：主測試台新增「語音模型」分頁，支援語音辨識 (ASR) 與語音合成 (TTS)：`qwen-audio-3.0-asr-flash`、`qwen-audio-3.0-asr-flash-streaming`（SSE 串流回傳中間辨識結果）、`qwen-audio-3.0-tts-plus`、`qwen-audio-3.0-tts-flash`。後端新增 `/api/voice/asr`、`/api/voice/asr/stream`、`/api/voice/tts` 三個 endpoint，走 NenAI 網關 OpenAI 相容的 `/v1/audio/transcriptions`、`/v1/audio/speech`；TTS 輸出音檔比照圖片/影片走 `_cloud_put()` 雲端儲存、退回本機 `outputs/audio`。`NENAI_BASE` 改為可用同名環境變數覆蓋（預設仍是正式環境 `https://nen.com.tw`），方便在這 4 個新模型上線正式環境前，先指向測試網關驗證。
  - 這 4 個模型當時只存在於測試網關（`192.168.0.245`），正式環境 `nen.com.tw` 尚未提供；已用測試金鑰對測試網關實測 4 個模型全部打通。
  - 實測發現 TTS 的 `voice` 參數目前不接受 Qwen-TTS（Cherry/Ethan/Serena/Chelsie）或 CosyVoice 慣用音色名稱，帶了會整個請求失敗（`[cosyvoice:] Engine error [411]`）；完全不帶 `voice` 欄位才會成功並使用上游預設音色。因此 `voice` 欄位改為選填，留空時不會送給上游，前端輸入框預設也改為空白。
  - 實測發現 `-streaming` 這個 ASR 模型，在「整檔上傳」（非即時分段音訊）情境下即使帶 `stream=true` 上游也只會回傳一次性 JSON（`Content-Type: application/json`），不是 SSE；`/api/voice/asr/stream` 因此改成先檢查回應的 `Content-Type`，非 SSE 時把整包 JSON 包成單一 SSE 事件回傳，避免真的收到非 SSE 回應時前端完全收不到任何內容。

## 2026-07-30

- infra：正式環境（`nenai-testing-platform`）加上自訂網域 `playground.nen.com.tw`。因為 Cloud Run 的「網域對應」功能不支援 `us-east5` 這個區域，改用 External HTTPS Load Balancer 達成：Serverless NEG（`nenai-testing-platform-neg`）→ Backend Service（`nenai-testing-platform-backend`，已開 Cloud CDN）→ 靜態 IP `34.128.190.102` → Google 代管 SSL 憑證（`nenai-testing-platform-cert`）→ URL Map/HTTPS Proxy/轉發規則。DNS（Route 53，`nen.com.tw`）新增 A 記錄指到該靜態 IP。同時把 Cloud Run 服務的 ingress 改成 `internal-and-cloud-load-balancing`，讓原本的 `*.run.app` 網址對外直接存取會被擋掉（實測回 404），只能透過 `playground.nen.com.tw` 存取；GitHub Actions 的部署走 Cloud Run Admin API，不受此限制影響。
- infra：新增 GitHub Actions workflow（`.github/workflows/deploy-cloud-run.yml`），推上 `nenai` 分支時自動建置並部署到 GCP Cloud Run 正式環境（`nenai-testing-platform`）。需要在 repo 設定裡新增 `GCP_SA_KEY` 這個 secret（服務帳戶金鑰 JSON）才能運作。
- feat：主測試台圖片/影片生成結果卡片補上「耗時」顯示（跟文字生成、影片任務輪詢一致），新增共用的 `fmtElapsed()` 格式化函式；圖片生成（`sendImage`）與影片生成的同步結果路徑（`addVideoResult`，例如 Gemini Omni）之前完全沒有記錄耗時。
- feat：瀏覽器分頁 icon、登入畫面 logo、主頁面左上角品牌 logo 全部換成 NenAI Platform 真正的識別圖案（`static/img/logo.png`），取代原本自製的藍色 SVG 圖示。
- infra：把這個專案部署到 GCP Cloud Run（`nenai-testing-platform` 服務，`ai-model-hub-newapi` 專案，`us-east5` / 哥倫布區域），做為 NenAI 平台的測試 playground。1 vCPU / 1Gi、min-instances=0、gen2 執行環境。詳見 `CLAUDE.md`/記憶裡的正式環境紀錄。
- `65c628a` feat：GCS 儲存後端新增 `GCS_USE_ADC=true` 選項，改用 Cloud Run/GCE 附加的服務帳戶（Application Default Credentials）+ IAM SignBlob API 遠端簽章網址，不需要另外保管服務帳戶金鑰檔；README 補上對應的 IAM 設定步驟（`serviceAccountTokenCreator` 自我模擬 + bucket 權限）。新增 `update.md` 這份變更紀錄檔案，並在 `CLAUDE.md` 加上「每次更新都要記錄」的規定。
- `09320ec` feat：雲端物件儲存新增 AWS S3、GCP GCS 支援，三選一取代原本只有阿里雲 OSS。統一由 `_cloud_put()` 分派，可用 `STORAGE_BACKEND` 明確指定，或依 oss → s3 → gcs 順序自動偵測憑證。

## 2026-07-28

- `8d6bc40` fix：影片生成（t2v/i2v/r2v）關閉「自動配音」開關卻仍然輸出聲音——後端沒收到 `audio` 欄位時，上游會自行判斷是否配音，不能只在開啟時才帶這個欄位，改成無論開關狀態都明確帶 `audio: true/false`。
- `467cab7` fix：修正圖片/影片/文字結果區塊生成數量一多，後面幾筆的下載按鈕會消失的問題——CSS flex 子項少了 `min-height: 0`，加上 grid 容器 `grid-auto-rows` 預設值會把多餘的列壓縮到看不見，兩層都補上修正。

## 2026-07-27

- `7842073` fix：更新 `canvas.js`/`canvas.css` 的 `?v=` 快取版本號，讓前兩次的修改（生成進度條、`getInputData` force_update 修復）真正對使用者生效，而不是被瀏覽器快取擋住。
- `7038cba` fix：AI Canvas 生成節點讀取上游資料改用 `getInputData(slot, true)`（force_update），修正「參考圖節點尚未生成完成」的誤判——LiteGraph 的持續執行迴圈只要有節點噴例外就會靜默永久停止，導致新建立的連線讀不到資料。

## 2026-07-26

- `0b471db` feat：AI Canvas 生成中的等待畫面改用動態假進度條（跑動漸層 + 呼吸光暈 + 已等待秒數），取代原本純文字的「生成中…」提示。
- `ce1155e` feat：AI Canvas 支援參考圖多插槽（可動態新增「+ 新增參考圖輸入」）、專案存檔還原（文字/相機角度/圖片節點的序列化）、新增影片編輯（Video Edit）與動作動畫（Animate）節點，並修掉選單卡住、快取相關的既有 bug。

## 2026-07-24

- `8a176ec` feat：AI Canvas 新增相機角度（Camera Angle）與上傳圖片（Load Image）節點，修正選單卡住與快取問題。

## 2026-07-23

- `7c1bb7d` feat：文字生成補齊 Claude/GPT/Gemini 系列模型，影片卡片的耗時秒數移到模型名稱旁邊顯示。
- `687f2ab` feat：影片生成新增 Veo、Gemini Omni 的 i2v（圖生影片）、r2v（參考生影片）支援。
- `229bee5` feat：圖片生成新增 GPT/Gemini 的 i2i（圖像編輯）與 MAI Image 系列模型，移除部分千問模型。

## 2026-07-22

- `5d520ed` docs：更新 README，補上新增模型清單與 AI Canvas 功能說明。
- `03006b5` fix：品牌藍改用使用者指定的精確色值 `#01A0C7`。
- `9c2623a` feat：AI Canvas 新增 MuleAI Spicy 節點，主平台配色改為 nen 藍。

## 2026-07-21

- `ed8f308` feat：AI Canvas 節點設定面板改為「選中時才浮現」的固定尺寸浮層，取代原本常駐顯示的方式。
