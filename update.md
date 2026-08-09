# update.md

本檔案記錄這個專案的每一次重要更新（新功能、修 bug、重構、設定調整……），依日期分組、由新到舊排列，每筆盡量附上對應的 git commit hash（`git show <hash>` 可查看完整內容）。

依照 `CLAUDE.md` 的規定：之後每完成一次變更、準備 commit 時，都要在最上方新增一筆條目，不要事後補記。

---

## 2026-08-10

- fix：另一個 session 對照 nen-ai-platform（阿里/DashScope 渠道）目前的路由與參數結構，幫忙抓出這個測試平台影片端點的兩個實際不會生效的 bug，修正如下：
  - **影片端點的解析度選擇完全沒送到上游**：`/api/video/{t2v,i2v,r2v,vedit}` 原本會把使用者選的 720P/1080P 用 `_res_to_wh()` 轉成 `width`/`height` 放進 payload 頂層，但上游統一任務 API（`TaskSubmitReq`）根本沒有 width/height 這兩個欄位，會被直接忽略——等於 UI 上選的解析度從未真正送達，一律吃伺服器端每個模型自己的預設值。改成直接把 resolution 字串（"720P"/"1080P" 等）放進 payload 的 `size` 欄位，交由上游依模型判斷要轉成 size 還是 resolution 參數；順手移除已無用的 `_res_to_wh`/`_RESOLUTION_WH`。
  - **`/api/video/animate`（wan2.2-animate-mix/move）沒有送 `images`，人物圖/參考影片實際上沒傳到上游**：此檔案裡 i2v handler 早就在註解寫明「平台 TaskSubmitReq 只認 images（陣列），media/image 會被忽略」，i2v/vedit/r2v 三個 handler 也都確實額外補了 `images` 欄位，但 animate handler 唯獨漏補，只送了會被忽略的 `media` 陣列，等於上游收到的圖片/影片內容是空的。補上 `payload["images"] = [人物圖, 參考影片]`（順序對應 wan2.2-animate 上游規則）。
  - 修完後 `docker compose build --no-cache && docker compose up -d` 重新建置部署。

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
