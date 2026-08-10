# update.md

本檔案記錄這個專案的每一次重要更新（新功能、修 bug、重構、設定調整……），依日期分組、由新到舊排列，每筆盡量附上對應的 git commit hash（`git show <hash>` 可查看完整內容）。

依照 `CLAUDE.md` 的規定：之後每完成一次變更、準備 commit 時，都要在最上方新增一筆條目，不要事後補記。

---

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
