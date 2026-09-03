# update.md

本檔案記錄這個專案的每一次重要更新（新功能、修 bug、重構、設定調整……），依日期分組、由新到舊排列，每筆盡量附上對應的 git commit hash（`git show <hash>` 可查看完整內容）。

依照 `CLAUDE.md` 的規定：之後每完成一次變更、準備 commit 時，都要在最上方新增一筆條目，不要事後補記。

---

## 2026-09-04

- 記錄：**Veo 三項新能力實測**（文檔 session 提需求、Levi 本人核准 $5.6 預算；正式站 `veo-3.1-generate-001`，720p 無配音；閘道帳單淨 $7.00，含一筆失敗未退款 $1.60）。①**lastFrame 生效**：原生路由 `:predictLongRunning` 首幀純紅、尾幀純藍、4 秒 → 成片首格 RGB (218,21,20)、末格 (26,42,220)，1280×720 無音軌。②**referenceImages 無效**：OpenAI 相容路徑字串 data URI 被上游拒（`The image field is required for reference image`）、官方物件形狀被閘道拒（unmarshal 型別 string）；原生路由收了、扣 $1.60，但成片（鉻金屬結雕塑）與不帶參考圖的對照組無差別——分不出是閘道沒轉發還是 Veo 忽略，客戶端結論都是「沒效果照收費」。③**延伸只能走原生路由、只收 7 秒**（上游原文 `supported durations are [7] for feature video_extension`；OpenAI 相容路徑白名單 4/6/8 送不出 7）；回來是 4+7＝**11 秒累積完整影片**，末格綠 (0,202,1) 照提示詞走；輸出帶 aac 音軌但送了 generateAudio:false、按無聲檔次計費。④閘道問題七項（durationSeconds 只收整數、operation 輪詢兩種路徑都壞、referenceImages 兩種寫法、延伸時長校驗、失敗未退款且狀態無錯誤、延伸音軌、4 張參考圖回 500）已送平台 session `nen-ai-platform-ab`；渠道：平台 session 從任務日誌確認**六筆全部走 Vertex AI 渠道（type 41）**、分散在五條渠道，沒有一筆走 Gemini API（type 24）；相容路徑失敗那筆任務資料只有 name、沒有 done/error，與「失敗沒被正確判定」的推測一致，退款待 Levi 親自確認。**平台讀碼根因（轉述，未修）**：referenceImages 是**閘道沒轉發**（原生路由把 instance 攤平成字串陣列、官方物件形狀靜默丟掉），不是 Veo 忽略；延伸音軌是**上游無視 generateAudio:false**，閘道有轉發但計費不校正（平台擬改延伸一律按有聲計費）；延伸時長白名單 4/6/8 不看 metadata.video、且 metadata.durationSeconds 繞過檢查；失敗未退款是 /v1/videos/{id} 即時查詢只抄狀態不退款、背景輪詢不再碰終態任務；4 張圖 500 是驗證在預扣費後。修好部署後要重測一次。**沒動 playground 程式**：影片頁籤目前沒有 lastFrame／參考圖／延伸給 Veo 的 UI，等閘道修完再接（現在接上去只會把「沒效果照收費」推給客戶）。素材與 request body 在 scratchpad `veo/`。
- fix(canvas)：**標記節點改成「記號轉文字、圖片保持乾淨」**（Levi 實測：標記圖接 wan3.0-video 的參考圖，箭頭與編號原樣出現在成片裡；「照你的想法做」）。根因：wan3.0 的參考模式是外觀／內容參考，記號在它眼裡就是畫面內容；「在圖上畫箭頭」那套是 Seedance 2.5 網頁版有訓練過的能力，wan 沒有，提示詞說「不要出現」也擋不住。修法：①**輸出圖預設是不含記號的原圖**（另開離屏 canvas 只畫底圖），模型永遠看不到記號；②新增 `_buildPrompt()` 把記號轉成方位文字（九宮格：左上／上方／右上／左側／中央／右側／左下／下方／右下；編號給位置、箭頭給起→終、筆畫給起中終三點去重後的路線、顏色照色票命名），透過相機角度節點現成的 `_buildPrompt` 掛鉤自動接在下游提示詞前面，使用者在提示詞裡可直接寫「從 1 飄到 2」；③`_autoCameraAnglePrefix()` 改成收集**所有**有 `_buildPrompt` 的上游節點並串接（先前只取第一個，相機角度與標記同接會漏一個）；④浮層多一個唯讀的「自動加進提示詞的動作描述」讓使用者看到送出去的文字，以及「把記號畫進輸出圖片」開關（預設關；開了描述尾端會加「圖上的箭頭、線條與數字只是動作指引，成片中不要出現」），留給之後驗過會讀草圖標註的模型；⑤提示文案改寫。瀏覽器驗證（本機、假金鑰）：畫兩個編號＋一支紅箭頭＋一條藍筆畫後描述為「動作指引：編號 1 在畫面右上；編號 2 在畫面左側；紅色箭頭從左下指向右下；藍色路線 左上→中央→右下。」；乾淨輸出在箭頭位置取樣是底圖的黑（非紅）、與編輯畫布的合成圖不同，開開關後輸出等於合成圖；影片節點同時接相機角度（prompt 插槽）與標記（參考圖 1）時，前綴只收到標記的描述（prompt 來源正確排除）。`canvas.js` 版號 56。**未實測**：真的送去生成看模型照不照文字方位動（要花錢，等 Levi）；Seedance 2.5 API 版讀不讀草圖標註（同上）。
- feat(canvas)：**標記節點接影片節點時預設走「參考圖 1」（參考生影片）**（Levi 看實際用法後指示）。記號是給模型看的指引，先前預設接 `first_frame`（第一個相容插槽）會把箭頭與編號原封不動變成成片第一格。①`connectToFirstCompatibleInput()` 加一個來源節點可實作的 `preferredInputFor(target)` 鉤子，AnnotateNode 對 `nenai/video` 回「參考圖 1」，型別相容才接、否則照舊；②範本「上傳圖片→畫記號→圖生影片」的邊改用插槽名稱 `'參考圖 1'`（LiteGraph 的 `connect` 吃字串）；③標記節點提示文案改成建議接參考圖並說明接首幀圖的後果。瀏覽器驗證（本機、假金鑰）：範本接到 `3:參考圖 1`、「+」快速新增影片節點接到 `3:參考圖 1`、對照組上傳圖片節點「+」仍接 `1:first_frame`。`canvas.js` 版號 55。
- chore(models)：**文字模型下拉改成每組「最新在上」**（Levi 看了下拉截圖指示）。只動 `MODELS["text"]` 的順序、不改任何欄位：Claude 組 `claude-fable-5-1` 移到組首（帶著它的實測註解）；Gemini 組嚴格照版號新→舊（3.8、3.7、3.6、3.5、3.5-lite、3.1-pro-preview、3-flash-preview、2.5×3，先前 3.1-pro 在最上、3.5-lite 在最下）；Qwen 均衡 3.7-plus 移到 3.6-plus 前；GLM 5.2 移到 5.1 前；ByteDance `dola-seed-2.1-turbo` 移到組首。Grok、GPT、DeepSeek 本來就是新的在上，沒動。**消費端檢查**（照 CLAUDE.md「往清單插新項目前先查」）：各組組首變動不影響整份清單的第一筆（仍是 `qwen3.8-max`），Canvas 文字模型預設值與灰模節點的 `textModels[0]` 不變；88 測試全過。
- feat(pricing)：**gemini-3.6／3.7／3.8-flash 的顯示價改成原價**（Levi 裁示：閘道目前是半價優惠到 2026 年底，體驗站不跟著優惠、顯示原始價）。閘道 `model_ratio` 0.375 換算出來是 $0.75→$3.75/1M，新增 `_LIST_PRICE_OVERRIDE` 覆蓋成 **$1.5→$7.5/1M、快取輸入 $0.15**。影響範圍：模型下拉的價格後綴、「本次花費」估算（會比實際扣款高一倍，刻意的）、MCP 的 pricing 透傳。順帶讓所有 token 計費模型的 `/api/pricing` 多回 `cached_input`（閘道 `cache_ratio` × 輸入單價，例如 3.5-flash 0.15）——目前 UI 只顯示 input→output，這個欄位先進資料不進畫面。優惠結束或閘道倍率改回去時把那張表清掉。本機 `/api/pricing` 實測三顆都回 1.5／7.5／0.15，3.5-flash 維持 1.5／9.0；`tests/` 加一條，88 全過。
- feat(text)：**新增 `gemini-3.8-flash`**（Levi 指定對正式站測，本輪合計約 $0.02）。閘道 `/v1/models` 有、計價 `model_ratio` 0.375／`completion_ratio` 5（$0.75→$3.75 每 1M，與 3.7、3.6 相同）、權限已開通。①**思考（多步算術題，各 5 次）**：不帶設定 240～362 都有思考（預設就開，不進 `_GEMINI_THINKING_OFF_BY_DEFAULT`）；`thinkingBudget: 0` → 126～247（中位 136）vs 不帶 → 170～245（中位 237），跟 3.7 一樣「大幅降低但不歸零」，區間有一格重疊（247 vs 170），開關給。`includeThoughts` 拿得到 thought 區塊（經 playground `reasoning_content` 1659 字）；`budget=0`＋`includeThoughts` 同送會 400（"include_thoughts is only enabled when thinking is enabled"），`_build_gemini_body` 本來就不會同時送。②**看得到圖**：原生 `inlineData` 1x1 紅色 PNG 答 "Red"（n=1）；文檔 session 指出純色圖分不出「真的看到」與「猜最常見答案」（丟圖的 bug 存在期間這種測試也會過），補測**圖上隨機三位數字 137 → 答 "137"**（n=1，猜中機率千分之一）。**但 playground 的 Gemini 原生路徑先前把帶圖的 content 陣列整個 `str()` 成文字、圖片靜默丟掉**——這是 Gemini 文字模型一顆都沒標 `vision` 的真正原因。新增 `_gemini_parts()` 把 `image_url` 的 data URI 轉成 `inlineData`（公網網址沒有 fileData 那條路，略過不假裝送了），非串流與串流都經 playground 端點驗過（"Red"）。只有 3.8 標 `vision`；其餘 9 顆 Gemini 沒逐顆實測、不照家族推。③**penalty 任一 >0 就 400 "Penalty is not enabled for this model"**：3.8 的 presence／frequency 各單獨驗、3.7 與 3.6 各驗一次同樣被拒，三顆標 `no_penalties`；`_build_gemini_body` 對 `_TEXT_NO_PENALTIES` 一律不送（先前只在 >0 時送，所以使用者一拉滑桿就會失敗），**前端新增 `no_penalties` 時收起兩個 penalty 滑桿**（`textPresencePenaltyGroup`／`textFrequencyPenaltyGroup` 新加 id；grok-4.6 順帶受益，它先前旗標有、滑桿卻還在）。經 playground 送 penalty 0.5 實測成功。④topP／topK／seed／stopSequences 同送一次 200。⑤desc：3.8「最新均衡模型，支援看圖」、3.7 改「新一代均衡模型」、3.6 改「前代均衡模型」。⑥系統公告 id=30 已寫入並回讀確認；README 文字表更新；`tests/` 加兩條（旗標對應實測值、`_build_gemini_body` 圖片轉 inlineData 與 penalty 閘控），87 全過。**未做**：其餘 Gemini 型號的 vision／penalty 逐顆驗證（各一次呼叫即可，待指示）。

## 2026-09-03

- ci(deploy)：**Cloud Run 部署流程改用服務層級的實例數＋部署後自動清舊修訂版本**（9/1 改的、9/3 進版）。①原本 `--min-instances=0 --max-instances=5` 是修訂版本層級、每次部署烙在新修訂版本上，與服務層級的 20 取較嚴的一邊，實際被壓在 5。改送服務層級 `--min=1 --max=20`，並把修訂版本層級明確設 `default` 清掉（gcloud 581 說明文件確認兩組旗標語意）。`--min=1` 對應 Levi 在主控台已設的 1/20，不是新的成本；之後要省錢改這裡的 `--min`。②Cloud Run 沒有「保留 N 個修訂版本」設定，只增不減（9/1 清過一次 147→2）；部署後多一步保留最新兩個＋任何還在吃流量的版本，其餘刪除，刪除失敗不讓 workflow 變紅。
- feat(canvas)：**灰模節點再評估「要不要改用 Blender」後的四項補強**（Levi：「都做」＋加筆刷標記）。再評估結論（查 Toonkit／騰訊雲 25 案例／即夢白模介紹）：社群「用 Blender 做運鏡參考」的產出**只是一支低畫質灰模 mp4**，沒有 3D 資料或相機 metadata，「小、低畫質就夠」「3–5 秒最穩」；Blender 的價值不在算圖，而在自由相機路徑、會動的物件與角色、多角色站位、精確時間點——四樣 three.js 都做得到，伺服器跑 Blender 的硬性條件（Cloud Run `--cpu=1 --memory=1Gi --timeout=300`，無佇列、無 GPU）仍在，維持不做。①**上傳影片節點 `nenai/load_video`**：先前 Canvas 沒有獨立的影片上傳節點（影片上傳只藏在影片編輯／動作動畫節點裡），有 Blender 的使用者做完灰模接不進 Spicy 節點的「參考影片」插槽。照抄 LoadImageNode，多讀 `<video>.duration` 顯示秒數、超過 15 秒先警告。②**灰模 DSL 加三種動態語法**：`figure H at x y z`（人形假人：腿＋軀幹＋雙臂＋頭，**y 是腳底**，其他量體的 y 仍是中心）、任何量體後接 `move to x y z`（片長內從 at 走到 to，smoothstep 緩入緩出）、`camera from x y z to x y z [look x y z]`（自訂相機起止與注視點，**存在時覆蓋運鏡選單**，浮層顯示提示）。`_applyCamera` 改名 `_applyFrame`（先擺物件再擺相機）；時長上限 8→15 秒（w3.0 參考影片上限）；系統提示詞補上新語法與「figure 的 y 是腳底」「會動的東西用 move to」；AI 產出過濾改 `isGreyboxLine()` 讓 camera 行過得了；新增「走位 Walk」範本示範三種語法。③**匯出場景檔按鈕**：spec 前面加 `# move/distance/height/duration/ratio` 註解，下載 `greybox-scene.txt`（`#` 行解析器本來就略過，貼回節點不會出錯）。④**Blender 腳本 `scripts/blender_greybox_export.py`**：兩種模式——對自己的 `.blend` 只改算圖設定（Workbench、單一灰、Studio 光、無外框、832×480、24 fps、H.264、片長超過 `--max-seconds` 截斷並警告、沒相機就報錯），或 `--spec` 從場景檔重建（座標 three.js→Blender 是 `(x,y,z)→(x,-z,y)`，行列式 +1 所以 rot 不反號；相機用 Track To 約束對一個 Empty，等價於節點的 lookAt；內建運鏡逐格 K 線性關鍵格、自訂相機與 move to 用兩個 Bezier 關鍵格）。**⚠️ bpy 那段沒在真的 Blender 跑過**（本機沒裝），檔頭已註明；純函式部分有測試。⑤**標記節點 `nenai/annotate`**（Levi：「筆刷工具可以在圖片上畫記號指引影片，這類方式網路上也有」——即 Seedance 2.5 的「在圖上畫箭頭」）：輸入 image、輸出畫好記號的 image；工具＝筆刷／箭頭／編號（自動遞增）／橡皮擦＋復原／清除，五色、線寬；沒接輸入也可直接上傳。**筆畫存向量（正規化座標）**放 properties，重整後重畫；記號層與底圖分開畫，橡皮擦（destination-out）只擦記號；來源圖一律走 `fetchAsBlob()`（遠端簽名網址經代理抓回，否則 canvas 被 taint、`toDataURL()` 會丟例外）；輸出最長邊上限 1536。這裡沒有專用的動態筆刷通道，是「畫好記號的圖當首幀圖／參考圖」的軟引導，文案寫「參考性質」。⑥三個新範本：上傳圖片→標記→圖生影片、3D 灰模→運鏡參考→Spicy、上傳影片→運鏡參考→Spicy。⑦**測試**：Python 解析器與 JS 解析器對同一份 9 行樣本（含錯誤行）逐欄比對一致；`tests/` 新增 4 條鎖住解析器、比例表／量體參數表與 canvas.js 同步、運鏡端點值、座標轉換為正旋轉（85 全過）。**瀏覽器冒煙（本機 5052、假金鑰）**：四種節點新增無錯；走位範本 5 個量體＋camera 行解析、預覽 18 階灰、2 秒影片編碼完成、blob 標頭 `ftypisom`（45 KB）；姿勢→標記接線後底圖載入、筆刷／箭頭／編號三種筆畫落入 properties、復原正確遞減編號；三個範本各自接到正確插槽（Spicy 的「參考影片」、影片節點的 `first_frame`）；重整後標記的 3 筆筆畫保留且輸出位元組數相同、灰模範本與 camera 行還原、上傳影片節點提示重選。**未實測**：真的把標記圖／灰模影片送去生成看模型遵不遵循（要花錢，Levi 沒下這個指示）；Blender 腳本在 Blender 裡執行。
- fix(canvas)：**Canvas 自我 review（Levi：「review 看看 canvas 有沒有問題 還有排版 UI/UX」）修掉五處**。①**設定浮層超出視窗就按不到**：浮層固定像素、不隨縮放，灰模的浮層量到 774px、Spicy 700px，節點在畫面下半時底部控制項落在視窗外，而 body 是 `overflow:hidden` 捲不到。`positionAllPanels()` 現在依浮層頂到視窗底的距離設 `max-height`（最少 160）並 `overflow-y:auto`，在浮層內捲動（浮層本來就擋掉 wheel 冒泡，不會順便縮放畫布）。這是通用修法，所有節點受益。②**灰模節點還原存檔後滑桿與數字沒同步**：滑桿是建構時照預設值填的，`onConfigure` 只重設了選單；時長是這次新加的漏掉，距離／高度是先前就漏的。③**標記節點來源圖競態**：`_loadSource()` 是 async，上游快速換圖時舊的可能最後才回來蓋掉新的，加 token 作廢過期結果。④**上傳影片節點預覽框寫「尚未生成」**——它不是生成節點，改「尚未選擇影片」。⑤**灰模浮層文案露出腳本路徑** `scripts/blender_greybox_export.py`——那是給開發者看的，客戶介面只講「可搭配 Blender 匯入腳本重建」。另外**標記線寬預設 4→6、基準 /500→/400**：截圖看 1024 寬的圖上原本只有 8px 線、箭頭頭很小，模型讀圖時容易忽略。驗證：瀏覽器內量到灰模浮層 `max-height:325px`、`scrollTop` 可到 449；重整後三個滑桿值與 properties 一致、標記 4 筆筆畫還原、上傳影片預覽文字正確。**沒改的（看過、判定不是問題）**：新節點一律出現在畫布中央會疊在一起——既有行為，所有節點都這樣；還原的灰模節點顯示「請重新產生灰模影片」即使先前沒產過——存檔沒記錄有沒有產過，措辭保守可接受。

## 2026-09-02

- fix(scripts)：**修掉 `probe_model.py --drift` 自己的清單漂移**（Levi 指示：那 16 顆都不補，但工具要修）。①**漏收**：`ours` 原本是寫死的分類清單（`text`／`image`／`video`／`muleai` ＋ `voice.asr`／`voice.tts`），**漏了 `voice.realtime` 與 `voice.music`**——那 7 顆明明有收錄，卻被報成「閘道有、我們沒收錄」（實際未收錄 16 顆，卻報 23 顆）。**清單漂移工具自己漂移比沒有工具更糟**：它會催人去補上根本已經在的東西。改成遞迴走訪整個 `MODELS`，往後新增分類（或 voice 下再多子類）不必再改這裡。②**改完之後又收過頭**：TTS 模型底下的音色也長成 `{"id","name","desc"}`，`loongjohn`／`longanlingxin` 這些**音色 id 被當成模型 id**，變成一堆「MODELS 有、閘道沒有」的假警報。加 `_NOT_MODEL_KEYS = {"voices"}` 排除。③走訪函式從 `drift_check()` 內部提到模組層級成為 `collect_model_ids()`，這樣測試測得到**真正的那份**（第一版測試複製了一份邏輯，鎖不住實際程式碼）。④**補測試** `test_drift_collects_every_model_category_but_not_voices`：每個分類（含 voice 各子類）至少收到一顆、音色一顆都不能收。已驗證它抓得到回歸——故意把 `_NOT_MODEL_KEYS` 清空，測試立刻紅。⑤修正後的結論：閘道 146 顆、我們收錄 231 筆（含同 id 不同 type 的多筆），**真正未收錄 16 顆**——其中 4 顆是刻意的（`w3.0-video` 四顆，我們用 `-spicy` 別名）、2 顆沒有對應介面（embedding），其餘 10 顆（Claude 舊款 3、Gemini 3.1 系 4、GPT 2、`glm-5.2-us`）**Levi 裁示都不補**。
- 記錄／feat：**Claude 圖片輸入的兩種形式、以及 omni 延長的 `duration` 都驗完了**（正式站，合計 $0.1144）。①**Claude 圖片：base64 與遠端 URL 都可用**。base64 data URI ✅；遠端 https URL **一開始判定失敗是錯的**——第一個測試網址（Wikipedia）回 `failed to convert openai request to claude request: get file data failed: failed to download file`，但那是**該站擋閘道的下載器**；換成 httpbin 與 raw.githubusercontent 兩個網址都成功。**又一次「單次失敗不等於不支援」**——這次我在下結論前就換網址複驗了。注意 URL 形式的下載是**閘道在做**（錯誤訊息來自轉換階段），所以會受來源站的防盜連影響。②**omni 延長可以指定新增段長度**：`response_format[].duration` 在 extend 任務上**可用**，而且控制的是「**新增那一段**」的長度、不是成品總長——實測 3.008 秒的來源 ＋ `duration=3s` → **成品 6.016 秒**（新增段 3 秒；video tokens 5,793 ≈ 360p 的 3 秒），費用 $0.1132。不帶則是上游預設的約 10 秒，**帶短一點可以省錢**。已接進 `/api/video/omni/extend`（吃 `duration` 表單欄位、走既有的 3–10 秒驗證），前端沿用目前選的秒數。
- feat/記錄：**Claude 全家族補上 `vision`＋omni 的 `aspect_ratio` 與場景延長實測完成**（正式站，本輪 $0.5667）。①**Claude 看圖逐顆實測 11 顆全通**（1x1 PNG 都描述得出來），全部補上 `vision: True`——先前一顆都沒標，客戶在文字頁籤看不到圖片上傳區。**不是照家族推的**：`claude-opus-4-8` 首次回 Bedrock 暫時性錯誤、`claude-fable-5` 首次回空字串（`max_tokens` 給太小），各重試 2 次都正常——**單次失敗不足以判定不支援**。②**`aspect_ratio` 實測通過**：送 `9:16` ＋ 360P ＋ 3 秒，量到 **360×640 直向、3.008 秒**（預設是 16:9 橫向），三個非預設值同時生效，$0.1041。③**場景延長實測通過，而且行為與預期不同**：3.008 秒的來源延長後拿到 **13.013 秒**——**回來的是累積後的完整影片，不是新的一段**，所以 UI 不需要拼接。⚠️ **量到的秒數 ≠ 被收費的秒數**：那支 13 秒收 **$0.3468**，約等於「10 秒新影片」的價（360p 10 秒約 $0.338）加上來源影片的輸入 token，不是按 13 秒收。文檔站事先提醒要分開確認，提醒得對。④**延長時不能帶 `aspect_ratio`**：上游回 `Aspect ratio cannot be set in response format for extend task.`（400、不計費），比例本來就跟著來源走。已從延長路徑移除。⑤**未驗證**：延長的新增長度能不能用 `duration` 指定（目前不送，實測預設約 10 秒）。
- feat(text)：**新增 `claude-fable-5-1` 與 `grok-4.6`**（兩顆都只在正式站，測試站 110 顆裡沒有；對正式站實測，總花費 $0.0099）。①**`grok-4.6` 的值域與 grok-4.3 不同，不能照抄**：`reasoning_effort` 接受 minimal／low／medium／high／xhigh，**`none` 被拒**（3/3 都是 `openai_error`／`bad_response_status_code`——Grok 不列舉合法值，只能逐一試）；4.3 剛好相反（接受 none、拒絕 xhigh）。預設就會推理（reasoning_tokens 71，且不計入 completion_tokens），不回 reasoning_content，**不支援看圖**（n=1）。各檔 reasoning_tokens 各測一次：minimal 52／low 74／medium 97／high 93／xhigh 199——**high 低於 medium，n=1 撐不起「單調遞增」**，只寫值域不寫強弱。②**`claude-fable-5-1`**：`temperature` 被上游拒（`\`temperature\` is deprecated for this model`）→ `no_sampling`；**看得到圖**（1x1 PNG 實測答得出顏色）。③⚠️ **修掉一個會讓新模型整個不能用的既有 bug**：後端**無條件**送 `presence_penalty`／`frequency_penalty`（預設 0.0），而 `grok-4.6` **只要帶了其中任一個就 400**——直接打閘道會通、透過 playground 卻失敗，二分法逐欄位試才找出來。新增 `no_penalties` 旗標閘控（同 `no_sampling` 的作法）。這類「連 0.0 都不收」的模型以後會再出現。④**補上 MCP 欄位守門測試的漏洞**：`test_models_fields_are_known_to_mcp` 先前**只涵蓋 image 與 video**，所以我新增的 `no_penalties` 靜悄悄溜過去了——那正是這條測試該擋下的東西。已把 `text` 加進涵蓋範圍，並把 6 個文字專屬旗標（`no_sampling`／`no_penalties`／`thinking_budget`／`clear_thinking`／`preserve_thinking`／`repetition_penalty`）歸類為內部欄位，理由是 **MCP 目前沒有任何對話／文字工具**，agent 拿不到也用不到；之後若新增 chat 類工具要重新分類（註解已寫進測試）。⑤`grok-4.3` 的 desc 從「最新旗艦」改成「旗艦」（4.6 才是最新）。⑥**順帶發現、未處理**：`claude-sonnet-5` 實測也看得到圖，但 **Claude 家族在 MODELS 裡一顆都沒標 `vision`**——這是既有缺漏（客戶在文字頁籤看不到圖片上傳區）。只有這次新增的 fable-5-1 標了，因為只有它是我實測過的；其餘 11 顆要逐一實測才補，不照家族推。
- feat(video)：**Omni 影片延長改用正解，並開放時長與畫面比例**。**先前的延長實作從頭就走錯**：用 `previous_interaction_id`，那是 Gemini Developer API／`google.genai` SDK 的東西；我們接的是 **Agent Platform**，正解是 `generation_config.video_config.task = "extend"` ＋ 帶**恰好一支**輸入影片。上游那句 `… on this path do not support previous_interaction_id.` 字面上就是實話，「this path」指的就是 Agent Platform——不是模型名、不是 preview 限制、也不是閘道擋的。`_OMNI_EXTEND_ENABLED` 旗標整個移除（它守著的是一個走錯的實作，不是等平台補的功能）。①**權威來源改口**：Omni 的請求規格以 **Agent Platform 自己的 REST 文檔**為準（`docs.cloud.google.com/gemini-enterprise-agent-platform/models/video/` 底下六頁：text／image／first-and-last-frames／references／extend／edit），**不是** Google 部落格的 SDK 範例、也不是模型卡。②**免費探測拿到的完整值域**（送非法值、400、不計費，比文件可靠）：`task` = text_to_video／image_to_video／reference_to_video／edit／extend；`duration` = 3–10 秒；`aspect_ratio` = 16:9／9:16；`resolution` = 360p/720p/1080p/4k；延長要求 `Exactly one input video is required for extend task.`。③**影片餵法**：inline base64（結構驗證通過）；https 網址上游**會去抓**但被我們自家 `robots.txt` 擋（`URL is blocked by robots.txt rules`），所以走 base64。④**UI**：時長滑桿改 3–10 秒（移除 `no_duration`）、比例選單新增資料驅動分支——模型有宣告 `ratios` 就以它為準（Omni 只有兩個值，沿用萬相家族的共用清單會讓客戶選到必被拒的值）。`ratios` 一併加進 `_MCP_CONSTRAINT_FIELDS`（測試 `test_models_fields_are_known_to_mcp` 主動抓到這個新欄位、逼我當下決定要不要讓 agent 看到——這條測試發揮作用了）。⑤**`duration` 已實測**：送 5 拿到 **5.013 秒**（預設約 10.005 秒），費用同步減半。⚠️ 但這兩支是**意外產生的**：我想用 `resolution=9999P` 當哨兵做免費探測，**卻忘了自己那層的 `_omni_resolution()` 會先把非法值濾掉**，請求因此變成合法的預設生成，跑掉兩支、$1.02。教訓：**透過自己的 API 做免費探測前，要確認哨兵值從自己手上到上游之間不會被任何一層改寫**；要探非法值就直接打閘道、繞過自己那層。⑥**仍未實測**：`aspect_ratio`（兩支產出都是 16:9＝預設值，等於沒測，要用 9:16 驗）、`extend`（要量的是「回來的是新的一段還是累積後的完整長度」，回應看不出來；文檔站另提醒**量到的秒數與被收費的秒數不必然相同**，兩者要分開確認，這會影響「最長 40 秒」能不能寫）。
- feat(video)：**Omni 1.1 的解析度補上 1080P 與 4K**（先前只開 360P／720P）。**先前太保守**：發布文章講 1080p／4K 時用的是「upscale」措辭，我判斷未必是同一個欄位的值域就沒列。實際上**送一個非法值，上游會把完整值域列出來**（免費、400、不計費）：`The value '9999p' is not supported for 'response_format.resolution'. Supported values: '360p', '720p', '1080p', '4k'.`——四個都是同一個欄位的合法值。**上游自己列的值域比文件可靠**，而且這種探測不花錢，以後遇到「文件講得含糊、不確定某個值能不能用」的情況應該先做這個，而不是直接放棄。①選單新增 `4K` 選項；`360P` 與 `4K` 的「只有模型明確列出才顯示」規則抽成 `OPT_IN_RESOLUTIONS` 常數（往後再加新解析度選項記得加進去，否則會重演一次「憑空出現在不支援的模型上」）。②`_OMNI_RESOLUTIONS` 擴成四個值、三筆 MODELS 的 `resolutions` 與 desc 一併更新。③已用實際 `/api/models` 資料模擬驗證：omni 1.1 顯示 720P/1080P/360P/4K，其餘模型（wan2.7、veo、seedance、animate）的可見選項與先前完全相同。⚠️ **未實測 1080P／4K 的實際輸出尺寸與費用**——上游列舉證明「值合法」，不證明「產出就是那個尺寸」，也還不知道費用怎麼隨解析度變（已知 360p 19,310 tokens／$0.34、720p ~58,000／$1.02，**不是按像素等比**）。omni 是 token 計價，UI 顯示的 $1.5→$17.5/1M 不會隨解析度變動，所以選 4K 的費用**事前看不出來**。
- feat(video)：**Gemini Omni 場景延長（`previous_interaction_id`）實作完成，但旗標關著——閘道擋下來**。①**後端**：`_generate_omni_video()` 支援 `previous_interaction_id`，並把上游回的 interaction id、模型、解析度、段數存進 `_OMNI_TASK_CACHE`；新增 `POST /api/video/omni/extend`（吃我方的 task_id，取出 interaction id 接著送）；段數上限 `_OMNI_MAX_SEGMENTS = 4`（📄發布文章：10 秒為單位、累計最長 40 秒）——**上限自己擋**，超過之後上游行為未驗證，不要讓使用者用錢去試。②**前端**：完成的影片卡片上多一顆「＋ 延長場景（第 N 段）」，顯示與否由後端回的 `can_extend` 決定、前端不自己推算段數。③⛔ **實測被擋**：用真實 interaction id 送延長，閘道回 400 `gemini-omni-1.1-flash-preview on this path do not support previous_interaction_id.`（不計費；用真實 id 與亂填 id 的訊息完全相同，不是 id 無效）。**這與平台 session 的讀碼結論相反**（他們讀到的是「具名欄位、同一條 marshal 路徑、全程帶著走」），已回報請他們順著錯誤字串找哪一層擋的。對照組：同模型同路徑的 `response_format` 完全正常，所以不是整條 interactions 被鎖。④**因此加了 `_OMNI_EXTEND_ENABLED = False`**：端點與 UI 程式都留著，但按鈕不會出現、端點直接回拒——不讓客戶點到一個必定失敗的功能。閘道支援後**只要把這個旗標改成 True**，不需要其他改動。⑤**順帶修正一條自己的判準**：我先前說「回 `upstream_error` 就代表欄位確實抵達上游」，但這次被擋的錯誤也標著 `upstream_error`，而句型（`on this path do not support…`）不像 Google 的文案（Google 是 `The 'type' parameter is required at 'response_format'` 那種、還會列出合法值域）。**錯誤的內容比它的 type 標籤可靠**，看到 upstream_error 不能直接斷定已抵達上游。⑥**後續（同日）**：平台 session 逐路徑重查，**我方程式碼沒有任何一處會擋或改寫 `previous_interaction_id`**，且 `RelayErrorHandler` 只在上游沒帶 `type` 時補 `upstream_error` 當預設值、**訊息文字是直接從上游 JSON 的 `error.message` 解出來的、我們不重寫**——repo 裡也找不到能拼出那句話的樣板。他們提出的假說「延長請求要跟 `response_format` 一起送」我實測**排除**（兩者一起送，錯誤一字不差，不計費）。另外發現 Google 文章的範例用的是 **GA 名稱 `gemini-omni-1.1-flash`**、我們平台掛的是 `-preview`，送 GA 名稱會被我們自己的 distributor 擋（`model_not_found`／503／中文訊息／帶 request id——**這剛好是「我方錯誤」的對照樣本**，文體與那句英文完全不同）。目前判斷：**preview deployment 不開放這個能力**，「on this path」指的可能就是這件事；但這是推論，要證實得能用不帶 `-preview` 的模型打一次。
- feat(video)：**Gemini Omni 1.1 開放輸出解析度（360P／720P）**——推翻同日稍早「上游沒有這個參數」的錯誤結論。**先前錯在哪**：只看參考庫 §2.3.41 的模型卡摘要（那張「參數預設值」表只列 `temperature`／`topP`／`candidateCount`），把「表上沒有」當成「參數不存在」，還跟文檔站宣稱那是「結構事實」。實際上 Google 發布文章有官方範例 `response_format={"resolution": "360p"}`。**模型卡不是參數規格書，不能拿它當否定證據。**①**免費探測先確認透傳**：漏 `type` 時上游回 `The 'type' parameter is required at 'response_format'`（400、不計費），送非法 type 會回**合法值清單**（boolean/video/number/integer/object/image/text/string/array/audio）——這兩個錯誤都來自 `upstream_error`，證明閘道確實把 `response_format` 原樣送到上游（與平台 session 讀碼結論一致：`dto/interactions.go` 用 `json.RawMessage` 承接，gemini channel 沒實作 `interactionsRequestConverter` 故走 passthrough）。正確形狀是 `{"type": "video", "resolution": "360p"}`，SDK 範例省略 `type` 是因為 SDK 會補。②**實測量到尺寸**（正式站，走 playground）：`640×360`、`10.005` 秒、含音軌、**$0.3429**。先前 720p 那支是 `1280×720`／10.005 秒／$1.017——**時長相同**，所以費用降到 0.333 倍確實來自解析度而非影片變短（只看 token 數分不出這兩者，19,310 tokens 同時能解釋成「360p×10 秒」與「720p×3.33 秒」，必須量檔案）。③**後端閘門**：`_omni_resolution()` 只有在 MODELS 明確給了 `resolutions` 的型號才送 `response_format`——影片頁籤的解析度欄位**即使被 `no_resolution` 隱藏也仍會帶預設值 720P 送過來**，若只檢查值是否合法就送，會在驗證完成前把行為從「不帶欄位、走上游預設」悄悄改成「明確指定 720p」。④**1080p／4K 不列**：發布文章的措辭是「upscale」，未必是同一個欄位的值域，沒驗過就不列（列出來卻做不到的選項比沒有更糟）。⑤**選單新增 360P 選項的副作用已擋掉**：沒有 `resolutions` 清單的影片模型走「顯示全部選項」那條路，新選項會憑空出現在 `seedance-1.5-pro`／`seedance-2.0` 等 4 個型號上；改成 **360P 一律只在模型明確列出時才顯示**，其餘模型的可見選項與先前完全相同（已用實際 `/api/models` 資料模擬驗證）。
- docs(models)：**六筆 Gemini Omni 的 desc 補上「長度與解析度由模型決定」**（Levi 回報「omni 1.1 不能選擇輸出解析度」）。**行為本身是對的、不是 bug**——參考庫 §2.3.41（Google 官方模型卡原文）列出的參數只有 `temperature`／`topP`／`candidateCount`，**沒有任何輸出解析度或時長欄位**；文檔裡「解析度 360p／720p／1080p／4k」那句是講**影片輸入**的限制，而官方定價註腳寫「輸出方面**每秒 720p 影片**（含音訊）計 5,792 tokens」，等於定價本身就假設輸出是 720p，與我們實測到的 1280×720 一致。問題出在**介面沒講清楚**：選單只寫「最長約 10 秒」，沒提解析度也不能選，使用者會以為是我們漏做。六筆（1.1 與無版號各三個型別）一起改。
- feat(canvas)：**灰模節點支援「用提示詞生成場景」**（Levi 需求：用 prompt 產灰模內容，再拿去參考生成）。節點多了提示詞欄位＋文字模型選擇＋「✦ 生成場景」鈕，呼叫 `/api/text/generate`，系統提示詞把量體 DSL 的文法、座標系（Y 上、地面 Y=0、相機看向 -Z、y 是量體中心）與一個完整範例寫死進去——這個格式是我們自己定的，模型不可能知道，而且例子比規則管用。①**產出一律再過 `parseGreyboxSpec()` 過濾**：模型很常多寫解釋文字或包 markdown 圍欄，直接塞進場景會整批解析失敗；過濾後只留合法的行並回報「略過 N 行非量體內容」。②**「場景描述／提示詞」輸入插槽現在吃兩種東西**：解析得出量體就直接當場景，解析不出來就當提示詞帶進欄位——**但不自動呼叫模型**，`onExecute` 每一幀都會跑，自動呼叫等於無限計費。③**新增 `_autoFrame()`**：AI 產的場景尺度不固定（3 公尺的房間到 24 公尺的高樓都有），固定相機距離常常框不下；生成後依包圍盒自動調距離與高度，**只在生成之後跑、不在每次重建時跑**，否則使用者手動調的滑桿會被蓋掉。④**驗證**：用 canvas.js 裡**同一份**系統提示詞與**同一個**解析器（從原始碼抽出來跑）對正式站的 `qwen3.5-flash` 實測——「黃昏的街口，左邊一棟高樓、右邊兩棟矮樓，路中央停一台車，遠處有一個人站著」回傳 6 行乾淨 DSL、全數保留，語意也對（左高樓 24 高、右兩棟 12 高、中央扁盒＝車、遠處圓柱＋球＝人）；另用一段刻意夾雜中文說明＋markdown 圍欄＋註解的假回應測過濾器，正確保留 4 個量體、略過 5 行。把那份生成結果貼進節點，瀏覽器渲染正常。**未實測**：按鈕點下去到填回 textarea 這一段（需要真金鑰在瀏覽器裡，不想把正式站金鑰放進測試流程）——兩端都各自驗過，中間是十行直線程式碼；`_autoFrame()` 的取景數學是手算核對（該場景會落到距離 40／高度 11），沒有跑起來看。
- feat(canvas)：**3D 灰模節點（Greybox）**——在瀏覽器裡搭沒有材質的粗塊場景、跑一段相機運鏡，輸出灰模影片當影片模型的**運鏡參考**（接 MuleAI 節點的「參考影片」插槽）。**不在伺服器跑 Blender**：這條工作流真正需要的只有「粗塊體積 ＋ 一條相機軌跡」，headless Blender 要 +1GB image、GPU、任務佇列與 bpy 沙箱，成本全花在無關的地方。①**場景用文字描述**（`box 寬 高 深 at x y z`／`cyl`／`cone`／`sphere`，可加 `rot`），一行一個量體——不必寫拖拉 gizmo（那是一整套 UI），而且這個格式可以由上游文字節點／AI 直接產生（節點有「場景描述」輸入插槽）。四個範本：街道／室內／人物舞台／空白。解析不了的行**顯示成錯誤**不靜默跳過（靜默跳過的話使用者會以為那行有生效、只是模型沒照做）。②**七種運鏡**（推近／拉遠／左右環繞／升降／穿越／橫搖）＋相機距離與高度滑桿；五種畫面比例。③**影片編碼走 WebCodecs ＋ mp4-muxer**——上游只收 mp4/mov，`MediaRecorder` 只給 webm 送不進去，所以不能用它；不支援 WebCodecs 的瀏覽器只提供單格輸出。④**踩到的坑**：`wireConfigOverlay()` 會把 `.cv-controls` 整塊搬去「選取時才顯示的設定浮層」，第一版把所有控制項都放在裡面，結果重新整理後**節點本體整個空掉**（新增當下因為節點是選取狀態才看得到）。改成本體＝預覽畫布＋生成鈕＋狀態、設定＝浮層，跟 `PoseNode` 同一套。⑤**瀏覽器實測**（本機，用假金鑰通過登入閘——這個節點不打任何 API）：three/mp4-muxer/VideoEncoder 都載入、節點註冊成功、渲染出 832×480 全灰階畫面（12 個灰階值，非空白）、3 秒影片編碼 1 秒完成、編出來的 blob 標頭是 `ftypisom`（合法 mp4）、重新整理後節點與場景正確還原且 blob 網址失效的提示正常。
- 記錄：**`reference_videos` 三項待實測全部收掉（正式站實測）**。①**簽名網址／抓取**：部署後的 playground 送參考影片，任務跑到 `completed`——先前只知道「網址產得出來」，現在確認**上游真的抓得到**（失敗長相會是 `code 4002 / Failed to download file`，沒有出現）。②**「Video 1」指涉＝傳遞運鏡，不傳遞內容**：第一支用「跟參考影片相同內容」的提示詞測，成品與參考幾乎一模一樣——**這支分不出「跟著運鏡」還是「整支複製」，不能拿來下結論**；改用完全不同的內容（橘貓＋明亮房間）＋同一支街道參考影片再測一次，產出是貓、但運鏡照抄了推近，**這才證明傳遞的是運鏡不是內容**。樣本限制照實記：n=1、單一運鏡（推近）、單一模型。③**`w3.0-video-prime-pro-spicy` 計價**：1080p／2 秒實測 $0.52，與 `0.26 × 2` 完全相符，四顆別名全部驗畢。本輪費用合計約 $0.82。
- feat(muleai/canvas)：**w3.0 的 reference 模式（主測試台）＋ Canvas 節點補齊**。①**後端**：`/api/muleai/generate` 的 w3 分支加上 `last_frame` 與 `reference_image_1..10`／`reference_video_1..5`／`reference_audio_1..5`；**兩種素材模式互斥的檢查放在最前面**（keyframe 與 reference 混送閘道會 422），而且刻意擋在**上傳影片到雲端儲存之前**——那一步會真的寫檔／打雲端 API，讓一個必定失敗的請求走到那裡沒有意義。參考圖走 base64，參考影片與音訊**只收 http(s) URL**（送 data URI 會被閘道以「accepts public http/https URLs only」擋下），所以先過 `_upload_video_for_url()`／`_upload_audio_for_url()` 取得可下載網址。新增 `_collect_uploads()`：**中間有缺號也照收**，不在第一個空號就停（前端刪掉中間一張後重新編號是常見情況，早停會靜默少送素材）。②**前端**：素材模式下拉（首尾幀／參考素材）、三種素材的多檔清單與上限（10／5／5）、以及**時長檢查**——參考影片總長 ≤15 秒、且「輸入總長 ＋ 輸出時長 ≤30 秒」。最後那條**閘道不擋**（要先抓回素材才知道長度），由上游拒絕，所以用 `<video>.duration` 在送出前自己算；讀不到長度的檔案一律當「不確定」而不是 0，否則總長會被低估、檢查形同虛設。③**Canvas**：MuleAI 節點補上畫面比例、智能時長、以及**「參考影片」輸入插槽**（接上游影片節點的輸出，提示詞裡用「Video 1」指涉）；同時接了參考影片與首幀圖時，前端會丟掉首幀圖並明說（後端也擋）。舊存檔只有 3 個插槽，`onConfigure` 補回第 4 個。④**驗證**：互斥檢查在本機擋下（400，沒打上游）、參考圖路徑送得到上游（拿到解析度 422）、尾幀圖路徑同樣送得到、舊的 `wan2.7-i2v-spicy` 不受影響；`_upload_video_for_url` 在本機無雲端憑證時回出可讀的錯誤訊息。**未驗證**：`reference_videos`／`reference_audios` 的實際生成（需要可公開下載的網址，本機環境拿不到，只能在部署後驗）、「Video 1」指涉行為、新 UI 沒跑瀏覽器冒煙。**本輪沒有做真實生成**（Levi 指示只寫程式、免費 422 探測為止）。

## 2026-09-01

- fix(canvas)：**修掉今天上架 w3.0 時一併帶出的 Canvas 回歸**（已部署，故列為修正）。`MuleAiGenNode` 的模型分類是**寫死的 id 比對**（`_isVideo()` 只認 `wan2.7-i2v-spicy`），而四顆 w3.0 被排在 `MODELS["muleai"]` 第一個、成為節點的預設模型——結果 `_isVideo()` 回 false：輸出插槽變成 `image`、不送解析度與時長、強制要求連來源圖（w3.0 首幀圖其實選填），最後從 `result.images[0]` 取結果而 w3.0 回的是 `videos[]`，拿到 `undefined`。改成**讀 MODELS 的中繼資料**（`type`／`image_optional`／`resolutions`／`min_dur`／`max_dur`）：影片判定、首幀圖選填、解析度檔次與時長上限逐顆套用，之後再加同類模型不必再改這裡。舊的四顆沒有這些欄位，走原本的預設值不受影響。**驗證程度照實記**：語法檢查＋`/api/models` 實際回傳的欄位逐顆核對過（四顆 type=video／image_optional=True／檔次與 2–30 秒都正確），但**沒有跑瀏覽器冒煙測試**——Canvas 要有效金鑰才載得到模型清單，不想把正式站金鑰放進測試流程。Canvas 的 w3.0 仍**沒有** ratio 與智能時長（沿用上游預設 adaptive），reference 模式也還沒做。
- docs(CLAUDE.md)：**新增「必做（三）：文檔的效果展示素材由這裡產」**（Levi 裁示）。文檔站不自己呼叫付費 API、也不持有金鑰，示範圖／影片一律由 playground 產：用該類別**最新型號**、主題用貓／狗或人物（看得出毛髮五官與動作連貫性，風景難分辨好壞）、影片要有運鏡描述、**只跑必要次數不為挑結果重跑**、交付原始檔絕對路徑（後製是文檔站的事）並附模型 id／網關／規格／提示詞。⚠️ 條文明訂**每一筆素材費用都要先取得 Levi 同意**——文檔 session 轉述的「Levi 核准了」不算，跨 session 轉述不能當成本人授權。
- 記錄：**產出第一筆文檔素材**——`gemini-omni-1.1-flash-preview`，**正式站**，一支貓跳上窗台＋緩推微仰運鏡；產出 1280×720／10.005 秒／含音軌／5.5MB，費用 **$1.0178**，跑一次沒重跑。路徑已交文檔站。順帶第二個資料點：1.1 這顆目前 2/2 都有音軌（測試站＋正式站各一），但沒有版號的那顆仍一次都沒跑過，所以文檔站「是否含音軌由模型決定」的保守寫法仍是唯一撐得住的版本。
- docs(CLAUDE.md)：**流程調整（Levi 裁示）**。①**網關每次都要明確問**——先前那條只寫「必問」，但實務上使用者訊息裡的措辭（「平台測已經上了」「先用官方名測試」）讀起來像已經指定了，這次就是照那個推斷直接動手。改成明寫「使用者的訊息裡好像已經暗示了，不算已確認」，用 `AskUserQuestion` 問一次；成本不對稱——問一次 vs 整輪測試打到沒有該模型的環境。②**不分網關都要寫文檔**——測試站驗完也要通知文檔 session，不是等上了正式站才寫（延後的結果是驗證細節在記憶裡過期），送出時要標明測試所在的網關。③補上一條環境陷阱：本機常有多個 dev server 常駐且指向正式站，`uvicorn` 綁不到 port 時只把 `Address already in use` 印進 log 就退出，而 `curl localhost:5050` 因為舊的那個還活著照樣有回應，看起來像啟動成功——實際上測的是別人那台。要另起沒被占用的 port，並在花錢測試前先用 `/api/pricing` 確認網關身分。
- feat(video)：**新增 `gemini-omni-1.1-flash-preview`**（t2v／i2v／r2v 三個任務型別，與既有 `gemini-omni-flash-preview` 同家族、同價、同走 `/v1beta/interactions`，模型自行決定長度與解析度、聲音隨影片產出，故沿用 `no_duration`／`no_resolution` 旗標；`_INTERACTIONS_VIDEO_MODELS` 一併加入，否則會被當成 `/v1/videos` 的非同步任務走錯路徑）。**實測（2026-09-01，測試網關）**：t2v 一支——產出 1280×720、時長 `10.005` 秒、**含音軌**（omni 的聲音是隨影片產出的，不需另外開）；計費 delta $1.017（token 計價，輸入 $1.50/1M、輸出 $17.50/1M，與參考庫 §2.3.41 的牌價一致）。i2v／r2v 兩個型別**沿用同家族既有實作、未逐一實測**，這點照實記。
- feat(muleai)：**w3.0 影片四顆進 NenAI Spicy 頁籤**（`w3.0-video-spicy`／`-pro-spicy`／`-prime-spicy`／`-prime-pro-spicy`，Levi 在閘道端做模型重定向，playground 用 `-spicy` 名字）。**上架過程踩到的坑：重定向做完後四顆一律回 `model_price_error:「模型 w3.0-video-spicy 倍率或价格未配置」`**——`/api/pricing` 上這四個別名是 `quota_type=0 / model_price=0 / model_ratio=37.5`，不帶 `-spicy` 的本尊才是 `quota_type=1` 且價格正確。根因（閘道 session 確認）：adaptor 的 `lookupW3VideoSpec()` 用 `strings.HasPrefix`，所以別名的**路由與參數校驗**全部自動跟上（我們實測到別名也吃到各自的檔次白名單就是這個），但 `GetModelPrice()` 是**精確查表**、`defaultModelPrice` 只登記官方模型名 → 別名查不到 → 掉到 token 兜底。**別名屬於部署設定，價格要在後台設、不寫進程式**（Levi 已補上，設完當下生效）。**通則：新增任何模型別名時，路由通 ≠ 計價通，兩邊都要看。**①**MODELS 新增四筆**：解析度檔次逐顆不同（base/prime 是 480p/720p/1080p，pro/prime-pro 是 1080p/2k/4k，**小寫**，與既有影片模型的 480P/720P/1080P 不同一套，閘道是字面比對）、duration 2–30、ratio 六值、smart_duration、audio、首幀圖選填。②**後端 `/api/muleai/generate` 新增 w3 分支**：resolution／ratio／prompt_extend／seed／first_frame 走 `metadata`，duration 走頂層；首幀圖選填（沒圖就是文生影片）。**順手修掉一個潛在 bug**：Spicy 這條路徑的三個上游網址（generate／status／debug）原本**寫死 `https://nen.com.tw`**，`NENAI_BASE` 對它完全無效——新模型要先對測試網關驗證時會直接打到正式站（那裡還沒有該模型），改成吃 `NENAI_V1`。③**前端**：解析度／比例／時長上限全部由 MODELS 資料驅動（不寫死），智能時長開關（開啟時不顯示估時，因為閘道只預扣 1 秒押金、事後拿不到成片秒數），w3 的配音是純開關故不露出音軌上傳區，首幀圖標示改「選填」。④**價格顯示**：四顆加進 `_VIDEO_SEC_PRICE`（官方每秒單價），否則照閘道的 token 計價會顯示成 `$75→$75/1M`；`videoPerSecondPrice` 新增大小寫不敏感比對與 `fallbackCheapest`（下拉選單固定拿 720P 當基準，但 pro 檔次沒有 720p，落空會退回無意義的 token 單價——改成退到該模型最便宜的一檔，且**顯示實際查到的檔次標籤**，不顯示該模型根本沒有的 720P）；`formatPriceSuffix` 改讀 Spicy 頁籤自己的解析度選單（先前固定讀影片頁籤的 `videoResolution`，Spicy 選什麼都不會反映在價格上）。⑤**參數約束全部由免費 422 探測實證**（送非法值觸發驗證、不產生內容不計費，2026-09-01 對測試網關）：各顆解析度白名單、duration 2–30 或 -1、ratio 六值、`first_frame/last_frame` 與 `reference_*` 互斥、prompt／keyframe／reference 三者至少要有一個、reference_images 上限 10、reference_videos 只收 http(s) URL（base64 被拒）、`seconds` 欄位在閘道是字串型別（送數字會 400，所以不要用）。**實際生成與計費核對（2026-09-01，測試網關）**：以官方名 `w3.0-video` 先驗一輪——480p/2 秒產出 832×480、時長 2.000 秒、h264、**無音軌**（我們送 `audio=false`，上游確實照做，與 Veo「省略就給聲音」的行為不同）；720p/2 秒的計費 delta 實測 **$0.20**，正好 `每秒單價 0.10 × 2 秒`，檔次倍率 2.0 與對帳公式 `model_price × 秒數 × 檔次倍率 × 分組倍率` 完全吻合。**四顆 `-spicy` 別名逐一補測（Levi 補上後台價格設定後）**：各生一支 2 秒、各自最便宜的檔次，逐顆量計費 delta——`w3.0-video-spicy` 480p $0.10（預期 $0.10）✅、`w3.0-video-prime-spicy` 480p $0.136（預期 $0.136）✅、`w3.0-video-pro-spicy` 1080p $0.36（預期 $0.36）✅，三顆分毫不差，**別名的計價確實走本尊的每秒單價**。產出規格：480p → 832×480、1080p → 1904×1104，時長皆 `2.000` 秒，皆無音軌。**第四顆 `w3.0-video-prime-pro-spicy` 送不出去**，閘道回 `do_request_failed: new request failed: parse " https://.../... first path segment in URL cannot contain colon`——注意錯誤訊息裡的網址**開頭多一個空格**，是該顆的上游網址設定混進了空白字元，Go 的 `url.Parse` 因此失敗。**已隔離出這不是別名的問題**：改用官方名 `w3.0-video-prime-pro` 直接打閘道，錯誤一字不差，所以是那一顆的渠道／模型部署設定要修（把網址前後的空白去掉），與重定向無關。修好後要補測這一顆。**仍未實測**：`reference_videos` 吃簽名網址、「Video 1」指涉行為。測試 80 全過。
- chore(deploy)：**Cloud Run 修訂版本自動清理，只留最新兩個**。Cloud Run 沒有「保留 N 個修訂版本」的設定，每次部署只增不減——正式服務 `nenai-testing-platform` 已累積到 147 個，先手動清成 2 個（保留 `00149-4jm` 現行版與 `00148-m82` 上一版，刪除 `00003-772`～`00147-x8s` 共 145 個；服務 Ready、流量 100% 在 00149 全程未受影響）。再在 `.github/workflows/deploy-cloud-run.yml` 加一步 `Prune old revisions (keep latest 2)`，往後每次部署自動收。兩層保護：①先取 `status.traffic[].revisionName` 把**任何正在吃流量的**修訂版本加進保留集（含 traffic tag），再併上依建立時間最新的兩個——順序不能顛倒，否則手動 `update-traffic` 回滾到舊版後，那個正在服務的舊版會被當成過期版刪掉；②刪除失敗只印訊息不讓 workflow 失敗（部署已成功，清理沒清乾淨是下次的事）。保留集與刪除清單的計算已對正式服務唯讀試跑驗證（keep 正確、prune 為 0）。**同批修正資源調度旗標**（Levi 確認主控台的 1/20 是他設的，以主控台為準）：workflow 原本用 `--min-instances=0 --max-instances=5`，那是**修訂版本層級**（`autoscaling.knative.dev/maxScale`，寫死在每個新修訂版本上）；主控台「資源調度：自動，下限 1／上限 20」是**服務層級**（`run.googleapis.com/minScale`／`maxScale`，可不建新修訂版本就改）。兩者同時存在時以較嚴的一邊生效——實際狀態就是服務允許 20、修訂版本 00149 卻被壓在 5。改為 `--min=1 --max=20`（服務層級）＋ `--min-instances=default --max-instances=default`（明確清掉修訂版本層級的設定，否則它會被新修訂版本繼承並悄悄蓋過服務層級）。其餘旗標（memory 1Gi／cpu 1／gen2／timeout 300）與線上一致；ingress `internal-and-cloud-load-balancing`、startup-cpu-boost、concurrency 80 未在 workflow 指定，由既有服務繼承，不受影響。

## 2026-08-29

- feat(canvas)：**兩個新功能（Levi 指定）**。①**視覺化尺寸選擇器**——圖片節點的尺寸下拉改成「比例鈕（帶方向圖示）＋同比例多尺寸 pills」（依 MODELS 的 sizes 依比例分組，1360x768 這類近似比例做 2% 內吸附顯示 16:9；選項與送出值與舊下拉完全相同）；Gemini 圖片的比例下拉同樣改比例鈕；影片／影片編輯／MuleAI 節點的解析度下拉改 pills（共用 makeResPicker，applyVideoLimits 對接）；影片節點新增畫面比例鈕（僅 wan3.0/wan2.7/happyhorse 的 t2v/r2v，i2v 依 no_ratio 隱藏，wan3.0 預設 adaptive——Canvas 首次補上 ratio 送出，值域與主測試台一致）。CSS 注意：面板按鈕通用樣式是 display:block+width:100%，pills 要以同特定度蓋回 inline-flex（第一版就是被這條蓋掉變整行直排）。Playwright 冒煙：比例鈕/pills 渲染、點 16:9 尺寸跟著切、影片節點三檔 pills＋自動比例正確。②**姿勢節點（nenai/pose）**——OpenPose 風格骨架編輯器：五個範本（站立/單手高舉/奔跑/坐姿/四足動物）、迷你畫布拖曳關節（pointer capture，stopPropagation 防 LiteGraph 攔截）、輸出 768×1024 骨架圖 dataURL 接圖片節點當參考圖。**上架前先實測遵循度（Levi 裁示「效果夠好才做」，費用已核可）**：wan2.7-image ×3＋qwen-image-3.0 ×3、同一張不對稱姿勢骨架（一臂高舉/一臂平伸/雙腿張開）、提示詞只說「依照骨架圖」**不用文字描述姿勢**（否則測到的是文字遵循）——**6/6 全數遵循**，連骨架手腕折角都重現。樣本限制照實記：單一姿勢、單人、攝影棚場景；複雜姿勢/多人未測，節點文案維持「參考性質」語氣不承諾嚴格遵循（上游無 ControlNet 條件化通道，本質是軟引導）。實測腳本與六張輸出在 scratchpad（不進 repo）。註冊三處＋工具列選單都補（NODE_TYPE_LABELS/NODE_MENU_TYPES/registerNodeTypes/canvas.html——兩張表都要加的坑已有註解）。測試 80 全過。**版面修正（Levi 回饋「編輯畫面不好操作、畫面比例不好」）**：編輯畫布移進節點本體（隨時可拖、不用先選取），移除與輸出預覽的重複顯示；新增畫布比例切換（直式 3:4／橫式 4:3／方形 1:1，切換時關節等比例映射保留姿勢形狀），四足動物範本自動用橫式；輸出解析度隨比例（768×1024／1024×768／1024×1024）。Playwright 複驗：畫布尺寸隨範本切換、拖曳關節正常。**範本擴充（Levi 指示「新增更多姿勢」）**：5 個 → 15 個——人形 12（站立/揮手/單手高舉/雙手高舉/T 字姿/奔跑/跳躍/出拳/踢腿/蹲姿/坐姿/躺臥，躺臥用橫式畫布）＋四足 3（站立/奔跑/坐姿）。Playwright 逐一渲染 15 格總覽圖檢查全數正確。

## 2026-08-28

- 記錄：**glm-5.2 reasoning_effort 分級實測（Levi 核可費用，正式站 13 次呼叫）——官方映射屬實，實效三態**。同題多步推理、enable_thinking=true，low/high/xhigh/max 各 3 次＋minimal 1 次。reasoning_tokens：low {24,435,678}／high {213,239,956}／xhigh {1383,1691,1955}／max {499,1562,2163}——相鄰檔（low vs high、xhigh vs max）中位數皆倒序、區間重疊＝不可分；兩群中位數差 4~5 倍＝群結構清楚；「四檔單調」假說在兩處非單調、不成立。minimal 未被拒：200、usage 無 reasoning_tokens 欄位（prompt_tokens 133 vs 其他檔 139，疑思考鷹架不掛）＝行為是關思考非值域外。與 08-25 舊資料不矛盾——舊的「七段全有」只證枚舉可送。樣本警語：每檔 n=3、檔內變異最大 28 倍，只撐得住「分群／不可分／minimal 不報錯」三句，不撐任何具體數值。結果已交平台（reference §2.3.24）與文檔站（glm/chat 頁）。playground UI 維持七段下拉（每個值都合法可送），是否改標「實效三檔」待裁。
- feat(text)：**百煉 Chat 方言參數九項解凍後接入四顆＋top_p「模型預設」**（閘道 PR #65（5339e2f93）2026-08-28 12:17 部署，平台實測 thinking_budget=16 → reasoning_tokens=16 分毫吻合）。①`thinking_budget` 數字輸入框（不用滑桿——官方未公布值域不設上下限；tooltip 警示過小預算會截斷思考導致輸出異常，平台實測 16 會把 </think> 漏進正文並開始重複，未測出門檻故不寫具體數字）；②`clear_thinking` 僅 GLM 5.x、③`preserve_thinking` 僅 qwen3.7/3.6 系——兩顆分家族獨立顯示（平台要求不做三態選擇器），三值下拉（預設=不送/開/關，顯式 false 照送、Rule 6 語意）；④`repetition_penalty` 百煉文字全家（19 模型）。四顆均 MODELS 旗標驅動＋後端依旗標閘控（不支援的模型帶了也不送）。⑤top_p 改 Optional：UI 勾「模型預設」整個不送；TextGenerateRequest 預設從 0.8 改 None——閘道曾把未帶的 top_p 硬補 0.001（近貪婪取樣，text.go:31），修掉後「沒填就不送」才正確；Canvas 等未帶 top_p 的呼叫方行為隨之從「補 0.8」變「交模型預設」（刻意，不替客戶端補預設值）。**不接的四項與理由**：tool_stream（文字分頁無工具呼叫 UI）、search_options/enable_code_interpreter（適用模型未查證，不在 UI 斷言）、messages[].partial（前綴續寫是新功能非參數放行，要做另議）。測試 80 全過（新增適用集合鎖定＋top_p Optional 防回歸）。
- feat(video/voice)：**接入閘道 abc0a8b7b 部署的四項新能力**（Levi 裁示免冒煙直接上——平台已依官方文件逐項調整；正式站 start_time 2026-08-28 04:17 UTC 驗證部署）。①wan2.7-i2v 口型同步：i2v_modes 開放 first_frame_audio（UI「首幀＋配音」模式與後端 audio_url 管線本來就在，當年因上游不讀而藏起來，守門測試如預期發動後更新）；②wan2.6-r2v-flash 無聲半價：audio False→True（r2v 端點的 _apply_audio_flag 管線現成）；③wan2.6 系 t2v/i2v/r2v 運鏡模式：新 shot_type 旗標＋UI 下拉（自動/單鏡頭/多鏡頭）＋三端點旗標閘控送扁平 metadata.shot_type；④ASR 熱詞/語言：voiceAsrLangHints（逗號分隔→JSON 陣列，最多 4）、voiceAsrVocab（每行一詞可加權重→JSON 物件，未標權重預設 4）、vocabulary_id，兩個 ASR 端點都透傳（sample_rate 後端也透傳但**不做 UI**——音檔取樣率是檔案固有屬性，讓使用者手填只會填錯）。未接：reference_voice 與 wan2.7-r2v 參考影片（未列入本輪裁示，待排）。
- fix(ui)：**前端參數稽核六條修正**（Levi 指示全修；對照 reference 官方權威檔）。①wan3.0 全系＋prime 加 no_negative_prompt——官方無此參數（P2-13，閘道轉發上游忽略），死控制項收起；②wan2.7-videoedit 時長改官方語意——min_dur 0（0=保留原長，durHintZero 提示既有機制）、max_dur 15→10、滑桿與送出對 1 進位到 2（非法值）；後端本來就「0 不送」；③happyhorse-1.0-video-edit 加 no_duration（官方無此參數）——並把 no_duration/no_resolution 拆成兩個獨立旗標（gemini-omni 兩者皆無、happyhorse vedit 有 resolution 沒 duration，原本共用一個旗標會把解析度一起藏掉）；④wan2.6-r2v/r2v-flash 加 no_prompt_extend（官方 r2v 無此參數）；⑤同兩模型 max_dur 15→10（官方 [2,10]）；⑥Claude 全系文字條目加 no_sampling——後端本就不送 temperature/top_p（Bedrock 限制），UI 滑桿同步收起。稽核第 1 條（智能時長 -1）證實無問題：平台讀碼確認 -1 明文豁免＋原樣轉發＋押 1 秒結算，reference 那格「未支援」是過時條目（平台已更正 6114eee）。順帶發現待辦：r2v 端點不送 negative_prompt（官方 r2v 有），沿革是早期「r2v 不讀」的結論，input 層修復後可重評，未列本輪。測試 78 全過（兩條守門測試如預期發動後更新）。**後記（文檔站抓到轉述矛盾後釐清）**：shot_type 的「需 prompt_extend=true」前置條件只出現在官方 t2v 頁；r2v 頁的 parameters 有 shot_type 但無 prompt_extend——兩者並不衝突（r2v 的 shot_type 沒有該前置條件）。運鏡 tooltip 已改為「文生／圖生影片需搭配 Prompt Extend 開啟」。
- fix(video)：**對齊閘道 PR #64（abc0a8b7b）的阿里影片參數收緊，防止客戶收到 422**。閘道改為「顯式 resolution 不在合法檔位→預扣費前 422」「ratio 只在官方支援的接口轉發」。對照 reference §2.3.13~2.3.16 修三處：①**wan2.7 全系與 wan2.6 全系補 `resolutions: [720P, 1080P]`**——官方兩檔而已，我們的 UI 一直開放 480P（此前被閘道靜默按最便宜檔計費後轉發，現在會 422）；happyhorse-1.0 系同兩檔（官方三檔未分版本，但閘道依定價頁只為 1.1 登記 480P、1.0 送 480P 刻意 422），happyhorse-1.1 系三檔全開；vedit 兩條目補兩檔白名單。②新增 `no_ratio` 旗標（wan2.7-i2v／happyhorse 兩代 i2v／happyhorse-1.0-video-edit——官方無 ratio、比例跟隨輸入素材）：前端 `syncVidRatio` 藏下拉、後端 i2v/vedit 端點剝除（防 Canvas 與直接呼叫方）、衍生集合 `_VIDEO_NO_RATIO`。③ASR 的 `language` 表單欄位行為變更查證**零影響**——我們從未送過該欄位。MCP 自動受益（resolution 預檢 valid_values 隨 MODELS 更新）；`no_ratio` 歸入 MCP 守門測試 non_constraint（MCP 工具本無 ratio 參數）。補兩條鎖定測試（白名單逐條＋no_ratio 集合），78 全過；spicy 分頁本就只有 720P/1080P 不用動。**同批的新能力（wan2.7-i2v 口型同步 audio_url、r2v reference_voice、wan2.6 shot_type、r2v-flash 無聲半價、ASR 熱詞/語言指定）未實作**——等平台確認正式站部署後再向 Levi 提優先序。**部署狀態（平台回覆確認）**：abc0a8b7b 只合併進 main、**正式站尚未部署**（/api/status start_time 08-26 19:03 早於合併時間，文檔站驗證）；我方三項收緊修正**不受順序影響、可先上**（對舊 binary 也成立，先部署反而消掉 422 窗口），新能力等平台的部署通知再接。**待同步提醒**：happyhorse-1.0 的 480P 若日後平台確認定價放行（P2-14 殘留），MODELS 白名單要跟著開，平台會在通知裡點名。

## 2026-08-27

- docs：**README.md 全面重寫為產品文件**（Levi 指示：不要測試報告式說明，只留運作邏輯、模型列表、基礎環境架構示意、專案功能與產品介紹）。從 692 行縮到約 260 行：移除所有實測敘事、勘誤紀錄、參數探測表格與證據等級標註（這些仍在 update.md／memory.md／git 歷史，README 開頭加了指引）；新增「運作邏輯」六點（金鑰不落地、格式轉譯、同步/task 雙模式、參數誠實原則、產出不上雲政策、計價快取）與 ASCII 架構示意圖（使用者端 → Playground → 網關 → 上游供應商，含儲存分支）；模型列表改為分類總覽表（能力描述維持既有驗證過的中性寫法，不新增未驗主張）；MCP 七工具首次寫進 README。部署、雲端儲存設定、依賴套件表保留並精簡。

## 2026-08-26

- 記錄：**百煉 Chat 方言八參數的 UI 全部凍結——參數尚未上線正式站**。平台 session 通知閘道已能轉發 thinking_budget／clear_thinking／preserve_thinking／tool_stream／repetition_penalty／search_options／enable_code_interpreter／messages[].partial（權威 reference §2.3.24），playground 已完成範圍設計（本輪只做適用清單明確的四顆：thinking_budget/clear_thinking/preserve_thinking/repetition_penalty；tool_stream 我方無工具呼叫 UI、search_options/enable_code_interpreter 適用模型未查證、partial 屬新 UX——四項不做）。**動手前平台補充關鍵前提：這批修改只在 feat/carrothub-channel-support 分支未提交，origin/main 與正式站都沒有**——現在接 UI 就是「填了被靜默丟棄」，與文檔站同步凍結，等平台回報合併＋部署後再接。屆時實作要點（平台更正，已記）：clear_thinking（GLM 5.x）與 preserve_thinking（qwen3.7/3.6 系＋qwen3.8-max 預設 true）是**兩顆給不同家族的參數**、按模型分別顯示、不做三態選擇器；thinking_budget **官方沒給值域**（「1～32768、預設 4000」是第三方流傳），不做 slider 上下限。
- 記錄：**top_p「不指定」選項也一併等部署，平台「先改不會錯」的說法有疑點已回問**。正式站現行行為是「客戶端沒帶 top_p → 閘道補 0.001」（§2.3.24 記載、修正未上線），現在加「不指定」選項會讓使用者立即拿到近乎貪婪取樣＝品質劣化。playground 滑桿永遠送明確值（預設 0.8）、從不踩「未帶」路徑，現狀無 bug、不動。**平台複查後撤回「先改不會錯」、確認補值屬實**：origin/main（aaff1acda）`relay/channel/ali/text.go:31` 對 `TopP == nil` 走 `FromPtrOr(request.TopP, 0)` → `topP <= 0` 分支塞 0.001，位於 ali adaptor ConvertOpenAIRequest 的 default 分支——**所有阿里渠道 chat 請求都經過**（qwen3.x/VL/coder、deepseek、glm-5.x 全中），唯一豁免是 `kimi/` 前綴。平台修正（nil 直接 return 不補值＋測試釘住）在功能分支上，**上線後九項（八參數＋top_p「不指定」選項）一起解凍**，平台會回報。

## 2026-08-25

- fix(ui)：**prime 費用顯示改比照 720P 每秒基準**（Levi 指示）。原顯示「$0.068/次」——漏把 prime 加進 `_VIDEO_SEC_PRICE` 每秒單價表，退回了按次顯示且把 480P 基準價當總價。補上官方三檔（480P 0.068／720P 0.14／1080P 0.28），現與其他影片模型一致顯示「約 $0.14/秒（720P）」，實測確認。
- fix(ui)：**依阿里官方文件批次收錄（reference §2.3.3–§2.3.19，Levi 提供截圖＋平台核對）修正三處無效控制項**。①wan2.7-image 四筆條目加 no_negative_prompt/no_prompt_extend——官方明載不支援 prompt_extend（改用 thinking_mode）、文檔無 negative_prompt（wan2.6 無官方頁維持現狀待補）；②happyhorse 影片七筆加 no_prompt_extend——官方四接口皆無此參數（閘道還硬編 true 送出，P1-3）；③animate 任務隱藏 watermark 開關——官方 watermark 在 input 層（全阿里唯一例外），閘道結構無該欄位且硬編 false，客戶開了送不出去（P1-1，平台修復後恢復）。六顆孤兒/歸屬參數結案（§2.3.6）：thinking_mode/enable_sequential/bbox_list/color_palette 是**萬相的**、steps/scale 五份官方文檔皆無（孤兒，解釋了 steps=99999 探測結果——上游不認得就忽略）。**記錄待辦**：qwen-image-2.0 生成的官方 size 是固定枚舉（2688×1536 等）與我方現列尺寸不同——實際輸出尺寸未驗，列驗證待辦；wan2.7-image 官方 size 值域 1K/2K/4K 或寬×高。qwen 系 seed 官方明言「不保證重現」。Playwright 驗證三處隱藏正確，測試 76 條全過。
- infra：**nenai.com.tw 計價「不同步」定性為 CloudFront 舊快取**（文檔站回報 prime 兩網域配置不一）。鑑別（唯讀）：兩網域同一後端實例（/api/status start_time 相同）；nenai.com.tw 前面是 CloudFront（3.169.121.x），GET /api/pricing 命中舊快取（x-cache Hit、age 27462s、早於填價時間），cache-busting 無效（behavior 忽略 query string）。**計費不受影響**：建任務是 POST、CloudFront 不快取、直達源站正確配置——受影響的只有查價顯示。Levi 裁示：invalidation 與快取政策**都不處理**——nenai.com.tw 那側是單向同步、會自行同步平台，查價顯示殘影屬預期行為；playground 與文檔站把自己這側更新好即可。
- feat(models)：**新增 wan3.0-video-prime（萬相 3.0 高速版，僅 t2v）**。Levi 後台填正單價後續行：計價驗證分毫不差（480P 2 秒實測，quota 68000＝$0.136＝0.068×2，other 欄位維度 resolution-480P/seconds:2/model_price:0.068 全正常）、任務 99 秒完成影片正常。**只列實測範圍**：t2v、480P/720P/1080P（閘道 422 白名單實測）、時長 [2,30]（同驗證分支）；i2v/r2v/vedit、smart_duration、adaptive ratio 均未驗——供應商文件到位後擴充，不照家族推。攔截機制順帶實證：計價填正前任務建立會被擋（未觸發 TaskPricePatches 例外）。
- 前情：**wan3.0-video-prime 上架（暫停在計價）**。探測（正式環境，Levi 確認）：計價表有（ali-wan、三群組已開通）、閘道驗證層就位（違規 resolution 回 wan3.0 白名單 422 非 AccessDenied）、不在我方 key 群組的 /v1/models 清單。**擋路問題**：計價配置錯誤——quota_type 0＋model_price 0＋model_ratio 37.5（token 形狀），是 2026-08-20「wan3.0-video $0.05 填錯欄位」同類事故的翻版。平台確認**無漏計費窗口**：relay_task 第 6 步攔截會在預扣費前回 400 model_price_not_set（例外：constant.TaskPricePatches 環境變數白名單可跳過，正式站 env 需 Levi 確認）。**正確單價（平台倍率表，官方定價 2026-08-25 核對）**：model_price=0.068（$/秒、480P 基準）、720P $0.14、1080P $0.28——後台只填 0.068 一個數字，其餘倍率表換算；model_price 是最便宜檔的價、不可單獨對外呈現。後台配置由 Levi 親自處理（各 session 不代改）。上架續行條件：①後台填 0.068 ②供應商文件進 reference/ ③實測生成一次（屆時再請授權）。

- 記錄：**文檔站兩項裁示（其使用者）**：z-image prompt_extend ×2 倍率完全不寫（含「詳見價格頁」折衷）；萬相四參數（thinking_mode/enable_sequential/bbox_list/color_palette）等閘道驗證後再寫。**懸置帳本（等 Levi 解凍，雙方對齊）**：①萬相 edits 9 顆效果層（marshal 已驗、缺 wan2.7 端點行為）②seedream 渠道歸屬 ③veo negative_prompt ④prime 擴充（另缺供應商專頁）⑤qwen-2.0 生成 size 驗證——**範圍擴大，兩站都在雷區**：官方（§2.3.18）生成是固定枚舉（2688×1536/1536×2688/2048×2048/2368×1728/1728×2368），文檔站宣稱連續像素範圍（589,824–16,777,216，疑似把 wan2.7 的值域套到千問）、我方 UI 列的五尺寸（1024*1024 等）同樣不在枚舉——當初「實測」可能只驗成功生成、沒量輸出尺寸（ref_strength 同型混淆）。**解凍協定**：qwen-image-2.0 t2i 送 size=1024*1024（雙方宣稱合法、官方枚舉沒有）→ 400＝值域錯誤；成功則下載量實際像素——若≠1024×1024 即「靜默改尺寸」（最糟型態，客戶不自知）。**編輯路徑的「不在雷區」結論已撤回**（文檔站抓到我方同型混淆）：edit 冒煙的輸入圖是 1024×1024 正方形＋送 size=1024x1024＋輸出 1024×1024——無法分辨「size 生效」還是「輸出隨輸入圖」，證據作廢。文檔站的實測反例：送 1280*720、輸入 900×506 → 輸出 2720×1520（皆不等）＝size 非絕對尺寸；但三者比例都 ≈16:9，「比例來自輸入圖」仍未證。**edit 判別協定（併入第 5 項）**：非 16:9 輸入圖（如 1000×500）＋size=800*800 → 量輸出比例：≈2:1＝比例隨輸入圖（文檔站寫法正確）；≈1:1＝比例來自 size（兩站都要改）；其他＝重寫。一張圖解三問。
- 裁決：**三項驗證全部懸置**（Levi 2026-08-25）——seedream 渠道綁定 DB 查詢、wan2.7-image edits 小額驗證、seedream 判別探測都不做。後果照實記錄：wan2.6/2.7 i2i 的 seed/negative_prompt 控制項維持現狀（開著、送出、轉發到未實測端點），若上游拒收會由客戶端先發現——已知風險、裁示接受；seedream 渠道歸屬懸案維持，文檔站 byteplus/seedream 頁與我方 seedream 旗標都不動。三項日後要解凍，證據路徑都已備好（平台 reference §2.3.1 三渠道對照表＋判別探法）。
- infra/協作：**ali edits 逐模型路由表定案**（平台讀碼，前任「wan 系不在修復範圍」說法已自我更正）。分派是兩層：isOldWanModel（含 wan 且不含 2.6/2.7）→ Wanx 路徑；否則按 Content-Type——**multipart→修復後函式（扁平欄位）、JSON→generations 同函式（巢狀）**，同一模型兩種寫法並存。結果：千問 fusion 四模型 11 顆全通（playground 零改動——本來就送扁平 multipart，先前被丟現在通了）；wan2.6/2.7 吃得到修復但上游端點不同（image-generation/generation），9 顆新參數屬讀碼未實測——**行為變更風險：客戶填 seed 從「靜默丟棄」變「轉發到未驗端點」可能 400**，擬小額驗證；seedream 兩顆出現跨 session 矛盾（新表：volcengine 渠道 edits 註解掉巢狀也死 vs batch-2 marshal 實測：像 ali 兩巢狀都轉發）——已丟回平台裁決（疑渠道 DB 配置與讀碼不一致），前不動。我方 steps=99999 探測異常的方法學更正：input.messages 形狀正是修復函式產物、不構成「另有路徑」證據；「上游沒回錯≠參數生效」已成平台 reference 檔的正式分級規則。另一句同輪沉澱（平台）：batch-2 誤判的線索其實藏在結果裡——**六顆不同來源的模型跑出完全一致的行為，比較可能是同一個 converter 跑了六次，而不是六顆模型碰巧一致；異常的一致性值得懷疑**。平台 B 軌（找官方參數頁，不花錢）不在凍結範圍、持續進行，第一批排 Qwen Image 3.0——官方若明載 steps/scale 型別與值域，可不靠實測把 ❓推論升到 📄官方。**後續（§2.3.2）**：六顆無註解參數（steps/scale/thinking_mode/enable_sequential/bbox_list/color_palette）git blame 全部出自上游貢獻者的「feat: add wan 2.7」commit——**很可能是 wan 的參數、不是千問的**；「欄位存在於共用結構體」既不證明 qwen 支援也不證明 wan 支援。B 軌第一批已改為 Qwen 3.0＋wan2.7-image 官方頁並重，先釐清歸屬。新警語入 reference：**結構體共用 ≠ 模型都支援**。閘道形態（讀碼級）可先用：三顆 *bool 是「顯式 false 照送、不帶則不出現」；steps/scale 以字串轉發（型別疑點維持）；bbox_list/color_palette 以 JSON 字串承載、僅驗合法性。playground 未實作此六顆、零影響。
- infra/協作：**平台 PR #59 部署上線**（db0a43857）——playground 免費探測證實 gpt-image 剝除生效（壞 size 探測法：回深層 size 錯誤而非 unknown_parameter＝三顆已剝）；wan 影片扁平相容生效（適用全部共用轉換的影片模型，seed 上游重現性仍屬效果層未驗）；MAI 未修、旗標保留。**ali edits 修復（41632edbb）狀態存疑待路由表**：對 qwen-image-2.0 edits 送非法 steps=99999 未被拒、直接成功生成（費約 NT$1）——疑 fusion 系 edits 另有轉換路徑；且兩任平台 session 對 wan 系 edits 歸屬說法相反（isOldWanModel 是否含 2.6/2.7）。已要求平台出九個 i2i 模型的逐模型路由表，到手前不動任何控制項。🔴 計費變更隨部署生效：edits 的 n 張數倍率（原漏收只算單張）與 z-image prompt_extend ×2——適用範圍同樣等路由表，屆時通知文檔站列必寫。平台新增 `reference/api-params-official.md` 參數權威來源（含「上游沒回錯≠參數生效」分級規則，源自我方 ref_strength 教訓）；其中 kling/vidu/hailuo/jimeng/carrothub 五家傳參機制未盤點——playground 陣容目前無此五家，無實測資料可貢獻。

- fix(image)：**圖像 negative_prompt 的家族分流修正**（文檔站掃頁撞出、平台 marshal 逐家實測；結論「非全死、各家機制不同」）。千問 3.0 系列：扁平被收進閘道 Extra 但無人讀取＝靜默失效，改送巢狀 `parameters.negative_prompt`（與 wan 影片 metadata 同型的坑，圖像版）；MAI 全系：`convertToMAIImageRequest` 重建請求只留 model/prompt/width/height，negative_prompt 與 **seed** 怎麼寫都到不了上游——t2i 三型加 no_seed/no_negative_prompt 旗標（UI 隱藏＋不送）。新增 `_apply_image_negative_prompt()` 分流 helper＋測試（75 全過）。**批次二結果**（平台 marshal 逐家實測）：①ali 全家族（含 seedream/z-image）平台層行為一致——扁平死、parameters 與 input 兩種巢狀都轉發，**上游吃哪個位置屬效果層**（冻結中），故僅 3.0 改巢狀（有 DTO 代次註解背書）、其餘維持扁平等平台「扁平相容」修案（採納則零改動修復）；②**ali edits（multipart，i2i 實際路徑）是重建請求，negative_prompt 無法用任何寫法救**，需平台修（已入 Levi 修案清單）；③MAI generations 與 edits 同函式重建、只留 model/prompt/size(→width/height)/n——六筆條目四旗標補齊（no_seed/no_negative_prompt/no_watermark/no_prompt_extend）、後端兩端點剝除。**ali edits 重建函式完整清單**（平台）：AliImageParameters 11 欄只組 3（size/n/watermark），negative_prompt/seed/prompt_extend/steps/scale/thinking_mode/enable_sequential/bbox_list/color_palette 全丟、不讀 Extra 無法自救；z-image 的 prompt_extend 在 generations 有 2 倍計費、edits 丟棄所以不多收——日後平台補轉發時計費提示要連動。
- fix(image)：**ref_strength 考古定案——從未生效，我方記錄為證據等級混淆**。平台 repo＋git 全歷史 grep「ref_strength」為 0：這個參數名**從未存在於閘道**。矛盾考古：我方 update.md 當初寫「實測 ref_strength 有效、不會被拒絕」——實際只驗了「帶了不會被拒」（閘道靜默丟棄當然不會被拒），從未驗「調整值改變輸出」，「不報錯」被記成「有效」——正是本專案一直在防的混淆，這次自己中招。處置：UI 滑桿整個收起（從第一天起就沒作用過）、後端停止送出、README 勘誤。懸案後續（平台追查）：ali edits 其實有**兩條**路徑——舊 wan（不含 2.6/2.7）走 `oaiFormEdit2WanxImageEdit`（直接解原始 body，**扁平 negative_prompt 有效**）、其餘走全丟棄那條。playground 陣容無舊 wan 圖像模型故零改動，記錄備用。qwen-2.0 的歷史拒絕仍無法解釋（它不含 wan、走全丟棄路徑理論上不可能拒）——標「存疑歷史記錄」結案，不影響現行處置。ali 圖像至此確認**四條轉換路徑四種行為**（generations 讀 Extra 巢狀／edits 主路徑只組 3 顆不讀 Extra／舊 wan edits 解原始 body／MAI 重建只留 4 顆），平台將以結構性問題報 Levi 裁決（局部修補 vs 統一重構）。
- feat(image)：**GPT Image 開放 moderation 參數**（auto/low，Levi 裁示；僅 t2i——generations 端點專屬，edits 沒有這個參數，I2I 時單獨收起）。UI 下拉＋後端白名單透傳＋MCP 參數（非 GPT 家族帶了明確報錯）。注意證據等級：閘道透傳經平台 marshal 實測，上游實際效果未冒煙（Levi 裁示暫緩付費驗證），首次真實使用即驗證。A/B 組其餘參數盤點結果（均未實作，等冒煙）：`mask`／`input_fidelity` edits 路徑閘道完整支援（mask 有專門檔案處理）；`output_compression`／`partial_images`／`n`（平台不設限，OpenAI 規格 1-10）generations 會透傳；`stream` 平台刻意註解掉不支援（要支援是功能開發不是參數放行）；`user` 會透傳（文檔站表列合理保留）；`user_id`（ModelArk/Seedream 專用欄位）對 gpt-image 也會透傳、疑似 response_format 同型雷，**冒煙解凍後優先驗**（確認 400 平台就補剝除）。**關鍵架構事實**：generations（JSON）是「結構宣告制」——單一規則三種狀態：有宣告→轉發（含 model/prompt/size/quality/n 等核心欄位，size 經 marshal 實測含非方形皆原樣轉發）、沒宣告→靜默丟棄、有宣告但對 gpt-image 剝除；**不是**「主體欄位＋額外參數白名單」兩層，照兩層記日後會誤判。edits（multipart）另一套全轉發制，結論不能互套；平台的 gpt-image 參數剝除只蓋 JSON 路徑，edits 仍會轉發 response_format→400（平台補修中）——**playground 的後端剝除層對 edits 因此必須保留**。
- fix(image)：**GPT Image 家族剝除不存在的參數**（Levi 實測 gpt-image-2 踩到 400 `Unknown parameter: 'response_format'` 引發的全面盤點）。平台端 marshal 實測：seed/watermark/response_format 會被閘道轉發→上游 400；negative_prompt/prompt_extend 被靜默丟棄——playground 的 UI 一直對 GPT 家族開放這四個控制項，屬「參數根本不存在」型缺陷（與 wan 的「路徑寫錯」不同型）。修正三層：MODELS 四筆 GPT 條目加 no_seed/no_watermark/no_negative_prompt/no_prompt_extend 旗標＋圖片分頁 UI 依旗標隱藏並清空；後端 /api/image/generate、/edit 對 GPT 家族剝除四欄（防 Canvas 與直接呼叫方；與平台之後的閘道剝除並存不衝突）；MCP 對 seed/negative_prompt 明確報錯不靜默剝。實測帶四顆假參數呼叫成功且轉發 body 乾淨。400 的**源頭是文檔站** GPT t2i 頁（參數表＋兩個範例都帶 response_format，i2i 頁反而寫對），已通知修頁。守門測試如設計發動（no_watermark 新欄位觸發歸類）。測試 70 條全過。A/B 組參數（output_compression/moderation/stream/mask/input_fidelity/n 上限）平台盤點中。
- feat(video)：**wan 系參數補全**（Levi 指示「把所有能帶的參數都在前端實現」；參數路徑全部依平台端實測盤點，不用猜的）。
  - **negative_prompt 修正**：第二顆「送了沒作用」——ali 系只吃 `metadata.input.negative_prompt`（⚠️ 是 input 層不是 parameters 層），扁平寫法一直靜默失效（追問平台端才實測確認，第一輪盤點漏了它）。三個端點（t2v/i2v/vedit）併入 `_apply_video_extra_params()`，wan/happyhorse 走 input 層、其他家族維持扁平；與 vedit 既有的 `input.media` 正確合併。
  - **ratio 各代分流**（平台實測值域）：wan3.0 下拉 adaptive＋五比例（預設 adaptive）；wan2.7/happyhorse 「自動（預設）」＝不送、交上游預設；wan2.6 及更早不吃 ratio、下拉整個不顯示；veo/seedance 維持原行為。`_default_ratio()` 同步改（wan2.7/hh/2.6 回空字串不送）。t2v/i2v/r2v 送出補 ratio 欄位（原本只有 vedit 有）。
  - **智能時長**（僅 wan3.0，`duration=-1`＝模型依內容自行決定長度）：獨立開關而非讓使用者打 -1；開啟時隱藏秒數滑桿、費用提示改「依實際生成長度計費」（平台走保證金預扣、按實際秒數結算，事前給數字都是假的）；昂貴確認以上限 30 秒當最壞情況估。MODELS 四筆 wan3.0 條目加 `smart_duration` 旗標；MCP 的 duration 驗證對支援模型放行 -1（valid_values 含 -1）、白名單同步。
  - **template 特效模板刻意不做**：wan3.0 官方參數表沒列，未驗證上游支援前做 UI 就是再造一次「送了沒作用」（平台端同建議）。
  - Playwright 實測五模型的顯示矩陣與智能時長開關全部正確；測試 64→69 條全過。
- fix(video)：**wan＊/happyhorse＊ 的 seed／watermark／prompt_extend 改送巢狀 `metadata.parameters.*`**。平台端 session 回報（文檔站發現行號）：閘道 ali 系 adaptor 只讀巢狀，扁平的 `metadata.seed` 被靜默忽略——playground 這三個開關在兩家等於一直沒作用（seed 設了仍隨機、浮水印關不掉、無錯誤訊息）。四個端點（t2v/i2v/vedit/r2v）的重複寫入抽成 `_apply_video_extra_params()`：**僅確認受影響的兩家走巢狀**（平台修扁平相容部署前後皆有效），veo/seedance 等 adaptor 行為未驗證維持扁平不動、不順手統一；三值皆空時不留空 `parameters` 鍵、seed=0 為合法值。補 5 條純函式測試，64 條全過。
- feat(mcp)：**MCP 第二階段——edit_image／tts／asr 三工具**。`nenai_edit_image`（多圖融合編輯，參考圖收 URL/data URI、上限依 MODELS max_ref、伺服器端逐張抓取轉 multipart image_N）、`nenai_tts`（音色驗證附合法 id 清單）、`nenai_asr`（音訊收 URL/data URI，上限 50MB）。`nenai_list_models` 補 voice 分類（巢狀 tts/asr，音色清單 id/name/desc）；`_mcp_fetch_image_input` 泛化為 `_mcp_fetch_media`（可調上限與預設 mime）。冒煙實測（Levi 確認付費，約 NT$2）：TTS「影音生成平台測試成功」→ ASR 回讀「影音生成平台测试成功。」逐字命中（一次驗兩工具＋data URI 路徑）；edit_image 以上輪青瓷杯圖為參考改色成功。已知行為：本機開發時 `_public_base_url` 對 localhost 回 None，audio_url 維持相對路徑（正式環境自動轉絕對，設計如此）。測試 55→59 條全過。
- fix(mcp)：**tools/call 補 key 有效性驗證**。文檔站對版時發現：假 key（任意 Bearer 值）可透過 `nenai_list_models` 取得完整模型目錄＋即時單價——根因是 `get_api_key()` 只驗「有沒有帶」不驗「是不是真的」（REST 既有行為，瀏覽器流程靠 /login 真驗證擋），加上 `/api/pricing` 全域共享快取會把真使用者暖過的價格服務給任何 key。Levi 裁示收緊 MCP：tools/call 統一先驗 key（上游 GET /v1/models，免費，按 key 雜湊快取 10 分鐘；上游不可達時 fail-open 不進快取），假 key 回 `invalid_api_key`。REST 端維持現狀（比照 /api/user/groups 先例）。補測試（快取 prime 不出網路），55 條全過。
- feat(mcp)：**MCP 第一階段實作**（`/mcp`，依 docs/mcp-tool-design.md）。四工具：`nenai_list_models`（能力/約束/即時單價 discovery）、`nenai_generate_image`、`nenai_generate_video`（t2v/i2v 合一，帶 image_url 即圖生）、`nenai_task_status`。實作要點：
  - 手寫極簡 stateless JSON-RPC（initialize/tools/list/tools/call/ping），**零新依賴**；不發 Mcp-Session-Id（Streamable HTTP 規範允許 stateless）、GET /mcp 回 405。接入方式：`claude mcp add --transport http nenai <站台>/mcp --header "Authorization: Bearer sk-..."`。
  - 工具實作用 in-process ASGI（httpx.ASGITransport）呼叫自己的 `/api/*`——與瀏覽器同路徑，轉譯層/統計自動生效且只計一次；/mcp 本身不在統計範圍。
  - 雙層驗證：MCP 層以 MODELS 預檢，錯誤附 `valid_values`（如 Veo 非法時長回 [4,6,8]）；initialize/tools/list 免 key，tools/call 無 key 回設定引導。
  - 冒煙實測（Levi 確認付費，z-image-turbo 一張＋happyhorse-1.1-t2v 3 秒，約 NT$3）：全鏈路通過；**抓到並修掉一個 bug**——結果 URL 原回相對路徑 `/outputs/...`，remote 客戶端抓不到，改為優先回上游絕對網址、本站暫存以 `_public_base_url` 轉絕對當備援（`fallback_url`）。
  - 新增 `tests/test_mcp.py` 16 條（協定面/驗證路徑/無 key 引導/**MODELS 欄位白名單守門**——出現 MCP 映射表不認識的新欄位即 fail），54 條全過。設計文件同步補：計價「文件不印價格、工具回即時價格」分工、Spicy 公開文件完全不提、llms.txt 引用、URL 時效上線前實測待辦。
- docs：**MCP tool 設計稿**（`docs/mcp-tool-design.md`，未實作）。定案方向：remote MCP 掛現有 FastAPI 的 `/mcp`、共用轉譯層與統計；8＋1 個動作型工具（model 當參數，非 158 模型 158 工具）、動態約束三層解法（list_models discovery＋description 引導＋伺服器端驗證附合法值）、影片 submit/status 分離、Spicy 預設隱藏、媒體 URL 附時效提示對齊「客戶內容不上雲」。AI Canvas 整合拆三層（L0 agent 原生鏈式呼叫／L1 workflow DSL 與 canvas_compose/parse 圖互通、圖不落地／L2 伺服器端 runner 等 L1 驗證後再投資）。新模型上線的同步機制：MODELS 單一真實來源＋部署即生效，工作流只多一步 MCP 冒煙；配套「欄位白名單」守門測試構想。文字對話刻意不收（OpenAI 相容 endpoint 已是標準）。

## 2026-08-23

- feat(stats)：**報表顯示使用者名稱、模型顯示名，長表前端分頁**。
  - 新增 `scripts/build_uid_map.py`：唯讀連網關正式 PostgreSQL（Levi 確認後執行）撈 tokens×users，把每把 key **在記憶體裡**算成 uid（sk- 前綴與裸 key 兩形都算），只落地 uid→使用者名稱對照檔 `outputs/uid-map.json`（gitignore、不進 image）——明文 key 全程不落地。⚠️ 對照檔是去匿名化資料，同報表級保管，更新重跑即可。踩坑記錄：共用 .env 該區塊的 ip/user/passwd 是 VM 的 SSH 憑證，Postgres 帳密在同區塊的 `postgresql://` URI 裡（密碼 URL 編碼、URI 內 host 是內網 IP），且 pg_hba 要求 SSL。
  - `usage_stats.py`：使用者欄顯示「名稱＋小字 uid」（無對照檔退回 uid）；模型欄用 `app.MODELS` 的顯示名（單一真實來源，import app 取得）；使用者／來源 IP／請求明細三張表前端分頁（vanilla JS，各 15/15/25 筆一頁），請求明細上限 200→1000。venv 補裝 psycopg2-binary（僅本地腳本用，刻意不進 requirements.txt）。
- feat(stats)：**報表補「來源 IP／模型／請求明細」三個分析維度**（使用者要求：分析哪個用戶、連線 IP、用了什麼模型、做了什麼 API 動作）。
  - IP 的設計原則不變：**統計檔照舊不存 IP**（外流無害的前提不動）。`scripts/usage_stats.py` 在本地產生報表的當下，用本機 gcloud 憑證即時查 Cloud Logging（平台請求日誌，保留 30 天），以「端點＋狀態碼＋最接近時間（容忍 10 分鐘，統計記於請求結束、日誌記於請求開始）」對回每筆統計紀錄——IP 只存在於本機產生的 HTML 裡。新增 `--no-ip` 旗標；gcloud 不可用時自動退化為無 IP 版本。實測 7 天資料：1994 筆日誌對上 151/203 筆統計（沒對上的是本機開發紀錄，本來就無雲端日誌）。
  - 模型維度：`app.py` 統計 middleware 改由 `request.state.model` 取 model（ASGI scope 共享，繞過先前註解記錄的 BaseHTTPMiddleware 讀 body 與 ContextVar 兩條死路），15 個生成端點各補一行 `request.state.model = ...`；muleai 輪詢從路徑直接取。**需部署後新資料才有 model 欄位**，舊紀錄無此維度（報表 footer 已註明）。
  - 報表新增「來源 IP」彙總表（IP／次數／uid／主要端點／User-Agent）與「請求明細」表（最近 200 筆：時間／uid／IP／模型／端點／狀態／ms）；使用者表加「模型」「來源 IP」欄。
  - 時區修正：統計 ts 與 Cloud Logging 都是 UTC，報表顯示一律轉台北時間並標注（先前報表顯示 UTC 造成過誤讀）；`--days` 篩選基準同步改用 UTC。

## 2026-08-22

- feat(ui)：**前端對齊 NEN 官網設計系統**（Levi 驗收通過後正式化）。與官網 session（website repo）跨 session 協作完成，設計出自官網 DESIGN.md 的色料表與鐵律，四輪對版逐版截圖驗證：
  - 新增 `static/css/theme-nen.css`（index.html 於 style.css 之後載入）：亮色＝暖白紙面 #FBFAF8、髮絲線 #E6E2DB 分區、唯一彩度汝窯天青 #7FA39E、明體標題層（Spectral + Noto Serif TC）、單一 10px 圓角、陰影收斂成髮絲線、實心主按鈕改髮絲線＋天青文字階；狀態色收斂到與紙面同溫。含 v2 硬編碼覆寫（登入卡頭漸層、進度條、succeeded 光暈、focus ring、滑桿、rt 狀態點、rtPulse 共 8 處）與 v3 夜間紙面（`:root[data-theme="dark"]` 整塊接手：暖墨 #191715、夜間天青 #9CC0BA、狀態色明度抬高——取官網墨面色料表，非調暗亮色）。
  - 新增 `static/css/theme-nen-canvas.css`（canvas.html 於 canvas.css 之後載入）：依裁決僅動字型（正文系統字、選單標題與模型介紹彈窗名稱進明體）與高飽和收斂（--cv-accent 橘→夜間天青、--cv-accent-2 藍→天青亮階、danger/ok 對齊夜間組；生成鈕漸層改平塗、+ 鈕光暈換天青）；底色、佈局、litegraph 畫布繪製層一律未動。
  - 裁決記錄：登入頁彩色插畫（太陽/月亮/雲/檯燈）**維持原版**——曾整組重畫成暖灰 high-key（v4），Levi 看過否決，已移除並在檔內註記請勿再提案；tooltip 深色底一題尚未裁決，維持現狀。
  - 修掉原檔一個特定度問題：`.login-header-text h1/p` 寫死 white，須同形選擇器覆寫（詳見 theme-nen.css 註解）。
- docs：CLAUDE.md 新增「前端美學風格」一節——Playground 前端對齊 NEN 官網設計系統（暖白紙面、髮絲線分區、唯一彩度汝窯天青、明體標題層、單一 10px 圓角；完整規範以官網 repo 的 DESIGN.md 為準）。主測試台亮色主題覆寫（theme-nen.css）目前為本地實驗未進版控；tooltip 深底／登入插畫／深色切換／Canvas 深色四題待裁決。
- fix(canvas)：縮放列「重置」按鈕跑版——三顆按鈕共用固定 26px 寬，兩個中文字被擠成直排；給 `#zoomReset` 依內容撐開＋`white-space: nowrap`。

## 2026-08-21

- feat(canvas)：**模型介紹小視窗**——AI Canvas 任一節點（9 種）的模型下拉選到模型時，在下拉旁彈出該模型的介紹，點視窗外或按 Esc 關閉，純介紹不觸發生成。內容三部分：①家族特色文案（新檔 `static/js/model_intros.js`，以模型家族為單位手寫、約 25 個家族全涵蓋，遵守客戶文案規範）；②型號說明與規格 chips（尺寸種數／時長範圍／張數與參考圖上限／含配音／支援看圖／推理深度可調…），從 `/api/models` 資料動態產生、不寫死任何模型知識；③同一 id 多型態（t2i/i2i 等）分列說明。依使用者裁示不放官方連結、只留文字。家族歸類以 id 前綴規則比對（特定前綴排前面），已用腳本驗證全部模型 id 歸類正確；Playwright 實測圖片／影片／文字節點彈窗與 Esc／點外關閉行為。
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
