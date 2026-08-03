# update.md

本檔案記錄這個專案的每一次重要更新（新功能、修 bug、重構、設定調整……），依日期分組、由新到舊排列，每筆盡量附上對應的 git commit hash（`git show <hash>` 可查看完整內容）。

依照 `CLAUDE.md` 的規定：之後每完成一次變更、準備 commit 時，都要在最上方新增一筆條目，不要事後補記。

---

## 2026-08-04

- fix：語音合成 (TTS) 改走正確的上游 endpoint `/v1/services/audio/tts/SpeechSynthesizer`（原本用的 OpenAI 相容 `/v1/audio/speech` 一帶 `voice` 就整個失敗，實測發現這個新 endpoint 才是 `qwen-audio-3.0-tts-*` 系列真正支援 `voice` 的地方，回應是 DashScope 風格的 JSON，音檔要從 `output.audio.url` 下載）。新增 `instructions`（語氣/情緒風格描述，CosyVoice v3 專屬）、`sample_rate`、`volume`、`language_hints` 四個選填參數，前端「語音設定」分頁補上對應輸入欄位。已用測試網關（`192.168.0.245`）帶入官方範例的 `longanlingxin`/`loongjohn` 音色與完整參數實測成功，也驗證了不帶任何選填參數時仍可正常運作。

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
