"""
NenAI Testing Platform
FastAPI Backend - NenAI API Key per-user authentication
"""
import os, sys, json, time, uuid, mimetypes, base64, subprocess, re
from io import BytesIO
from PIL import Image as PILImage
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
import websockets
import asyncio
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI

# ─── Cloud Object Storage ─────────────────────────────────────────
# 容器化部署（尤其像 GCP Cloud Run 這種無狀態、多實例的環境）不能依賴本機磁碟保存
# 產出的圖片/影片——同一個檔案下一次請求可能落在別的實例上，本機路徑就讀不到了。
# 這裡支援阿里雲 OSS / AWS S3 / GCP GCS 三選一，設定好對應環境變數即可啟用，產出的
# 檔案會上傳到雲端物件儲存、回傳一個有效期 7 天的簽名網址，取代本機路徑；三個都沒
# 設定時（或上傳失敗時）呼叫端會自動退回寫入本機磁碟（見 _save_image_bytes 等）。
#
# 環境變數：
#   OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET             阿里雲 OSS
#   S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_BUCKET_NAME
#     [S3_REGION，預設 us-east-1] [S3_ENDPOINT，S3 相容服務才需要]   AWS S3
#   GCS_BUCKET_NAME + 下列其中一種身分：                                        GCP GCS
#     - GCS_CREDENTIALS_JSON（服務帳戶金鑰 JSON 內容）或
#       GOOGLE_APPLICATION_CREDENTIALS（金鑰檔路徑）——本地就有私鑰，直接簽章
#     - GCS_USE_ADC=true——改用 Cloud Run/GCE 附加的服務帳戶（Application
#       Default Credentials），不需要金鑰檔，但附加身分沒有私鑰，簽名網址要
#       改呼叫 IAM SignBlob API 遠端簽章，該服務帳戶需要額外兩個 IAM 設定：
#       1) 對「自己」授予 roles/iam.serviceAccountTokenCreator（自我模擬）
#       2) 對目標 bucket 授予 roles/storage.objectAdmin（或至少
#          objectCreator + objectViewer）
#       且專案需啟用 iamcredentials.googleapis.com（IAM Service Account
#       Credentials API）。gcloud 設定範例：
#         gcloud services enable iamcredentials.googleapis.com --project=$PROJECT_ID
#         gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
#           --member="serviceAccount:$SA_EMAIL" --role="roles/iam.serviceAccountTokenCreator"
#         gsutil iam ch serviceAccount:$SA_EMAIL:roles/storage.objectAdmin gs://$BUCKET
#
# 多組都設定時，用 STORAGE_BACKEND=oss|s3|gcs 明確指定要用哪個；沒指定則依 oss → s3
# → gcs 的順序，自動選第一個「憑證齊全」的當作啟用的後端。
_SIGNED_URL_EXPIRE = 7 * 24 * 3600  # 7 天預簽名網址，三個後端共用

_OSS_BUCKET_NAME = "aimodel-oss"
_OSS_ENDPOINT    = "https://oss-ap-southeast-1.aliyuncs.com"

def _oss_put(data: bytes, key: str) -> Optional[str]:
    ak = os.environ.get("OSS_ACCESS_KEY_ID", "")
    sk = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    if not ak or not sk:
        return None
    try:
        import oss2
        bkt = oss2.Bucket(oss2.Auth(ak, sk), _OSS_ENDPOINT, _OSS_BUCKET_NAME)
        bkt.put_object(key, data)
        return bkt.sign_url("GET", key, _SIGNED_URL_EXPIRE)
    except Exception as e:
        print(f"[storage:oss] upload error [{key}]: {e}")
        return None

_s3_client_cache = None

def _s3_put(data: bytes, key: str) -> Optional[str]:
    global _s3_client_cache
    ak     = os.environ.get("S3_ACCESS_KEY_ID", "")
    sk     = os.environ.get("S3_SECRET_ACCESS_KEY", "")
    bucket = os.environ.get("S3_BUCKET_NAME", "")
    if not ak or not sk or not bucket:
        return None
    try:
        if _s3_client_cache is None:
            import boto3
            _s3_client_cache = boto3.client(
                "s3", aws_access_key_id=ak, aws_secret_access_key=sk,
                region_name=os.environ.get("S3_REGION", "us-east-1"),
                endpoint_url=os.environ.get("S3_ENDPOINT") or None,
            )
        _s3_client_cache.put_object(Bucket=bucket, Key=key, Body=data)
        return _s3_client_cache.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=_SIGNED_URL_EXPIRE)
    except Exception as e:
        print(f"[storage:s3] upload error [{key}]: {e}")
        return None

_gcs_client_cache = None

def _gcs_put(data: bytes, key: str) -> Optional[str]:
    global _gcs_client_cache
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    creds_json  = os.environ.get("GCS_CREDENTIALS_JSON", "")
    creds_path  = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    use_adc     = str(os.environ.get("GCS_USE_ADC", "false")).lower() in ("true", "1", "yes")
    if not bucket_name or not (creds_json or creds_path or use_adc):
        return None
    try:
        from datetime import timedelta
        from google.cloud import storage as gcs_storage
        if _gcs_client_cache is None:
            if creds_json:
                from google.oauth2 import service_account
                info = json.loads(creds_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                _gcs_client_cache = gcs_storage.Client(credentials=credentials, project=info.get("project_id"))
            else:
                # creds_path（GOOGLE_APPLICATION_CREDENTIALS 金鑰檔）或 use_adc（純附加
                # 服務帳戶）都讓 google-cloud-storage 自己走 google.auth.default() 解析身分
                _gcs_client_cache = gcs_storage.Client()
        blob = _gcs_client_cache.bucket(bucket_name).blob(key)
        blob.upload_from_string(data)
        if creds_json or creds_path:
            # 本地就有私鑰（服務帳戶金鑰內容／檔案），可以直接簽章
            return blob.generate_signed_url(version="v4", expiration=timedelta(seconds=_SIGNED_URL_EXPIRE))
        # use_adc：Cloud Run/GCE 附加的服務帳戶沒有私鑰，改呼叫 IAM SignBlob API
        # 遠端簽章——該服務帳戶需要對自己有 roles/iam.serviceAccountTokenCreator
        # （見上方環境變數說明的 gcloud 設定範例），否則這裡會拋權限錯誤
        import google.auth
        import google.auth.transport.requests
        adc_credentials, _ = google.auth.default()
        adc_credentials.refresh(google.auth.transport.requests.Request())
        return blob.generate_signed_url(
            version="v4", expiration=timedelta(seconds=_SIGNED_URL_EXPIRE),
            credentials=adc_credentials,
            service_account_email=adc_credentials.service_account_email,
            access_token=adc_credentials.token,
        )
    except Exception as e:
        print(f"[storage:gcs] upload error [{key}]: {e}")
        return None

_STORAGE_BACKENDS = {"oss": _oss_put, "s3": _s3_put, "gcs": _gcs_put}

def _cloud_put(data: bytes, key: str) -> Optional[str]:
    """依 STORAGE_BACKEND 指定，或依序偵測 oss/s3/gcs 憑證是否齊全，上傳到雲端物件
    儲存並回傳簽名網址；沒有任何後端可用（或上傳失敗）時回傳 None，呼叫端會退回
    寫入本機磁碟。"""
    forced = os.environ.get("STORAGE_BACKEND", "").strip().lower()
    if forced:
        put_fn = _STORAGE_BACKENDS.get(forced)
        if put_fn is None:
            print(f"[storage] 未知的 STORAGE_BACKEND={forced!r}（可用：{', '.join(_STORAGE_BACKENDS)}），退回本機磁碟")
            return None
        return put_fn(data, key)
    for put_fn in _STORAGE_BACKENDS.values():
        url = put_fn(data, key)
        if url:
            return url
    return None

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="NenAI Testing Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NENAI_BASE          = os.environ.get("NENAI_BASE", "https://nen.com.tw")
BASE_URL_COMPATIBLE = f"{NENAI_BASE}/v1"
NENAI_V1            = f"{NENAI_BASE}/v1"

UPLOAD_DIR      = Path(__file__).parent / "static" / "uploads"
OUTPUT_IMG_DIR  = Path(__file__).parent / "outputs" / "images"
OUTPUT_VID_DIR  = Path(__file__).parent / "outputs" / "videos"
OUTPUT_AUD_DIR  = Path(__file__).parent / "outputs" / "audio"
for d in (UPLOAD_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR, OUTPUT_AUD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 靜態檔案掛載
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=Path(__file__).parent / "outputs"), name="outputs")

# ─── TTS 音色清單 ───────────────────────────────────────────────
# 來源：qwen-audio-3.0-tts-* 官方音色列表（每個模型支援的音色不同，不可混用）
# https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-voice-list
_QWEN_TTS_PLUS_VOICES = [
    {"id": "longanlingxin", "name": "Longan Lingxin", "desc": "溫暖有同理心，25 歲，中文/英文"},
    {"id": "longanlufeng",  "name": "Longan Lufeng",  "desc": "開朗活潑，25 歲，中文/英文"},
]
_QWEN_TTS_FLASH_VOICES = [
    {"id": "longanfengyue",   "name": "Longan Fengyue",  "desc": "自然親切，30 歲，中文/英文"},
    {"id": "longanyuanfei",   "name": "Longan Yuanfei",  "desc": "高傲典雅，30 歲，中文/英文"},
    {"id": "longanlingxi",    "name": "Longan Lingxi",   "desc": "可愛甜美，25 歲，中文/英文"},
    {"id": "longanxiaoxin",   "name": "Longan Xiaoxin",  "desc": "友善活潑，22 歲，中文/英文"},
    {"id": "longanhuan_v3.6", "name": "Longan Huan",     "desc": "25 歲，中文/英文"},
    {"id": "longjielidou_v3.6", "name": "Longjie Lidou", "desc": "天真男孩，5 歲，中文/英文"},
    {"id": "longpaopao_v3.6",   "name": "Long Paopao",   "desc": "軟萌可愛，5 歲，中文/英文"},
    {"id": "longhuohuo_v3.6",   "name": "Long Huohuo",   "desc": "調皮男孩，8 歲，中文/英文"},
    {"id": "longchuanshu_v3.6", "name": "Long Chuanshu", "desc": "川普大叔，40 歲，中文/英文"},
    {"id": "loongmary",      "name": "loongmary",  "desc": "溫暖英式口音，20 歲，英文"},
    {"id": "loongeva_v3.6",  "name": "loongeva",   "desc": "聰慧優雅，28 歲，英文"},
    {"id": "loongjohn",      "name": "loongJohn",  "desc": "沉穩親切美式口音，28 歲，英文"},
]
# 來源：Google Gemini TTS 官方音色列表（3 個 gemini-*-tts 模型共用同一組 30 個音色）
# https://ai.google.dev/gemini-api/docs/speech-generation
_GEMINI_TTS_VOICES = [
    {"id": "Zephyr", "name": "Zephyr", "desc": "Bright"}, {"id": "Puck", "name": "Puck", "desc": "Upbeat"},
    {"id": "Charon", "name": "Charon", "desc": "Informative"}, {"id": "Kore", "name": "Kore", "desc": "Firm"},
    {"id": "Fenrir", "name": "Fenrir", "desc": "Excitable"}, {"id": "Leda", "name": "Leda", "desc": "Youthful"},
    {"id": "Orus", "name": "Orus", "desc": "Firm"}, {"id": "Aoede", "name": "Aoede", "desc": "Breezy"},
    {"id": "Callirrhoe", "name": "Callirrhoe", "desc": "Easy-going"}, {"id": "Autonoe", "name": "Autonoe", "desc": "Bright"},
    {"id": "Enceladus", "name": "Enceladus", "desc": "Breathy"}, {"id": "Iapetus", "name": "Iapetus", "desc": "Clear"},
    {"id": "Umbriel", "name": "Umbriel", "desc": "Easy-going"}, {"id": "Algieba", "name": "Algieba", "desc": "Smooth"},
    {"id": "Despina", "name": "Despina", "desc": "Smooth"}, {"id": "Erinome", "name": "Erinome", "desc": "Clear"},
    {"id": "Algenib", "name": "Algenib", "desc": "Gravelly"}, {"id": "Rasalgethi", "name": "Rasalgethi", "desc": "Informative"},
    {"id": "Laomedeia", "name": "Laomedeia", "desc": "Upbeat"}, {"id": "Achernar", "name": "Achernar", "desc": "Soft"},
    {"id": "Alnilam", "name": "Alnilam", "desc": "Firm"}, {"id": "Schedar", "name": "Schedar", "desc": "Even"},
    {"id": "Gacrux", "name": "Gacrux", "desc": "Mature"}, {"id": "Pulcherrima", "name": "Pulcherrima", "desc": "Forward"},
    {"id": "Achird", "name": "Achird", "desc": "Friendly"}, {"id": "Zubenelgenubi", "name": "Zubenelgenubi", "desc": "Casual"},
    {"id": "Vindemiatrix", "name": "Vindemiatrix", "desc": "Gentle"}, {"id": "Sadachbia", "name": "Sadachbia", "desc": "Lively"},
    {"id": "Sadaltager", "name": "Sadaltager", "desc": "Knowledgeable"}, {"id": "Sulafat", "name": "Sulafat", "desc": "Warm"},
]

# MAI Image 家族（2.5 / 2.5-Flash / 2.5-Pro）共用的尺寸清單。約束見 MODELS 裡的註解：
# 每邊 ≥ 768 像素、總像素 ≤ 1,056,768。這五個都逐一對正式網關實測確認可用。
# MAI 的尺寸有兩條**互相獨立**的限制，兩條都要滿足：每邊至少 768 px、總像素 ≤ 1,056,768。
# 只看總像素會誤判——767x1024 的總像素只有 785,408、遠低於上限，照樣回
# 「'width' must be at least 768 pixels」。所以 1536x1024 / 1024x1536 這兩個在別家
# 很常見的尺寸在 MAI 上都不可用（曾經列進選單，變成兩個永遠會被拒的選項）。
#
# 另外上游會把尺寸**往下對齊到 16 的倍數**：實測請求 1366x768（官方文件自己舉的例子）
# 拿回來的是 1360x768（讀 PNG header 確認，不是看回應欄位）。所以這裡直接登記對齊後的
# 1360x768 / 768x1360——實測這兩個值上游照收、且輸出與請求完全相符，使用者選什麼就拿到
# 什麼。其餘三個尺寸本來就是 16 的倍數，不受影響。
_MAI_IMAGE_SIZES = ["1024x1024", "1360x768", "768x1360", "1152x896", "896x1152"]

# MAI 的尺寸不是固定枚舉，而是「滿足約束的任意值」——實測 size="1200x800"（不在上面的
# 清單裡）輸出就是 1200x800，完全相符。所以 UI 除了預設選項，另外開放自訂寬高。
#
# 自訂尺寸有**兩條路**，都實測可用（2026-08-12 對正式環境）：
#   size 字串        送 size="1088x960"                → 輸出 1088x960
#   width/height     送 {"width":1088,"height":960}    → 輸出 1088x960
# 兩者都給時**上游以 width/height 為準**：送 size=1024x1024（合法）配
# width/height=2000x2000（超限）會回 400，而不是照 size 產圖。
#
# 這兩個欄位一度是無效的：更早實測送 {"width":2000,"height":2000} 竟正常產出
# 1024x1024，因為閘道的 dto.ImageRequest 沒有宣告 width/height，未宣告的欄位會落進
# Extra map 而 MarshalJSON 刻意不把 Extra 合併回去，於是靜默消失。閘道端已改成從
# Extra 取值（且刻意仍不宣告在 DTO——那個結構是所有 image 渠道共用的，一宣告就會把
# width/height 透傳給 dall-e 這類只認 size 的上游，讓原本能跑的請求變 400），現已上線。
#
# 兩條路的對齊行為一致：width/height 同樣會往下對齊到 16 的倍數（送 1366 得 1360）。
_MAI_CUSTOM_SIZE = {
    "min_side": 768,        # 每邊至少 768（獨立於總像素限制：767x1024 只有 78 萬像素照樣被拒）
    "max_pixels": 1056768,  # 總像素上限
    "align": 16,            # 上游會往下對齊到 16 的倍數，前端先對齊好，免得使用者拿到非預期尺寸
    "modes": ["size", "wh"],  # 兩條路都通，UI 讓使用者選要用哪一種送出
}

# ─── Model Registry ───────────────────────────────────────────
# sizes: 支援的尺寸清單；max_n: 最大生成張數；audio: 支援配音；min/max_dur: 影片時長範圍
MODELS = {
    "text": [
        # ── 旗艦 ──────────────────────────────────────────────────
        {"id": "qwen3.8-max",        "name": "Qwen3.8 Max",      "group": "旗艦",   "desc": "最新旗艦，最強推理",     "thinking": True},
        {"id": "qwen3.7-max",        "name": "Qwen3.7 Max",      "group": "旗艦",   "desc": "前代旗艦，強推理",       "thinking": True},
        {"id": "qwen3.6-max-preview","name": "Qwen3.6 Max",     "group": "旗艦",   "desc": "新一代旗艦，強推理",     "thinking": True},
        # ── 均衡 ──────────────────────────────────────────────────
        {"id": "qwen3.6-plus",       "name": "Qwen3.6 Plus",     "group": "均衡",   "desc": "1M context，性價比最佳", "thinking": True},
        {"id": "qwen3.7-plus",       "name": "Qwen3.7 Plus",     "group": "均衡",   "desc": "最新均衡模型",           "thinking": True},
        {"id": "qwen3.5-plus",       "name": "Qwen3.5 Plus",     "group": "均衡",   "desc": "前代均衡模型",           "thinking": True},
        # ── 極速 ──────────────────────────────────────────────────
        {"id": "qwen3.6-flash",      "name": "Qwen3.6 Flash",    "group": "極速",   "desc": "新一代極速模型",         "thinking": True},
        {"id": "qwen3.5-flash",      "name": "Qwen3.5 Flash",    "group": "極速",   "desc": "速度快、成本低",         "thinking": True},
        # ── 代碼（實測 enable_thinking 對這兩個 coder 模型完全沒有效果——true/false
        #    都不會有 reasoning_content，代碼生成場景本來就不需要思考過程，
        #    thinking 維持 False 避免顯示一個沒作用的開關）──────────────
        {"id": "qwen3-coder-plus",   "name": "Qwen3 Coder Plus", "group": "代碼",   "desc": "代碼生成旗艦",           "thinking": False},
        {"id": "qwen3-coder-flash",  "name": "Qwen3 Coder Flash","group": "代碼",   "desc": "代碼生成極速",           "thinking": False},
        # ── 角色 ──────────────────────────────────────────────────
        {"id": "qwen-plus-character", "name": "Qwen Plus Character","group": "角色", "desc": "角色扮演，Plus 品質",   "thinking": False},
        # ── 第三方 ────────────────────────────────────────────────
        # DeepSeek/GLM 實測都支援 enable_thinking 開關（會回傳獨立的 reasoning_content
        # 思考過程），DeepSeek V4 系列預設就是思考模式開啟，enable_thinking:false 可關閉
        {"id": "deepseek-v4-pro",    "name": "DeepSeek V4 Pro",  "group": "第三方", "desc": "最新旗艦推理",           "thinking": True},
        {"id": "deepseek-v4-flash",  "name": "DeepSeek V4 Flash","group": "第三方", "desc": "最新極速推理",           "thinking": True},
        {"id": "deepseek-v3.2",      "name": "DeepSeek V3.2",    "group": "第三方", "desc": "前代深度推理",           "thinking": True},
        # GLM 5.x 除了布林的 enable_thinking，另外支援字串的 reasoning_effort 分段推理
        # 強度（實測 2026-08-10，各段的 reasoning_tokens：none/minimal → 0、low 182、
        # medium 198、high 202、xhigh 239、max 208）。兩者可同時送，enable_thinking:false
        # 的優先權高於 reasoning_effort；但反向不成立——實測 enable_thinking:true 配
        # reasoning_effort:none 仍然不會思考，也就是「關」的那一方永遠贏。
        # 支援的枚舉各型號不同，送錯值會回 400 並列出正確清單，故以 reasoning_efforts 標明。
        {"id": "glm-5.1",            "name": "GLM 5.1",          "group": "第三方", "desc": "智譜 GLM 前一版",        "thinking": True,
         "reasoning_effort": True, "reasoning_efforts": ["none", "minimal", "low", "medium", "high", "xhigh"]},
        {"id": "glm-5.2",            "name": "GLM 5.2",          "group": "第三方", "desc": "智譜 GLM 最新版（1M context）", "thinking": True,
         "reasoning_effort": True, "reasoning_efforts": ["none", "minimal", "low", "medium", "high", "xhigh", "max"]},
        # ── ByteDance Seed（字節跳動豆包大模型；seed-2.0 系列無條件會回思考過程
        #    reasoning_content，實測過 enable_thinking:false 對它們沒有效果
        #    （跟 Gemini 3.x 系列同樣「關不掉」），thinking 維持 False 不顯示
        #    會誤導使用者以為能控制的開關；seed-sc 則完全沒有思考過程）──
        {"id": "dola-seed-sc",       "name": "Seed SC",          "group": "ByteDance", "desc": "字節跳動豆包，一般對話", "thinking": False},
        {"id": "dola-seed-2.0-lite", "name": "Seed 2.0 Lite",    "group": "ByteDance", "desc": "字節跳動豆包，輕量推理", "thinking": False},
        {"id": "dola-seed-2.0-pro",  "name": "Seed 2.0 Pro",     "group": "ByteDance", "desc": "字節跳動豆包，旗艦推理", "thinking": False},
        # ── Claude（實測過 enable_thinking 與 Anthropic 原生 thinking 參數在這個
        #    網關上都不會回傳任何思考過程，thinking 一律維持 False；temperature/
        #    top_p 也不能送，Bedrock 後端會直接回 400 "temperature is deprecated"）──
        {"id": "claude-opus-5",               "name": "Claude Opus 5",     "group": "Claude", "desc": "最新旗艦",         "thinking": False},
        {"id": "claude-opus-4-8",             "name": "Claude Opus 4.8",   "group": "Claude", "desc": "前代旗艦",         "thinking": False},
        {"id": "claude-opus-4-7",             "name": "Claude Opus 4.7",   "group": "Claude", "desc": "前代旗艦",         "thinking": False},
        {"id": "claude-opus-4-6",             "name": "Claude Opus 4.6",   "group": "Claude", "desc": "前代旗艦",         "thinking": False},
        {"id": "claude-opus-4-5-20251101",    "name": "Claude Opus 4.5",   "group": "Claude", "desc": "前代旗艦",         "thinking": False},
        {"id": "claude-opus-4-1-20250805",    "name": "Claude Opus 4.1",   "group": "Claude", "desc": "前代旗艦",         "thinking": False},
        {"id": "claude-sonnet-5",             "name": "Claude Sonnet 5",   "group": "Claude", "desc": "最新均衡，推薦使用", "thinking": False},
        {"id": "claude-sonnet-4-6",           "name": "Claude Sonnet 4.6", "group": "Claude", "desc": "前代均衡模型",     "thinking": False},
        {"id": "claude-sonnet-4-5-20250929",  "name": "Claude Sonnet 4.5", "group": "Claude", "desc": "前代均衡模型",     "thinking": False},
        {"id": "claude-haiku-4-5-20251001",   "name": "Claude Haiku 4.5",  "group": "Claude", "desc": "極速模型",         "thinking": False},
        {"id": "claude-fable-5",              "name": "Claude Fable 5",    "group": "Claude", "desc": "創意寫作模型",     "thinking": False},
        # ── GPT（推理強度用 reasoning_effort 字串控制，不是 enable_thinking 布林值——
        #    實測過對 GPT 模型送 enable_thinking 會直接 400 "Unknown parameter"；
        #    reasoning_effort 這個網關接受的枚舉是 none/low/medium/high/xhigh（不是
        #    OpenAI 官方文件常見的 minimal/low/medium/high），reasoning_effort=none
        #    會讓 reasoning_tokens 掉到 0，證實真的有效，但這個閘道沒有回傳可讀的
        #    思考過程文字，只影響耗費的 token 數/延遲）──
        {"id": "gpt-5.6-terra", "name": "GPT 5.6 Terra", "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.6-sol",   "name": "GPT 5.6 Sol",   "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.6-luna",  "name": "GPT 5.6 Luna",  "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.5",       "name": "GPT 5.5",       "group": "GPT", "desc": "均衡模型",     "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.4",       "name": "GPT 5.4",       "group": "GPT", "desc": "均衡模型",     "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.4-mini",  "name": "GPT 5.4 Mini",  "group": "GPT", "desc": "輕量極速",     "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.4-nano",  "name": "GPT 5.4 Nano",  "group": "GPT", "desc": "超輕量極速",   "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5.2",       "name": "GPT 5.2",       "group": "GPT", "desc": "前代均衡模型", "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        {"id": "gpt-5-mini",    "name": "GPT 5 Mini",    "group": "GPT", "desc": "前代輕量模型", "thinking": False, "reasoning_effort": True, "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]},
        # ── Gemini（改走 Gemini 原生 API，見 _GEMINI_NATIVE_TEXT_MODELS）──────────
        # 先前走 OpenAI 相容端點時，Gemini 的思考既看不到也關不掉，這裡一度全部標成
        # thinking: False。改走原生端點後兩件事都做得到了（實測 2026-08-10）：
        # thinkingConfig.includeThoughts 拿得到思考過程全文、thinkingBudget=0 能真的
        # 關掉思考。但支援度各型號不同，兩個例外見 _GEMINI_NO_THINKING_OFF /
        # _GEMINI_NO_INCLUDE_THOUGHTS：
        #   gemini-2.5-pro        思考關不掉（送 budget=0 直接 400），但過程看得到，
        #                         所以不給開關、一律顯示思考過程
        #   gemini-2.5-flash-lite 思考關得掉，但過程拿不到（送 includeThoughts 直接
        #                         400），所以有開關、但不會顯示思考區塊
        {"id": "gemini-3.1-pro-preview",      "name": "Gemini 3.1 Pro Preview",      "group": "Gemini", "desc": "旗艦，最強推理",   "thinking": True},
        {"id": "gemini-3.6-flash",            "name": "Gemini 3.6 Flash",            "group": "Gemini", "desc": "新一代均衡模型",   "thinking": True},
        {"id": "gemini-3.5-flash",            "name": "Gemini 3.5 Flash",            "group": "Gemini", "desc": "前代均衡模型",     "thinking": True},
        {"id": "gemini-3-flash-preview",      "name": "Gemini 3 Flash Preview",      "group": "Gemini", "desc": "前代均衡模型",     "thinking": True},
        {"id": "gemini-2.5-pro",              "name": "Gemini 2.5 Pro",              "group": "Gemini", "desc": "前代旗艦，深度推理", "thinking": False},
        {"id": "gemini-2.5-flash",            "name": "Gemini 2.5 Flash",            "group": "Gemini", "desc": "前代均衡模型",     "thinking": True},
        {"id": "gemini-2.5-flash-lite",       "name": "Gemini 2.5 Flash Lite",       "group": "Gemini", "desc": "前代輕量極速", "thinking": True},
        # gemini-3.5-flash-lite 跟 2.5-flash-lite 一樣「思考預設是關的、要送 budget=-1
        # 才會啟動」，但它**接受** includeThoughts（2.5 版送了會 400），所以思考過程
        # 看得到。實測：無 thinkingConfig → thoughts=None；budget=-1 → 186；
        # budget=-1 + includeThoughts → 180 且有 thought 區塊
        {"id": "gemini-3.5-flash-lite",       "name": "Gemini 3.5 Flash Lite",       "group": "Gemini", "desc": "新一代輕量極速", "thinking": True},
        # ── xAI Grok（reasoning / non-reasoning 是兩個獨立型號，不是同一模型的參數）──
        # 實測四個型號的行為（2026-08-11，正式環境）：
        #   -reasoning 版預設就思考，-non-reasoning 版完全不思考（reasoning_tokens 恆為 0）
        #   四個都**不回傳 reasoning_content**，所以思考過程一律看不到
        #   enable_thinking 對全部四個無效（grok-4-20-reasoning 送 false 反而更多）
        #   reasoning_effort：只有 grok-4-20-reasoning 有效（none → reasoning 0）；
        #     grok-4-1-fast-reasoning 送了無效（各 3 次中位數 176 vs 245，沒有下降）；
        #     兩個 non-reasoning 版直接回 400
        #   非法值只回通用的 "openai_error"，問不出合法枚舉，故 reasoning_efforts 只列
        #   實測有效的 none 與不帶值的預設
        # grok-4.3 是唯一有完整強度分段、而且支援看圖的 Grok（實測 2026-08-11）：
        #   reasoning_effort 枚舉 none/minimal/low/medium/high（xhigh 與 max 回 422）
        #   none 三次都得到 reasoning_tokens=0，是穩定有效的
        #   一樣不回 reasoning_content，所以思考過程看不到、thinking 維持 False
        {"id": "grok-4.3",                    "name": "Grok 4.3",                    "group": "xAI Grok", "desc": "最新旗艦，可調推理強度、支援看圖", "thinking": False,
         "reasoning_effort": True, "reasoning_efforts": ["none", "minimal", "low", "medium", "high"],
         "vision": True},
        {"id": "grok-4-20-reasoning",         "name": "Grok 4.20 Reasoning",         "group": "xAI Grok", "desc": "旗艦推理（可關閉推理）", "thinking": False,
         "reasoning_effort": True, "reasoning_efforts": ["none"]},
        {"id": "grok-4-20-non-reasoning",     "name": "Grok 4.20",                   "group": "xAI Grok", "desc": "旗艦，不推理", "thinking": False},
        {"id": "grok-4-1-fast-reasoning",     "name": "Grok 4.1 Fast Reasoning",     "group": "xAI Grok", "desc": "極速推理", "thinking": False},
        {"id": "grok-4-1-fast-non-reasoning", "name": "Grok 4.1 Fast",               "group": "xAI Grok", "desc": "極速，不推理", "thinking": False},
        # ── 千問 VL（視覺語言，可在對話中帶入圖片；用標準 OpenAI image_url 格式，
        #    實測 data URI 可用）。vision: True 讓前端顯示圖片上傳欄位 ──
        {"id": "qwen3-vl-plus",               "name": "Qwen3 VL Plus",               "group": "視覺語言", "desc": "看圖對話，Plus 品質", "thinking": False, "vision": True},
        {"id": "qwen3-vl-flash",              "name": "Qwen3 VL Flash",              "group": "視覺語言", "desc": "看圖對話，極速", "thinking": False, "vision": True},
    ],
    "image": [
        # ── 千問文生圖 ────────────────────────────────────────────
        # 千問 3.0 系列的尺寸不是固定枚舉，而是一條面積約束（2026-08-11 對正式網關
        # 實測，兩個型號一致、上下界都驗過）：**總像素 262,144（512×512）～
        # 6,553,600（2560×2560）**，格式必須是 `寬*高`。違反時分別回
        #   Image area must be between 262144 (512x512) and 6553600 (2560x2560) pixels
        #   Expected format: '<width>*<height>'   ← 送 1K/2K/4K 這種規格值會踩到
        # 注意錯誤訊息寫的是「for t2i requests」——i2i 不套用這條，實測 i2i 送
        # size=10*10 照樣會成功產圖。下面列的尺寸都落在上述範圍內（1024*1024、
        # 1280*720、2048*2048 三個另外實際產圖驗過）。
        # 計費：3.0 每次 $0.03、3.0-pro $0.04，但那是**1K 輸出價**；輸出 2K 時網關會
        # 依上游回傳的 usage.output_image_type 自動補倍率（pro 的 2K 是 $0.075，近一倍），
        # 而 /api/pricing 只讀得到 1K 價，所以 UI 顯示的參考單價在 2K 情境會低估。
        # 另外輸入圖是加法附加費 $0.003/張，不受輸出張數或解析度倍率影響。
        {
            "id": "qwen-image-3.0-pro", "name": "千問圖像 3.0 Pro", "group": "千問文生圖",
            "desc": "最新旗艦文生圖，細節更佳", "type": "t2i", "max_n": 6,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024","1664*928","928*1664","2048*2048"],
        },
        {
            "id": "qwen-image-3.0", "name": "千問圖像 3.0", "group": "千問文生圖",
            "desc": "最新文生圖", "type": "t2i", "max_n": 6,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024","1664*928","928*1664","2048*2048"],
        },
        {
            "id": "qwen-image-2.0-pro", "name": "千問圖像 2.0 Pro", "group": "千問文生圖",
            "desc": "文字渲染突出", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        {
            "id": "qwen-image-2.0", "name": "千問圖像 2.0", "group": "千問文生圖",
            "desc": "標準文生圖", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        # ── 萬相文生圖 ────────────────────────────────────────────
        # enable_sequential：組圖模式（連貫故事圖組），開啟時 n 上限由 4 變 12，
        # 實際張數由模型決定、不保證等於 n。
        #
        # size 有兩種寫法、不可混用（依官方文件；2026-08-10 對正式網關實測過所有
        # 清單內的值都能通過驗證）：
        #   方式一（官方推薦）規格值 1K/2K/4K——1K=1024*1024、2K=2048*2048、
        #     4K=4096*4096 總像素。有圖片輸入時輸出寬高比跟隨輸入（多圖時取最後
        #     一張）並縮放到該規格；沒有圖片輸入時輸出正方形。
        #     `4K` 只有 wan2.7-image-pro 的「純文生圖且非組圖」情境支援；
        #     wan2.7-image 完全不支援 4K，組圖模式也只到 2K（sequential_max_size）。
        #   方式二 明確的寬高像素值——總像素範圍 pro 文生圖 [768*768, 4096*4096]、
        #     其餘情境 [768*768, 2048*2048]，寬高比 [1:8, 8:1]。
        #
        # ⚠️ **網關的驗證是寬鬆的，擋不住上面這些個別限制**：實測 wan2.7-image 送
        #    4096*4096、pro 開組圖送 4K 都能通過網關驗證（它只檢查總像素落在
        #    589824~16777216，也就是 pro 文生圖那組最寬鬆的範圍），真正的限制要靠
        #    官方文件。所以能不能用不能靠打網關試出來——這裡的清單以文件為準。
        {
            "id": "wan2.7-image-pro", "name": "萬相 2.7 Image Pro", "group": "萬相文生圖",
            "desc": "旗艦文生圖，細節與畫質更佳", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1K","2K","4K","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960","2048*2048","4096*4096"],
            "supports_sequential": True, "sequential_max_size": "2K",
        },
        {
            "id": "wan2.7-image", "name": "萬相 2.7 Image", "group": "萬相文生圖",
            "desc": "標準文生圖", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1K","2K","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960","2048*2048"],
            "supports_sequential": True, "sequential_max_size": "2K",
        },
        {
            "id": "wan2.6-t2i", "name": "萬相 2.6 T2I", "group": "萬相文生圖",
            "desc": "自由選尺寸", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960"],
        },
        # ── Z-Image ───────────────────────────────────────────────
        {
            "id": "z-image-turbo", "name": "Z-Image Turbo", "group": "Z-Image",
            "desc": "輕量級快速生成", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        # ── MAI Image（Azure OpenAI 管道，尺寸格式與 GPT Image 相同為 WIDTHxHEIGHT）──
        # 尺寸不是固定枚舉，而是兩條同時成立的約束（2026-08-10 對正式網關實測，三個
        # 型號的錯誤訊息完全一致）：**每邊至少 768 像素**，且**總像素不得超過
        # 1,056,768**。違反時分別回
        #   'width'/'height' must be at least 768 pixels
        #   Invalid dimensions WxH: total pixel count (N) exceeds the maximum of 1056768
        # 先前這裡列的 1536x1024 / 1024x1536 都是 1,572,864 像素，超過上限、**一定會被
        # 拒**——三個尺寸裡有兩個從來就不能用。_MAI_IMAGE_SIZES 那組是逐一實測確認可用的。
        {
            "id": "MAI-Image-2.5-Pro", "name": "MAI-Image-2.5-Pro", "group": "MAI Image",
            "desc": "旗艦圖像生成 Pro", "type": "t2i", "max_n": 4,
            "sizes": _MAI_IMAGE_SIZES, "custom_size": _MAI_CUSTOM_SIZE,
        },
        {
            "id": "MAI-Image-2.5", "name": "MAI-Image-2.5", "group": "MAI Image",
            "desc": "旗艦圖像生成", "type": "t2i", "max_n": 4,
            "sizes": _MAI_IMAGE_SIZES, "custom_size": _MAI_CUSTOM_SIZE,
        },
        {
            "id": "MAI-Image-2.5-Flash", "name": "MAI-Image-2.5-Flash", "group": "MAI Image",
            "desc": "極速圖像生成", "type": "t2i", "max_n": 4,
            "sizes": _MAI_IMAGE_SIZES, "custom_size": _MAI_CUSTOM_SIZE,
        },
        # ── 萬相圖像編輯 ──────────────────────────────────────────
        # 參考圖張數上限逐一實測（2026-08-10，正式網關）：萬相 2.7 兩個型號 9 張都生效；
        # 萬相 2.6 上限是 4（送 5 張回 "the last message must contain 1 to 4 images"）。
        # 先前這裡標成 2 是依閘道端 Go struct 的 WanImageInput.images(<=2) 推斷的，
        # 沒有實測——實際上那條約束不適用於 /v1/images/edits 這條路徑。
        {
            "id": "wan2.7-image-pro", "name": "萬相 2.7 Image Pro（編輯）", "group": "萬相圖像編輯",
            "desc": "多圖融合、風格遷移（最多 9 張）", "type": "i2i", "max_n": 1, "max_ref": 9,
            "sizes": ["1024*1024","1K","2K","1280*720","720*1280","960*1280","1280*960","2048*2048"],
        },
        {
            "id": "wan2.7-image", "name": "萬相 2.7 Image（編輯）", "group": "萬相圖像編輯",
            "desc": "標準圖像編輯（最多 9 張）", "type": "i2i", "max_n": 1, "max_ref": 9,
            "sizes": ["1024*1024","1K","2K","1280*720","720*1280","960*1280","1280*960","2048*2048"],
        },
        {
            "id": "wan2.6-image", "name": "萬相 2.6 Image", "group": "萬相圖像編輯",
            "desc": "前代編輯模型（最多 4 張）", "type": "i2i", "max_n": 1, "max_ref": 4,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        # ── 千問圖像 3.0（同一模型 ID 兼具 T2I 與 I2I；上游規定 input.messages 只能一則，
        #    I2I 的 content 是 1～3 個 image ＋ **恰好一個** text，所以參考圖上限 3 張）──
        # no_size：**編輯情境下 size 完全不生效**，輸出尺寸由輸入圖／模型自己決定。實測
        # 送 1280*720（橫向）與 512*512 都一樣得到 2048x2048，連 t2i 的面積約束都不套用
        # （送 10*10 也不會被拒），所以那個尺寸選單放著只會誤導。這不是 3.0 專屬——
        # qwen-image-2.0 與 wan2.7-image 的編輯情境實測也一樣忽略 size（見 README）。
        {
            "id": "qwen-image-3.0-pro", "name": "千問圖像 3.0 Pro（編輯）", "group": "千問圖像編輯",
            "desc": "最新旗艦圖像編輯（最多 3 張，輸出尺寸跟隨輸入圖）", "type": "i2i",
            "max_n": 6, "max_ref": 3, "no_ref_strength": True, "fusion_edit": True, "no_size": True,
        },
        {
            "id": "qwen-image-3.0", "name": "千問圖像 3.0（編輯）", "group": "千問圖像編輯",
            "desc": "最新圖像編輯（最多 3 張，輸出尺寸跟隨輸入圖）", "type": "i2i",
            "max_n": 6, "max_ref": 3, "no_ref_strength": True, "fusion_edit": True, "no_size": True,
        },
        # ── 千問圖像 2.0（生成與編輯融合模型，同一模型 ID 兼具 T2I 與 I2I）──
        {
            "id": "qwen-image-2.0-pro", "name": "千問圖像 2.0 Pro（編輯）", "group": "千問圖像編輯",
            "desc": "生成與編輯融合模型 Pro 系列", "type": "i2i", "max_n": 6, "max_ref": 3,
            "no_ref_strength": True, "fusion_edit": True,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        {
            "id": "qwen-image-2.0", "name": "千問圖像 2.0（編輯）", "group": "千問圖像編輯",
            "desc": "生成與編輯融合模型，加速版", "type": "i2i", "max_n": 6, "max_ref": 3,
            "no_ref_strength": True, "fusion_edit": True,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        # ── GPT Image（尺寸格式為 WIDTHxHEIGHT，與其他模型的 WIDTH*HEIGHT 不同；
        #    supports_gpt_params 標記這個家族額外支援 OpenAI 標準的 quality/
        #    background/output_format 三個參數，已實測確認皆有效）──
        {
            "id": "gpt-image-2", "name": "GPT Image 2", "group": "GPT Image",
            "desc": "OpenAI 旗艦圖像模型", "type": "t2i", "max_n": 4,
            "sizes": ["1024x1024","1536x1024","1024x1536"], "supports_gpt_params": True,
        },
        {
            "id": "gpt-image-1.5", "name": "GPT Image 1.5", "group": "GPT Image",
            "desc": "OpenAI 前代圖像模型", "type": "t2i", "max_n": 4,
            "sizes": ["1024x1024","1536x1024","1024x1536"], "supports_gpt_params": True,
        },
        # ── ByteDance Seedream（尺寸格式同 GPT Image 為 WIDTHxHEIGHT；lite 版實測
        #    畫面至少要 ~369 萬像素，1024x1024 這種常見小尺寸會被拒絕，故只列
        #    2K 起跳的尺寸，pro 版則沒有這個限制）──
        {
            "id": "dola-seedream-5.0-pro", "name": "Seedream 5.0 Pro", "group": "ByteDance Seedream",
            "desc": "字節跳動 Seedream，旗艦文生圖", "type": "t2i", "max_n": 4,
            "sizes": ["1024x1024","1536x1024","1024x1536","2048x2048"],
        },
        {
            "id": "dola-seedream-5.0-lite", "name": "Seedream 5.0 Lite", "group": "ByteDance Seedream",
            "desc": "字節跳動 Seedream，輕量文生圖（畫面較大，最小約 2K）", "type": "t2i", "max_n": 4,
            "sizes": ["2048x2048","2k","3k","4k"],
        },
        # ── Gemini Image（走 Gemini 原生的 /v1beta/models/{model}:generateContent，
        #    以結構化的 imageConfig.aspectRatio 與 imageConfig.imageSize 控制輸出。
        #    先前走 /v1/chat/completions + modalities，那條路徑上 imageConfig 會被
        #    靜默忽略、只能用自然語言注入 prompt 要比例，而且**完全無法控制解析度**
        #    （使用者反映「選不了生成結果大小」）。sizes 為各型號實測支援的
        #    imageSize，見 _GEMINI_IMAGE_SIZES 的註解）──
        {
            "id": "gemini-3-pro-image", "name": "Gemini 3 Pro Image", "group": "Gemini Image",
            "desc": "Google 旗艦圖像生成，畫質最佳", "type": "t2i", "max_n": 4, "sizes": ["1K", "2K", "4K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-3.1-flash-image", "name": "Gemini 3.1 Flash Image", "group": "Gemini Image",
            "desc": "速度與品質平衡，建議日常使用", "type": "t2i", "max_n": 4, "sizes": ["1K", "2K", "4K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image", "group": "Gemini Image",
            "desc": "穩定版，較成熟的圖像模型", "type": "t2i", "max_n": 4, "sizes": ["1K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-3.1-flash-lite-image", "name": "Gemini 3.1 Flash Lite Image", "group": "Gemini Image",
            "desc": "輕量極速圖像生成", "type": "t2i", "max_n": 4, "sizes": ["1K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        # ── GPT Image 編輯（沿用一般 /v1/images/edits 流程）──────────
        {
            "id": "gpt-image-2", "name": "GPT Image 2（編輯）", "group": "GPT Image",
            "desc": "OpenAI 旗艦圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True,
            "sizes": ["1024x1024","1536x1024","1024x1536"], "supports_gpt_params": True,
        },
        {
            "id": "gpt-image-1.5", "name": "GPT Image 1.5（編輯）", "group": "GPT Image",
            "desc": "OpenAI 前代圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True,
            "sizes": ["1024x1024","1536x1024","1024x1536"], "supports_gpt_params": True,
        },
        # ── ByteDance Seedream 編輯（沿用一般 /v1/images/edits 流程，實測 ref_strength
        #    參數有效不會被拒絕，因此不加進 _NO_REF_STRENGTH_EDIT_MODELS）──
        {
            "id": "dola-seedream-5.0-pro", "name": "Seedream 5.0 Pro（編輯）", "group": "ByteDance Seedream",
            "desc": "字節跳動 Seedream，旗艦圖像編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024x1024","1536x1024","1024x1536","2048x2048"],
        },
        {
            "id": "dola-seedream-5.0-lite", "name": "Seedream 5.0 Lite（編輯）", "group": "ByteDance Seedream",
            "desc": "字節跳動 Seedream，輕量圖像編輯（畫面較大，最小約 2K）", "type": "i2i", "max_n": 1,
            "sizes": ["2048x2048","2k","3k","4k"],
        },
        # ── MAI Image 編輯（Azure OpenAI 管道，沿用一般 /v1/images/edits 流程，不支援 ref_strength）──
        {
            "id": "MAI-Image-2.5-Pro", "name": "MAI-Image-2.5-Pro（編輯）", "group": "MAI Image",
            "desc": "旗艦圖像編輯 Pro", "type": "i2i", "max_n": 1, "no_ref_strength": True, "max_ref": 1,
            "sizes": _MAI_IMAGE_SIZES,
        },
        {
            "id": "MAI-Image-2.5", "name": "MAI-Image-2.5（編輯）", "group": "MAI Image",
            "desc": "旗艦圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True, "max_ref": 1,
            "sizes": _MAI_IMAGE_SIZES,
        },
        {
            "id": "MAI-Image-2.5-Flash", "name": "MAI-Image-2.5-Flash（編輯）", "group": "MAI Image",
            "desc": "極速圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True, "max_ref": 1,
            "sizes": _MAI_IMAGE_SIZES,
        },
        # ── Gemini Image 編輯（同樣走原生 generateContent，參考圖以 inlineData 帶入。
        #    實測編輯情境下 aspectRatio 一樣生效——舊路徑在有參考圖時是連比例參數都
        #    完全不處理的，所以編輯模式等於沒有任何輸出尺寸控制）──
        {
            "id": "gemini-3-pro-image", "name": "Gemini 3 Pro Image（編輯）", "group": "Gemini Image",
            "desc": "Google 旗艦圖像編輯，畫質最佳", "type": "i2i", "max_n": 1, "no_ref_strength": True, "sizes": ["1K", "2K", "4K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-3.1-flash-image", "name": "Gemini 3.1 Flash Image（編輯）", "group": "Gemini Image",
            "desc": "速度與品質平衡，建議日常使用", "type": "i2i", "max_n": 1, "no_ref_strength": True, "sizes": ["1K", "2K", "4K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image（編輯）", "group": "Gemini Image",
            "desc": "穩定版，較成熟的圖像模型", "type": "i2i", "max_n": 1, "no_ref_strength": True, "sizes": ["1K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
        {
            "id": "gemini-3.1-flash-lite-image", "name": "Gemini 3.1 Flash Lite Image（編輯）", "group": "Gemini Image",
            "desc": "輕量極速圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True, "sizes": ["1K"],
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        },
    ],
    "video": [
        # ── 阿里（萬相／HappyHorse）家族的兩個上游限制，已對照閘道 adaptor 原始碼確認：
        #    1. 配音開關：阿里 task adaptor 從頭到尾沒有讀取統一請求的頂層 audio 欄位，
        #       只有 wan2.6-i2v-flash 會去讀 metadata.audio（bool）。其餘萬相型號有沒有
        #       聲音完全由上游自己決定，送任何欄位都無效——所以除了 wan2.6-i2v-flash
        #       以外的阿里模型一律 audio: False，不顯示一個實際上沒有作用的開關。
        #    2. i2v 只吃首幀：adaptor 的 i2v 分支只取 images[0] 當 first_frame，尾幀／
        #       驅動音訊／影片延伸片段送過去都會被靜默丟棄，故以 i2v_modes 限制成
        #       只有「首幀生成」一種模式。
        # ── 萬相 3.0（All-in-One）──────────────────────────────────
        # 單一模型 id 統一支援文生／圖生／參考生／視頻編輯，模型名沒有 i2v/r2v/
        # videoedit 後綴，所以上游無法從模型名判斷每個媒體的用途，改由「MIME／副檔名
        # ＋ 位置」推斷（data URI 取 data: 與第一個 ;／, 之間的 MIME 判定，HTTP URL
        # 先切掉 query string 再比副檔名，判不出來才回退到位置）。我們送的 data URI
        # 是判得出來的，但仍改走上游提供的覆寫管道，直接以 metadata.input.media 送出
        # 我們自己已標好型別的陣列（見 _WAN30_ALLINONE_MODELS）——理由見該處註解。
        #
        # ⚠️ 尚未實測：wan3.0-video 的官方 API 文檔目前是邀請制、尚未公開。可公開
        #    查證的只有模型名、端點、resolution/ratio/duration 三個參數與定價；
        #    media 的 type 取值（first_frame/last_frame/driving_audio/first_clip/
        #    reference_image/video）是上游實作者從 wan2.7 已公開文件的詞彙推導的，
        #    雙方都沒有打過真的請求。上架後要實測確認，若上游回報 type 不合法，
        #    要把正確清單回報給閘道端校正。
        #
        # 費率（官方每秒單價）：480P $0.05 / 720P $0.10 / 1080P $0.20
        {"id": "wan3.0-video", "name": "萬相 3.0（文生影片）", "group": "萬相 3.0",
         "desc": "All-in-One 影片生成，最長 30 秒，音畫真實", "type": "t2v",
         "audio": False, "min_dur": 2, "max_dur": 30, "resolutions": ["480P", "720P", "1080P"]},
        {"id": "wan3.0-video", "name": "萬相 3.0（圖生影片）", "group": "萬相 3.0",
         "desc": "首幀／首尾幀／驅動音訊／影片延伸", "type": "i2v",
         "audio": False, "min_dur": 2, "max_dur": 30, "resolutions": ["480P", "720P", "1080P"]},
        {"id": "wan3.0-video", "name": "萬相 3.0（參考生影片）", "group": "萬相 3.0",
         "desc": "多圖參考生影片，生產級角色一致性", "type": "r2v",
         "audio": False, "min_dur": 2, "max_dur": 30, "resolutions": ["480P", "720P", "1080P"]},
        {"id": "wan3.0-video", "name": "萬相 3.0（視頻編輯）", "group": "萬相 3.0",
         "desc": "統一支援參考／編輯／複刻／驅動", "type": "vedit",
         "audio": False, "min_dur": 2, "max_dur": 30, "resolutions": ["480P", "720P", "1080P"]},
        # ── 文生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-t2v", "name": "萬相 2.7 T2V", "group": "文生影片",   "desc": "多鏡頭，自動配音", "type": "t2v",   "audio": False,  "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-t2v", "name": "萬相 2.6 T2V", "group": "文生影片",   "desc": "前代文生影片（配音由模型自動決定）",     "type": "t2v",   "audio": False, "min_dur": 2, "max_dur": 15},
        # ── 圖生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-i2v", "name": "萬相 2.7 I2V", "group": "圖生影片",   "desc": "首幀生成", "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15, "i2v_modes": ["first_frame"]},
        # wan2.6-i2v / wan2.6-i2v-flash 目前 NenAI 平台端 pipeline 故障（無論送任何欄位格式都回
        # "Field required: input.img_url"，已用直連 API 排除是本專案的請求格式問題），保留在清單中等待平台方修復
        {"id": "wan2.6-i2v", "name": "萬相 2.6 I2V", "group": "圖生影片",   "desc": "前代圖生影片，含音頻",       "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15, "i2v_modes": ["first_frame"]},
        {"id": "wan2.6-i2v-flash", "name": "萬相 2.6 I2V Flash", "group": "圖生影片", "desc": "前代圖生影片極速版，可關閉配音", "type": "i2v", "audio": True, "min_dur": 2, "max_dur": 15, "i2v_modes": ["first_frame"]},
        # ── 參考生影片 ────────────────────────────────────────────
        {"id": "wan2.7-r2v", "name": "萬相 2.7 R2V", "group": "參考生影片", "desc": "角色形象參考（僅接受圖片）",       "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15, "ref_images_only": True},
        {"id": "wan2.6-r2v", "name": "萬相 2.6 R2V", "group": "參考生影片", "desc": "前代參考生影片（僅接受圖片）",     "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15, "ref_images_only": True},
        {"id": "wan2.6-r2v-flash", "name": "萬相 2.6 R2V Flash", "group": "參考生影片", "desc": "前代參考生影片極速版（僅接受圖片）", "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15, "ref_images_only": True},
        # ── HappyHorse ────────────────────────────────────────────
         {"id": "happyhorse-1.1-t2v",        "name": "HappyHorse 1.1 T2V",        "group": "HappyHorse", "desc": "高還原度文生影片",          "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 15},    
        {"id": "happyhorse-1.0-t2v",        "name": "HappyHorse 1.0 T2V",        "group": "HappyHorse", "desc": "前一代高還原度文生影片",          "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.1-i2v",        "name": "HappyHorse 1.1 I2V",        "group": "HappyHorse", "desc": "高還原度圖生影片（僅首幀）",   "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 15, "i2v_modes": ["first_frame"]},
        {"id": "happyhorse-1.0-i2v",        "name": "HappyHorse 1.0 I2V",        "group": "HappyHorse", "desc": "前一代高還原度圖生影片（僅首幀）",   "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 15, "i2v_modes": ["first_frame"]},
        {"id": "happyhorse-1.1-r2v",        "name": "HappyHorse 1.1 R2V",        "group": "HappyHorse", "desc": "多圖參考生影片（最多 9 張，僅接受圖片）", "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 15, "ref_images_only": True, "max_ref": 9},
        {"id": "happyhorse-1.0-r2v",        "name": "HappyHorse 1.0 R2V",        "group": "HappyHorse", "desc": "前一代多圖參考生影片（最多 9 張，僅接受圖片）", "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 15, "ref_images_only": True, "max_ref": 9},
        {"id": "happyhorse-1.0-video-edit", "name": "HappyHorse Video Edit 1.0", "group": "HappyHorse", "desc": "視頻編輯（最多 5 張參考圖）", "type": "vedit", "audio": False, "min_dur": 3, "max_dur": 15, "max_ref": 5},      
        # ── 視頻編輯 ──────────────────────────────────────────────
        {"id": "wan2.7-videoedit", "name": "萬相 2.7 視頻編輯", "group": "萬相視頻編輯",
         "desc": "文字/參考圖驅動編輯", "type": "vedit", "audio": False, "min_dur": 2, "max_dur": 15},
        # ── 動作動畫（視頻換人 / 圖生動作）──────────────────────────
        {"id": "wan2.2-animate-mix", "name": "萬相 2.2 視頻換人", "group": "萬相動作動畫",
         "desc": "將參考影片中的角色替換為人物圖片，保留原場景與動作", "type": "animate", "audio": False},
        {"id": "wan2.2-animate-move", "name": "萬相 2.2 圖生動作", "group": "萬相動作動畫",
         "desc": "將參考影片的動作與表情遷移到人物圖片", "type": "animate", "audio": False},
        # ── Veo（duration 僅接受 4/6/8 秒，dur_step 供前端滑桿對齊）──
        {"id": "veo-3.1-generate-001", "name": "Veo 3.1", "group": "Veo",
         "desc": "Google 旗艦文生影片，含原生配音", "type": "t2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-fast-generate-001", "name": "Veo 3.1 Fast", "group": "Veo",
         "desc": "Google 極速文生影片，含原生配音", "type": "t2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-lite-generate-001", "name": "Veo 3.1 Lite", "group": "Veo",
         "desc": "Google 輕量文生影片，含原生配音", "type": "t2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-generate-001", "name": "Veo 3.1（圖生影片）", "group": "Veo",
         "desc": "Google 旗艦圖生影片，含原生配音", "type": "i2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-fast-generate-001", "name": "Veo 3.1 Fast（圖生影片）", "group": "Veo",
         "desc": "Google 極速圖生影片，含原生配音", "type": "i2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-lite-generate-001", "name": "Veo 3.1 Lite（圖生影片）", "group": "Veo",
         "desc": "Google 輕量圖生影片，含原生配音", "type": "i2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-generate-001", "name": "Veo 3.1（參考生影片）", "group": "Veo",
         "desc": "Google 旗艦參考生影片，含原生配音", "type": "r2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-fast-generate-001", "name": "Veo 3.1 Fast（參考生影片）", "group": "Veo",
         "desc": "Google 極速參考生影片，含原生配音", "type": "r2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        {"id": "veo-3.1-lite-generate-001", "name": "Veo 3.1 Lite（參考生影片）", "group": "Veo",
         "desc": "Google 輕量參考生影片，含原生配音", "type": "r2v", "audio": True, "min_dur": 4, "max_dur": 8, "dur_step": 2},
        # ── ByteDance Seedance（字節跳動/即夢文生/圖生/參考生影片，走一般 /v1/videos
        #    任務制流程——跟萬相系列共用的 media/image/images 三欄位注入機制。
        #    t2v 三個模型都實測過；i2v 在 bytedance-seedance-1.5-pro 上實測完整跑到
        #    completed 拿到影片網址，r2v 在 dreamina-seedance-2.0-fast 上實測完整跑到
        #    completed；其餘模型 × 模式的組合沒有逐一窮舉，是基於同一套通用機制推斷
        #    同樣可用。duration 上限沿用其餘家族常見範圍，未測邊界值；
        #    視頻編輯 vedit 實測會直接被上游拒絕（image_url 參數不合法），不列入。
        #
        #    解析度支援度已逐一實測（2026-08-10，對正式網關送出、看上游收不收）：
        #      dreamina-seedance-2.0-fast  480P ✓  720P ✓  1080P ✗  4K ✗（t2v/i2v/r2v 三種模式都拒）
        #      dreamina-seedance-2.0       480P ✓  720P ✓  1080P ✓  4K ✓
        #      bytedance-seedance-1.5-pro  480P ✓  720P ✓  1080P ✓  4K ✗
        #    超出範圍會回 InvalidParameter「the parameter resolution ... is not valid」。
        #    先前送不到解析度時這個限制看不出來（上游一律當 720p 跑），是把解析度真的
        #    送達之後才浮現的，所以用 resolutions 明確限制住 fast 版的選項）──
        # ── Dreamina Seedance 2.5（2026-08-11 對**測試網關** 192.168.0.245 驗證；
        #    正式環境的模型清單裡雖然有它，但網關程式碼還沒部署，那邊叫不動）──
        # 與 2.0 的差異大到 UI 必須分開處理：
        #   解析度 **只有 480p/720p**（送 1080p/4k 實測回 InvalidParameter），2.0 則到 4K
        #   duration 範圍 [4,30] 或 -1（送 3 或 31 都被拒），2.0 是 2~15
        #   參考素材上限高很多：30 張圖 / 10 支影片 / 10 段音訊，且**支援純音訊輸入**
        #   不支援 camera_fixed / frames / draft（實測都回 InvalidParameter）
        #   ⚠️ peer 說 seed 也不支援，但**實測沒有被拒**（任務照樣建立），與其規格不符
        # 計費比 2.0 貴約 53%（720p 每秒 $0.2311 vs $0.1512），tokens = 寬×高×幀數/1024
        #
        # ── 配音與兩個無效參數（2026-08-12，依閘道 doubao adaptor 原始碼盤點，⚠️ 未實測）──
        # 1. 先前所有 seedance 都標 audio: False，前端不但隱藏開關、還會強制把勾選狀態設成
        #    false（app.js 的 onVidModelChange），於是每一次請求都明確送出
        #    metadata.generate_audio=false。而上游這個欄位的預設值是 **true**，等於平台把
        #    本來會有的聲音主動關掉了，而且使用者沒有辦法開啟。管線其實早就接好（
        #    _apply_audio_flag 當初就把 generate_audio 帶上了），缺的只是這個旗標。
        #    改成 audio: True 把控制權交還使用者；**預設仍維持不勾選**（與上游預設相反，
        #    但與平台其餘型號一致，且無聲對 1.5-pro 有 0.5x 折扣）。
        # 2. doubao 的請求結構裡**沒有** negative_prompt 與 prompt_extend 這兩個欄位，
        #    文字內容只用到 prompt（adaptor.go 的 ContentItem.Text）。metadata 是整包
        #    unmarshal 進結構、對不上的鍵直接丟棄，所以這兩個控制項對 seedance 是純裝飾。
        #    以 no_negative_prompt / no_prompt_extend 標記，前端據此隱藏。
        # ⚠️ 以上兩點都是讀閘道 Go 原始碼推斷的（memory.md 4d 的第④類「間接證據」），
        #    影片生成單價高，這次未實測。配音實際會不會出聲、關掉是否真的折價，仍待驗證。
        {"id": "dreamina-seedance-2.5", "name": "Seedance 2.5（即夢）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance 最新版，最長 30 秒，支援 480P/720P", "type": "t2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True,
         "min_dur": 4, "max_dur": 30, "resolutions": ["480P", "720P"]},
        {"id": "bytedance-seedance-1.5-pro", "name": "Seedance 1.5 Pro", "group": "ByteDance Seedance",
         "desc": "字節跳動 Seedance，旗艦文生影片", "type": "t2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0", "name": "Seedance 2.0（即夢）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，標準文生影片", "type": "t2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0-fast", "name": "Seedance 2.0 Fast（即夢）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，極速文生影片，支援 480P/720P", "type": "t2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True,
         "min_dur": 2, "max_dur": 15, "resolutions": ["480P", "720P"]},
        {"id": "bytedance-seedance-1.5-pro", "name": "Seedance 1.5 Pro（圖生影片）", "group": "ByteDance Seedance",
         "desc": "字節跳動 Seedance，旗艦圖生影片（首幀）", "type": "i2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0", "name": "Seedance 2.0（即夢，圖生影片）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，標準圖生影片（首幀）", "type": "i2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0-fast", "name": "Seedance 2.0 Fast（即夢，圖生影片）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，極速圖生影片（首幀），支援 480P/720P", "type": "i2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True,
         "min_dur": 2, "max_dur": 15, "resolutions": ["480P", "720P"]},
        {"id": "bytedance-seedance-1.5-pro", "name": "Seedance 1.5 Pro（參考生影片）", "group": "ByteDance Seedance",
         "desc": "字節跳動 Seedance，旗艦參考生影片", "type": "r2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0", "name": "Seedance 2.0（即夢，參考生影片）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，標準參考生影片", "type": "r2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True, "min_dur": 2, "max_dur": 15},
        {"id": "dreamina-seedance-2.0-fast", "name": "Seedance 2.0 Fast（即夢，參考生影片）", "group": "ByteDance Seedance",
         "desc": "即夢 Seedance，極速參考生影片，支援 480P/720P", "type": "r2v",
         "audio": True, "no_negative_prompt": True, "no_prompt_extend": True,
         "min_dur": 2, "max_dur": 15, "resolutions": ["480P", "720P"]},
        # ── Gemini Omni（走 /v1beta/interactions，模型自行決定長度/解析度，固定含原生配音）──
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview", "group": "Gemini",
         "desc": "Google 多模態影片生成（預覽版），最長約 10 秒，自動含原生配音（無需另設定）", "type": "t2v", "audio": False, "no_duration": True},
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview（圖生影片）", "group": "Gemini",
         "desc": "Google 多模態圖生影片（預覽版），最長約 10 秒，自動含原生配音（無需另設定）", "type": "i2v", "audio": False, "no_duration": True},
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview（參考生影片）", "group": "Gemini",
         "desc": "Google 多模態參考生影片（預覽版，最多 3 張參考圖），最長約 10 秒，自動含原生配音（無需另設定）", "type": "r2v", "audio": False, "no_duration": True, "max_ref": 3},
    ],
    "muleai": [
        {"id": "wan2.7-i2v-spicy",       "name": "Wan 2.7 I2V Spicy",  "group": "影片生成", "desc": "Spicy 模型 (支援文字/圖片)"},
        {"id": "z-image-spicy",           "name": "Z-Image Spicy",      "group": "圖片生成", "desc": "Spicy 圖片生成模型"},
        {"id": "qwen-image-edit-spicy",   "name": "圖像編輯 Spicy",     "group": "圖像編輯", "desc": "Spicy 圖像編輯模型 (prompt + 來源圖)"},
        {"id": "face-swap",               "name": "圖像換臉",            "group": "圖像換臉", "desc": "換臉模型 (來源圖 + 換臉參考圖)"},
    ],
    "voice": {
        "asr": [
            {"id": "qwen-audio-3.0-asr-flash", "name": "Qwen Audio 3.0 ASR Flash", "group": "語音辨識",
             "desc": "極速語音辨識，上傳完整音檔一次回傳逐字稿"},
            {"id": "qwen-audio-3.0-asr-flash-streaming", "name": "Qwen Audio 3.0 ASR Flash（串流）", "group": "語音辨識",
             "desc": "串流語音辨識，逐字回傳中間辨識結果"},
        ],
        "tts": [
            {"id": "qwen-audio-3.0-tts-plus", "name": "Qwen Audio 3.0 TTS Plus", "group": "語音合成",
             "desc": "高品質語音合成", "vendor": "qwen", "voices": _QWEN_TTS_PLUS_VOICES},
            {"id": "qwen-audio-3.0-tts-flash", "name": "Qwen Audio 3.0 TTS Flash", "group": "語音合成",
             "desc": "極速語音合成", "vendor": "qwen", "voices": _QWEN_TTS_FLASH_VOICES},
            {"id": "gemini-2.5-pro-tts", "name": "Gemini 2.5 Pro TTS", "group": "Gemini",
             "desc": "Google 旗艦語音合成", "vendor": "gemini", "voices": _GEMINI_TTS_VOICES},
            {"id": "gemini-2.5-flash-tts", "name": "Gemini 2.5 Flash TTS", "group": "Gemini",
             "desc": "Google 極速語音合成", "vendor": "gemini", "voices": _GEMINI_TTS_VOICES},
            {"id": "gemini-3.1-flash-tts-preview", "name": "Gemini 3.1 Flash TTS Preview", "group": "Gemini",
             "desc": "Google 新一代極速語音合成（預覽版）", "vendor": "gemini", "voices": _GEMINI_TTS_VOICES},
        ],
    },
}


# ─── Auth: API Key per user ────────────────────────────────────────
def get_api_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        api_key = auth[7:].strip()
    else:
        api_key = request.query_params.get("api_key", "").strip()
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Unauthorized - missing API key")
    return api_key


# ─── Pages ────────────────────────────────────────────────────────
@app.get("/robots.txt")
async def robots_txt():
    return FileResponse(Path(__file__).parent / "static" / "robots.txt", media_type="text/plain")

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "templates" / "index.html")

@app.get("/canvas")
async def canvas_page():
    return FileResponse(Path(__file__).parent / "templates" / "canvas.html")

class LoginRequest(BaseModel):
    api_key: str

# 登入失敗鎖定：純記憶體、依來源 IP 計數，重試超過 5 次鎖定 5 分鐘。這裡沒有資料庫，
# 且正式環境是 Cloud Run（min-instances=0、可能多實例），這份計數不會跨實例共享、
# 服務重啟或縮容到 0 也會清空——只是盡力而為地拉高brute-force門檻，不是嚴格保證。
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_LOCKOUT_SECONDS = 5 * 60
_login_attempts: Dict[str, Dict[str, float]] = {}  # ip -> {"count": int, "locked_until": float}

def _login_client_ip(request: Request) -> str:
    # Cloud Run 前面有 Load Balancer/CDN，真實來源 IP 落在 X-Forwarded-For 第一段
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _login_record_failure(ip: str) -> int:
    """回傳這次失敗後、鎖定前還剩幾次機會（已鎖定則回傳 0）。"""
    entry = _login_attempts.setdefault(ip, {"count": 0, "locked_until": 0.0})
    entry["count"] += 1
    if entry["count"] >= _LOGIN_MAX_ATTEMPTS:
        entry["locked_until"] = time.time() + _LOGIN_LOCKOUT_SECONDS
        entry["count"] = 0
        return 0
    return _LOGIN_MAX_ATTEMPTS - entry["count"]

@app.post("/login")
async def login(data: LoginRequest, request: Request):
    """Validate NenAI API key."""
    ip = _login_client_ip(request)
    now = time.time()
    entry = _login_attempts.get(ip)
    if entry and entry["locked_until"] > now:
        remaining = int(entry["locked_until"] - now) + 1
        return JSONResponse(status_code=429, content={
            "success": False, "locked": True, "retry_after": remaining,
            "message": f"登入失敗次數過多，請 {remaining} 秒後再試",
        })

    api_key = data.api_key.strip()
    if not (api_key and len(api_key) > 10):
        return JSONResponse(status_code=400, content={"success": False, "message": "請輸入有效的 NenAI API Key"})

    # Light check - try to call OpenAI compatible listing
    try:
        test_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
        # We just check if the key works by doing a tiny completion
        test_client.chat.completions.create(
            model="qwen3.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            stream=False,
        )
        _login_attempts.pop(ip, None)
        return {"success": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err or "invalid" in err.lower():
            attempts_left = _login_record_failure(ip)
            if attempts_left <= 0:
                return JSONResponse(status_code=429, content={
                    "success": False, "locked": True, "retry_after": _LOGIN_LOCKOUT_SECONDS,
                    "message": f"登入失敗次數過多，請 {_LOGIN_LOCKOUT_SECONDS // 60} 分鐘後再試",
                })
            return JSONResponse(status_code=401, content={
                "success": False,
                "message": f"API Key 無效或權限不足（剩餘 {attempts_left} 次機會，超過將鎖定 5 分鐘）",
            })
        # Other errors (rate limit etc.) - key is likely valid
        return {"success": True}

# ─── API: Models ──────────────────────────────────────────────────
@app.get("/api/models")
async def get_models(api_key: str = Depends(get_api_key)):
    return MODELS


# ─── API: Pricing（供前端顯示各模型參考單價，資料來源是網關自己的計費表）───
# 網關的 /api/pricing 是「New API」這類閘道專案的標準端點，回傳所有模型的計價
# 資訊。換算成美金的公式（已用 quota_per_unit=500000、group_ratio=1 實測反推
# 確認）：quota_type=1（圖片/影片/Spicy 等）的 model_price 本身就是每次呼叫的
# 美金價，不用換算；quota_type=0（文字/語音等 token 計費）則是
#   每 1M input token 美金 = model_ratio × 2 × group_ratio
#   每 1M output token 美金 = model_ratio × completion_ratio × 2 × group_ratio
# 這裡固定假設 group_ratio=1（實測目前所有分組確實都是 1），只當作參考價格，
# 不是精確帳單金額——快取 1 小時，避免每次載入頁面都打一次上游。
_PRICING_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
_PRICING_CACHE_TTL = 3600

async def _fetch_pricing_map(api_key: str) -> dict:
    now = time.time()
    if _PRICING_CACHE["data"] is not None and now - _PRICING_CACHE["ts"] < _PRICING_CACHE_TTL:
        return _PRICING_CACHE["data"]
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{NENAI_BASE}/api/pricing", headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        raw = resp.json()
    result = {}
    for m in raw.get("data", []):
        mid = m.get("model_name")
        if not mid:
            continue
        if m.get("quota_type") == 1:
            # 不要粗暴 round 到固定小數位——像語音辨識這類每次呼叫只要 $0.000035 的
            # 模型，round(x, 4) 會直接捨去變成 0，讓使用者誤以為免費。保留原始精度，
            # 顯示位數交給前端依數值大小動態決定。
            result[mid] = {"type": "fixed", "price": m.get("model_price", 0) or 0}
        else:
            model_ratio = m.get("model_ratio", 0) or 0
            completion_ratio = m.get("completion_ratio", 1) or 1
            result[mid] = {
                "type": "token",
                "input": model_ratio * 2,
                "output": model_ratio * completion_ratio * 2,
            }
    _PRICING_CACHE["data"] = result
    _PRICING_CACHE["ts"] = now
    return result

@app.get("/api/pricing")
async def get_pricing(api_key: str = Depends(get_api_key)):
    try:
        return await _fetch_pricing_map(api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Canvas 節點間傳遞的圖片/影片網址可能落在雲端物件儲存 bucket，瀏覽器直接 fetch 會被
# CORS 擋下，故提供一個僅允許白名單網域的代理端點；不可放行任意網址，否則會變成 SSRF 入口。
# 白名單涵蓋三個雲端儲存後端目前會產生簽名網址的網域：阿里雲 OSS、AWS S3、GCS
# （GCS 不論走本地私鑰簽章還是 ADC + IAM SignBlob 遠端簽章，網址都落在
# storage.googleapis.com）——一開始只寫了 OSS 的網域，導致正式環境改用 GCS 之後，
# AI Canvas 裡任何需要把上一個節點的生成結果重新抓回來當輸入的功能（例如影片延伸
# 接上一段影片、把生成的圖片再送進另一個節點）都會在這裡被拒絕，噴「無法取得來源檔案」。
# 這份白名單是刻意設計來防 SSRF 的，放寬要特別小心。除了我們自己的三個雲端儲存
# 後端，還要涵蓋**上游模型直接回傳產出網址的網域**——後端雖然會先把產出下載回本機
# 再給前端，但下載失敗時會退回原始網址（見 video_status 的 `local if local else
# video_url`），這時 AI Canvas 把上一步結果接成下一步輸入就會走到這支代理。
#   *.aliyuncs.com        阿里 OSS（萬相／千問的產出）
#   *.amazonaws.com       AWS S3
#   storage.googleapis.com GCS
#   *.volces.com          火山引擎 TOS（Seedance 系列的產出，實測網址形如
#                         ark-acg-ap-southeast-1.tos-ap-southeast-1.volces.com）
_PROXY_ALLOWED_SUFFIXES = (".aliyuncs.com", ".amazonaws.com", "storage.googleapis.com",
                           ".volces.com")

@app.get("/api/proxy/fetch")
async def proxy_fetch(url: str, api_key: str = Depends(get_api_key)):
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if not host or not any(host == s.lstrip(".") or host.endswith(s) for s in _PROXY_ALLOWED_SUFFIXES):
        raise HTTPException(status_code=400, detail="URL host not allowed")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail="Upstream fetch failed")
        return Response(content=r.content, media_type=r.headers.get("content-type", "application/octet-stream"))

# ─── API: Text Generation (SSE Streaming) ─────────────────────────
class TextGenerateRequest(BaseModel):
    model: str = "qwen3.5-flash"
    prompt: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: Optional[int] = None
    max_tokens: int = 4096
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    stop: List[str] = []
    stream: bool = True
    enable_thinking: bool = False
    reasoning_effort: Optional[str] = None  # 僅 GPT 系列支援：none/low/medium/high/xhigh
    history: List[Dict[str, str]] = []  # 多輪對話歷史，[{"role": "user"/"assistant", "content": "..."}]
    images: List[str] = []              # 視覺語言模型（qwen3-vl-*）用：data URI 或公開網址


class OmniChatRequest(BaseModel):
    model: str = "qwen3.5-omni-flash"
    messages: list = []
    voice: str = "Ethan"
    instructions: Optional[str] = None

@app.post("/api/omni/chat")
async def omni_chat(data: OmniChatRequest, api_key: str = Depends(get_api_key)):
    msgs = []
    if data.instructions:
        msgs.append({"role": "system", "content": data.instructions})
    msgs.extend(data.messages)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            user_client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            stream = await user_client.chat.completions.create(
                model=data.model,
                messages=msgs,
                modalities=["text", "audio"],
                audio={"voice": data.voice, "format": "pcm16"},
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # text content
                text_content = getattr(delta, "content", None)
                if text_content:
                    yield f"data: {json.dumps({'type': 'text', 'content': text_content})}\n\n"

                # audio — may be attribute or in model_extra (depends on SDK version)
                audio = getattr(delta, "audio", None)
                if audio is None:
                    extra = getattr(delta, "model_extra", None) or {}
                    audio = extra.get("audio")

                if audio:
                    # handle both object-style and dict-style
                    audio_data   = audio.get("data")     if isinstance(audio, dict) else getattr(audio, "data", None)
                    audio_trans  = audio.get("transcript") if isinstance(audio, dict) else getattr(audio, "transcript", None)
                    if audio_data:
                        yield f"data: {json.dumps({'type': 'audio', 'data': audio_data})}\n\n"
                    if audio_trans:
                        yield f"data: {json.dumps({'type': 'transcript', 'content': audio_trans})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ⚠️ 這支 realtime 代理目前**沒有任何 UI 入口**，而且上游也還不支援——保留是因為
# 阿里那邊補上 realtime 之後就能直接用，不必重寫。現況（2026-08-11 對正式環境實測
# ＋ 網關端查證原始碼確認）：
#
#   - 網關的 WebSocket 路由只有 `/v1/realtime` 與 `/v1/responses` 兩條。原本這裡寫死的
#     `/api-ws/v1/realtime` **從來就不是網關對外的路徑**（那是 DashScope 上游給 TTS/ASR
#     用的內部路徑），所以一直回 404。已改成正確的 `/v1/realtime`。
#   - 但改對路徑也還不能用：realtime 中繼**只有 OpenAI 系的 adaptor 有實作**，
#     `qwen3.5-omni-*-realtime` 屬於阿里渠道，而阿里 adaptor 的 WebSocket 支援只涵蓋
#     TTS 與 ASR，沒有 realtime 分支。請求會掉進一般 HTTP 路徑，導致網關端型別斷言
#     失敗而 panic——這就是實測看到「握手成功（HTTP 101）後立刻斷線、連 close frame
#     都沒有」的原因。網關端已修成回傳可讀的錯誤，但那只是讓失敗可診斷，不會讓
#     qwen omni realtime 變得可用。
#
# 所以前端的 realtime UI 在 commit 8c012ac 之後被移除是**正確的決定**：那個功能當時
# 就是壞的。要恢復需要阿里 adaptor 先補上整套 realtime 支援（撥上游 WS、DashScope
# realtime 協定與 OpenAI realtime 事件格式互轉、usage 計費），那是獨立的功能開發。
@app.websocket("/ws/omni")
async def ws_omni_proxy(websocket: WebSocket, api_key: str, model: str = "qwen3.5-omni-flash-realtime"):
    await websocket.accept()
    url = f"{NENAI_BASE.replace('https://', 'wss://').replace('http://', 'ws://')}/v1/realtime?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with websockets.connect(url, additional_headers=headers) as target_ws:
            async def forward_to_target():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await target_ws.send(data)
                except Exception:
                    pass

            async def forward_to_client():
                try:
                    while True:
                        data = await target_ws.recv()
                        await websocket.send_text(data)
                except Exception:
                    pass
            
            task1 = asyncio.create_task(forward_to_target())
            task2 = asyncio.create_task(forward_to_client())
            await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
            task1.cancel()
            task2.cancel()
    except Exception as e:
        print(f"WS proxy error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

# ─── Gemini 文字模型走 Gemini 原生 API ────────────────────────────────────────
# 端點 /v1beta/models/{model}:generateContent（串流是 :streamGenerateContent?alt=sse）。
# 先前跟其他家族一樣走 OpenAI 相容的 /v1/chat/completions，那條路徑上 Gemini 的思考
# 過程既看不到、也關不掉（README 一度據此寫「Gemini 3.x 無條件思考、關不掉」）。
# 2026-08-10 實測原生端點兩件事都做得到：
#   generationConfig.thinkingConfig.includeThoughts=true → 回傳帶 "thought": true 的
#     content part，裡面就是思考過程全文；串流時同樣以 thought part 逐段送出
#   generationConfig.thinkingConfig.thinkingBudget=0     → thoughtsTokenCount 消失，
#     思考真的被關掉（同一題原本要花 300~800 個 thought token）
# 但支援度**各型號不同**，送錯會直接 400，所以下面兩個例外集合是必要的。
_GEMINI_NATIVE_TEXT_MODELS = {
    m["id"] for m in MODELS["text"] if m["id"].startswith("gemini-")
}
# gemini-2.5-pro 不接受 thinkingBudget=0（回 "The model does not support setting
# thinking_budget to 0"）——它的思考關不掉，只能顯示過程
_GEMINI_NO_THINKING_OFF = {"gemini-2.5-pro"}
# gemini-2.5-flash-lite 不接受 includeThoughts（回 "Thinking_config.include_thoughts
# is not supported"）——它的思考可以關掉，但過程拿不到
_GEMINI_NO_INCLUDE_THOUGHTS = {"gemini-2.5-flash-lite"}
# 這些型號的思考**預設是關的**（不帶 thinkingConfig 時 thoughtsTokenCount 是 None），
# 要送 thinkingBudget: -1（動態預算）才會真的啟動——否則「思考開」那一檔會完全沒有
# 作用。其餘型號預設就會思考，只要 includeThoughts 把過程要出來即可。
_GEMINI_THINKING_OFF_BY_DEFAULT = {"gemini-2.5-flash-lite", "gemini-3.5-flash-lite"}


def _build_gemini_body(data: "TextGenerateRequest", messages: list) -> dict:
    """把內部的 OpenAI 風格請求轉成 Gemini 原生格式。"""
    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue          # Gemini 的 system 是頂層 systemInstruction
        contents.append({"role": "model" if role == "assistant" else "user",
                         "parts": [{"text": m.get("content") or ""}]})

    gen: dict = {
        "temperature": data.temperature,
        "topP": data.top_p,
        "maxOutputTokens": data.max_tokens,
    }
    if data.top_k is not None and data.top_k > 0:
        gen["topK"] = data.top_k
    if data.stop:
        gen["stopSequences"] = data.stop[:4]
    if data.seed is not None:
        gen["seed"] = data.seed
    if data.presence_penalty:
        gen["presencePenalty"] = data.presence_penalty
    if data.frequency_penalty:
        gen["frequencyPenalty"] = data.frequency_penalty

    thinking: dict = {}
    if data.enable_thinking:
        if data.model in _GEMINI_THINKING_OFF_BY_DEFAULT:
            thinking["thinkingBudget"] = -1
        if data.model not in _GEMINI_NO_INCLUDE_THOUGHTS:
            thinking["includeThoughts"] = True
    else:
        if data.model in _GEMINI_NO_THINKING_OFF:
            # 關不掉，那就至少把過程顯示出來，而不是白花 token 又看不到
            if data.model not in _GEMINI_NO_INCLUDE_THOUGHTS:
                thinking["includeThoughts"] = True
        else:
            thinking["thinkingBudget"] = 0
    if thinking:
        gen["thinkingConfig"] = thinking

    body: dict = {"contents": contents, "generationConfig": gen}
    if data.system_prompt:
        body["systemInstruction"] = {"parts": [{"text": data.system_prompt}]}
    return body


def _gemini_usage(um: dict) -> dict:
    """thoughtsTokenCount 也是實際計費的輸出 token，要一起算進 completion。"""
    return {"prompt_tokens": um.get("promptTokenCount", 0),
            "completion_tokens": (um.get("candidatesTokenCount", 0) or 0)
                                 + (um.get("thoughtsTokenCount", 0) or 0)}


async def _gemini_text_generate(data: "TextGenerateRequest", messages: list, api_key: str):
    body = _build_gemini_body(data, messages)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{NENAI_BASE}/v1beta/models/{data.model}:generateContent",
                                 headers=headers, json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
        rj = resp.json()
    texts, thoughts = [], []
    for cand in rj.get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            if not part.get("text"):
                continue
            (thoughts if part.get("thought") else texts).append(part["text"])
    result: dict = {"content": "".join(texts), "done": True}
    if thoughts:
        result["reasoning_content"] = "".join(thoughts)
    if rj.get("usageMetadata"):
        result["usage"] = _gemini_usage(rj["usageMetadata"])
    return result


async def _gemini_text_stream(data: "TextGenerateRequest", messages: list,
                              api_key: str) -> AsyncGenerator[str, None]:
    body = _build_gemini_body(data, messages)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{NENAI_BASE}/v1beta/models/{data.model}:streamGenerateContent?alt=sse"
    usage: dict = {}
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    yield f"data: {json.dumps({'error': detail})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for cand in ev.get("candidates", []):
                        for part in (cand.get("content") or {}).get("parts", []):
                            text = part.get("text")
                            if not text:
                                continue
                            key = "reasoning" if part.get("thought") else "content"
                            yield f"data: {json.dumps({key: text})}\n\n"
                    # usageMetadata 每個 chunk 都可能出現，但只有最後一個是完整的
                    um = ev.get("usageMetadata") or {}
                    if um.get("promptTokenCount"):
                        usage = _gemini_usage(um)
        done_payload: dict = {"done": True}
        if usage:
            done_payload["usage"] = usage
        yield f"data: {json.dumps(done_payload)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _openai_usage(usage) -> dict:
    """把 OpenAI 相容回應的 usage 轉成前端用的格式，並補上被漏掉的推理 token。

    **Grok 的推理 token 不計入 `completion_tokens`**（實測 grok-4.3：prompt 31 +
    completion 1 + reasoning 844 = total 876），但那些 token 是照樣要收費的。
    只讀 completion_tokens 的話，一次花了 844 個推理 token 的呼叫會被算成 1 個
    ——「本次花費」會嚴重低估。
    其餘家族（DeepSeek V4、GLM、Seed 2.0）的推理 token 本來就含在 completion_tokens
    裡，重複相加會變成兩倍，所以這裡用 total 反推：completion 取
    `total - prompt`，這對兩種帳法都正確。
    """
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    if total and total - prompt > completion:
        completion = total - prompt
    return {"prompt_tokens": prompt, "completion_tokens": completion}


@app.post("/api/text/generate")
async def text_generate(data: TextGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    messages = []
    if data.system_prompt:
        messages.append({"role": "system", "content": data.system_prompt})
    messages.extend(data.history)
    if data.images:
        # 視覺語言模型（qwen3-vl-*）：content 從字串改成陣列，圖片用標準的 OpenAI
        # image_url 格式帶入（實測 data URI 可用）。只有這一輪帶圖，history 裡的
        # 舊訊息維持純文字——上游對「歷史訊息裡的圖片」行為未驗證，不主動送。
        content: Any = [{"type": "image_url", "image_url": {"url": u}} for u in data.images]
        content.append({"type": "text", "text": data.prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": data.prompt})

    # Gemini 文字模型走 Gemini 原生 API
    if data.model in _GEMINI_NATIVE_TEXT_MODELS:
        if not data.stream:
            return await _gemini_text_generate(data, messages, api_key)
        return StreamingResponse(
            _gemini_text_stream(data, messages, api_key),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 這裡不能只在 enable_thinking=True 時才帶這個欄位——實測發現 qwen3.5-flash/
    # qwen3.6-flash/qwen3.8-max/deepseek-v4-*/glm-5.* 這些模型預設就是思考模式開啟，
    # 完全不帶 enable_thinking 欄位並不會關閉思考，只有明確送 enable_thinking:false
    # 才有效——原本的寫法導致使用者把「思考模式」開關關掉後其實毫無作用，模型仍在
    # 思考（多花 token、多等時間）。GPT 系列則是完全不同的機制（reasoning_effort），
    # 送 enable_thinking 給它會直接 400 "Unknown parameter"，因此只排除 GPT 系列，
    # 其餘家族（含 Claude/Gemini，實測過送 enable_thinking:false 不會報錯，只是沒
    # 有效果）一律明確帶上 True 或 False。
    extra_body = {}
    if not data.model.startswith("gpt-"):
        extra_body["enable_thinking"] = data.enable_thinking

    create_kwargs = dict(
        model=data.model,
        messages=messages,
        max_tokens=data.max_tokens,
        presence_penalty=data.presence_penalty,
        frequency_penalty=data.frequency_penalty,
        stream=data.stream,
        extra_body=extra_body or None,
    )
    # Claude 系列在此平台的 Bedrock 後端不接受 temperature/top_p（部分模型視為已棄用參數，
    # 部分模型不允許兩者同時指定），一律不送這兩個參數，讓後端使用預設取樣設定
    if not data.model.startswith("claude-"):
        create_kwargs["temperature"] = data.temperature
        create_kwargs["top_p"] = data.top_p
    if data.top_k is not None and data.top_k > 0:
        create_kwargs["extra_body"] = {**(extra_body or {}), "top_k": data.top_k}
    if data.seed is not None:
        create_kwargs["seed"] = data.seed
    if data.stop:
        create_kwargs["stop"] = data.stop[:4]
    if data.stream:
        # 要求上游在最後一個 chunk 附上 usage（token 數），供前端計算即時花費
        create_kwargs["stream_options"] = {"include_usage": True}
    # reasoning_effort 是 GPT-5 系列專屬的推理強度控制（實測這個網關接受的枚舉值是
    # none/low/medium/high/xhigh，不是 OpenAI 官方文件常見的 minimal/low/medium/high；
    # 帶 minimal 會被直接拒絕：400 "does not support 'minimal' with this model"），
    # 跟 Qwen/DeepSeek/GLM 用的 enable_thinking 是完全不同的機制、參數名稱衝突
    # 不能共用同一個開關：實測過對 GPT 模型送 enable_thinking 會直接 400
    # "Unknown parameter"，對其他家族送 reasoning_effort 則會被忽略或報錯，
    # 因此兩者互斥，由呼叫端（前端）依模型家族只送其中一個。
    if data.reasoning_effort:
        create_kwargs["reasoning_effort"] = data.reasoning_effort

    if not data.stream:
        try:
            user_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            resp = user_client.chat.completions.create(**create_kwargs)
            message = resp.choices[0].message if resp.choices else None
            content = message.content if message else ""
            # reasoning_content（DeepSeek/GLM 等會回的思考過程）不是 openai SDK
            # 正式定義的欄位，SDK 解析時會落在 model_extra 而不是屬性上
            reasoning = getattr(message, "reasoning_content", None) if message else None
            if reasoning is None and message is not None:
                reasoning = (getattr(message, "model_extra", None) or {}).get("reasoning_content")
            result = {"content": content, "done": True}
            if reasoning:
                result["reasoning_content"] = reasoning
            if resp.usage:
                result["usage"] = _openai_usage(resp.usage)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate() -> AsyncGenerator[str, None]:
        try:
            user_client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            stream = await user_client.chat.completions.create(**create_kwargs)
            usage = None
            async for chunk in stream:
                if chunk.usage:
                    usage = _openai_usage(chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is None:
                    reasoning = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                if reasoning:
                    yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            done_payload = {"done": True}
            if usage:
                done_payload["usage"] = usage
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── API: Vision（AI Canvas「圖片 → 文字」節點用，圖片以 base64 data URI 內嵌傳入）──
class AnalyzeImageRequest(BaseModel):
    model: str = "qwen3.5-flash"
    prompt: str = "請用一句話描述這張圖片的內容。"
    image_data_uri: str

@app.post("/api/text/analyze_image")
async def analyze_image(data: AnalyzeImageRequest, api_key: str = Depends(get_api_key)):
    if not data.image_data_uri:
        raise HTTPException(status_code=400, detail="image_data_uri is required")
    try:
        user_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
        resp = user_client.chat.completions.create(
            model=data.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": data.prompt or "請描述這張圖片"},
                {"type": "image_url", "image_url": {"url": data.image_data_uri}},
            ]}],
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return {"success": True, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ─── API: MuleAI Video/Image Generation ───────────────────────────────────────
@app.post("/api/muleai/generate")
async def muleai_generate(
    request: Request,
    model: str = Form("wan2.7-i2v-spicy"),
    prompt: str = Form(""),
    negative_prompt: Optional[str] = Form(None),
    resolution: str = Form("1080P"),
    duration: Optional[int] = Form(5),
    img_resolution: Optional[str] = Form("1024*1536"),
    prompt_extend: bool = Form(True),
    seed: Optional[int] = Form(None),
    enable_audio: bool = Form(False),
    image: Optional[UploadFile] = File(None),
    face_image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    api_key: str = Depends(get_api_key)
):
    is_face_swap    = model == "face-swap"
    is_img_edit     = model == "qwen-image-edit-spicy"
    is_image_model  = "z-image" in model or is_img_edit or is_face_swap

    if not prompt and not is_face_swap:
        raise HTTPException(status_code=400, detail="Prompt is required")

    MULEAI_URL = (
        "https://nen.com.tw/v1/image/generations"
        if is_image_model
        else "https://nen.com.tw/v1/video/generations"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def _to_data_uri(f: UploadFile) -> str:
        raw = await f.read()
        mime = f.content_type or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    # ── face-swap ──────────────────────────────────────────────
    if is_face_swap:
        if not image or not image.filename:
            raise HTTPException(status_code=400, detail="來源圖片為必填")
        if not face_image or not face_image.filename:
            raise HTTPException(status_code=400, detail="換臉參考圖為必填")
        payload = {
            "model": model,
            "image": await _to_data_uri(image),
            "face_image": await _to_data_uri(face_image),
        }

    # ── qwen-image-edit-spicy ──────────────────────────────────
    elif is_img_edit:
        if not image or not image.filename:
            raise HTTPException(status_code=400, detail="來源圖片為必填")
        payload = {
            "model": model,
            "prompt": prompt,
            "image": await _to_data_uri(image),
            "prompt_extend": prompt_extend,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed

    # ── z-image-spicy ──────────────────────────────────────────
    elif "z-image" in model:
        payload = {"model": model, "prompt": prompt, "prompt_extend": prompt_extend}
        if img_resolution:
            parts = img_resolution.split("*")
            if len(parts) == 2:
                payload["width"] = int(parts[0])
                payload["height"] = int(parts[1])
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed

    # ── wan2.7-i2v-spicy（影片）────────────────────────────────
    else:
        if not image or not image.filename:
            raise HTTPException(status_code=400, detail="Image is required for video generation")
        payload = {
            "model": model,
            "prompt": prompt,
            "size": resolution,
            "duration": duration,
            "image": await _to_data_uri(image),
            "prompt_extend": prompt_extend,
        }
        if enable_audio:
            if audio and audio.filename:
                audio_bytes = await audio.read()
                audio_mime = audio.content_type or "audio/mpeg"
                payload["audio"] = f"data:{audio_mime};base64,{base64.b64encode(audio_bytes).decode()}"
            else:
                payload["audio"] = True
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(MULEAI_URL, headers=headers, json=payload)
            
            if resp.status_code in (200, 202):
                data = resp.json()
                task_id = data.get("task_info", {}).get("id") or data.get("id")
                if not task_id:
                    return JSONResponse(status_code=500, content={"success": False, "error": f"No task_id in response: {data}"})
                return {"success": True, "task_id": task_id, "status": "pending", "model": model}
            else:
                return JSONResponse(status_code=resp.status_code, content={"success": False, "error": resp.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/muleai/status/{model}/{task_id}")
async def muleai_task_status(model: str, task_id: str, api_key: str = Depends(get_api_key)):

    _is_img = "z-image" in model or model in ("qwen-image-edit-spicy", "face-swap")
    MULEAI_STATUS_URL = f"https://nen.com.tw/v1/{'image' if _is_img else 'video'}/generations/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(MULEAI_STATUS_URL, headers=headers)
            if resp.status_code == 200:
                rj = resp.json()
                # 實際結構：rj["data"] = outer（含 status/result_url）
                #           rj["data"]["data"] = inner（含 task_info/videos）
                outer = rj.get("data") if isinstance(rj.get("data"), dict) else rj
                inner = outer.get("data") if isinstance(outer.get("data"), dict) else {}
                status = (
                    inner.get("task_info", {}).get("status")
                    or outer.get("status")
                    or rj.get("status")
                    or "pending"
                )
                videos = inner.get("videos") or outer.get("videos") or rj.get("videos") or []
                images = inner.get("images") or outer.get("images") or rj.get("images") or []
                if not images and (inner.get("image_url") or outer.get("image_url")):
                    images = [inner.get("image_url") or outer.get("image_url")]
                # result_url fallback：圖片模型放 images，影片模型放 videos
                if outer.get("result_url"):
                    if _is_img and not images:
                        images = [outer.get("result_url")]
                    elif not _is_img and not videos:
                        videos = [outer.get("result_url")]
                err = inner.get("task_info", {}).get("error") or outer.get("fail_reason")

                if status.upper() in ("COMPLETED", "SUCCEEDED", "SUCCESS"):
                    if _is_img:
                        # 圖片模型：優先下載 images，若 API 誤放在 videos 則也抓過來
                        img_url = None
                        if images:
                            img_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
                        elif videos:
                            img_url = videos[0]
                            videos = []
                        if img_url:
                            local_path = await _async_download_image(img_url)
                            images = [local_path if local_path else img_url]
                    else:
                        if videos:
                            local_path = await _async_download_video(videos[0])
                            if local_path:
                                videos = [local_path]
                        elif images:
                            img_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
                            local_path = await _async_download_image(img_url)
                            if local_path:
                                images = [local_path]
                return {"success": True, "status": status, "videos": videos, "images": images, "error_message": err}

            else:
                return JSONResponse(status_code=resp.status_code, content={"success": False, "error": resp.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/muleai/debug/{model}/{task_id}")
async def muleai_debug(model: str, task_id: str, api_key: str = Depends(get_api_key)):
    """回傳平台原始 JSON，診斷 status 欄位位置。"""
    _is_img = "z-image" in model or model in ("qwen-image-edit-spicy", "face-swap")
    url = f"https://nen.com.tw/v1/{'image' if _is_img else 'video'}/generations/{task_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        return {"http_status": resp.status_code, "url": url, "raw": resp.json()}


# Gemini 圖像模型不走 /v1/images/generations，須改用 /v1/chat/completions + modalities
_GEMINI_CHAT_IMAGE_MODELS = {
    "gemini-3-pro-image", "gemini-3.1-flash-image",
    "gemini-2.5-flash-image", "gemini-3.1-flash-lite-image",
}
_B64_IMAGE_RE = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)")

async def _save_image_bytes(data: bytes, ext: str) -> Optional[str]:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"img_{ts}_{uuid.uuid4().hex[:6]}.{ext}"
        cloud_url = _cloud_put(data, f"images/{name}")
        if cloud_url:
            return cloud_url
        fp = OUTPUT_IMG_DIR / name
        fp.write_bytes(data)
        return f"/outputs/images/{fp.name}"
    except Exception as e:
        print(f"Image save error: {e}")
        return None

# OpenAI 相容的 /v1/images/generations 回應，圖片可能是 "url" 或直接內嵌的 "b64_json"
# （例如 gpt-image-2/1.5 預設就回 b64_json），兩種都要處理，否則會誤判為生成失敗
def _image_usage(rj: dict) -> Optional[dict]:
    """把圖片端點回傳的 usage 正規化成 {prompt_tokens, completion_tokens}。

    按 token 計費的圖片模型（MAI 三個、GPT Image 兩個、Gemini Image 四個）先前
    完全沒有把 usage 帶回前端，導致「本次花費」**完全不累加**這九個模型的花費。
    直接用上游回報的 token 數，比在前端維護一份「解析度→token」的換算表可靠——
    不必猜尺寸，也不會因為上游改了計算方式而失準。

    兩種欄位命名（都實測過）：
      OpenAI 相容端點：num_input_text_tokens / num_input_image_tokens / num_output_tokens
      Gemini 原生端點：promptTokenCount / candidatesTokenCount（由 _gemini_usage 處理）
    """
    u = rj.get("usage") or (rj.get("metadata") or {}).get("usage") or {}
    if not isinstance(u, dict) or not u:
        return None
    if "num_output_tokens" in u or "num_input_text_tokens" in u:
        return {"prompt_tokens": (u.get("num_input_text_tokens", 0) or 0)
                                 + (u.get("num_input_image_tokens", 0) or 0),
                "completion_tokens": u.get("num_output_tokens", 0) or 0}
    # 已經是標準命名就直接用
    if "prompt_tokens" in u or "completion_tokens" in u:
        return {"prompt_tokens": u.get("prompt_tokens", 0) or 0,
                "completion_tokens": u.get("completion_tokens", 0) or 0}
    return None

async def _extract_images_from_data(data_list: list) -> list:
    images = []
    for item in data_list:
        url = item.get("url")
        if url:
            images.append({"url": url, "local_path": await _async_download_image(url), "actual_prompt": item.get("actual_prompt")})
            continue
        b64 = item.get("b64_json")
        if b64:
            raw = base64.b64decode(b64)
            images.append({"url": None, "local_path": await _save_image_bytes(raw, "png"), "actual_prompt": item.get("actual_prompt")})
    return images


async def _extract_images_from_metadata(rj: dict, already: list) -> list:
    """從 metadata.output.choices 補抓 data[] 漏掉的圖片。

    千問 3.0 這條 multimodal-generation adaptor 在 n>1 時，OpenAI 相容的 `data[]`
    **只會回第一張**，但 `usage.output_image_count` 是實際張數、也照那個張數計費——
    上游確實產了 n 張，全部都在 metadata.output.choices[].message.content 裡。
    只讀 data[] 的話，使用者選 n=2 會被扣 2 張的錢卻只看到 1 張。
    （萬相與千問 2.0 走另一條 adaptor，data[] 本來就是完整的，不受影響。）
    """
    seen = {img.get("url") for img in already if img.get("url")}
    extra = []
    for choice in ((rj.get("metadata") or {}).get("output") or {}).get("choices") or []:
        for item in (choice.get("message") or {}).get("content") or []:
            url = item.get("image")
            if not url or url in seen:
                continue
            seen.add(url)
            extra.append({"url": url, "local_path": await _async_download_image(url),
                          "actual_prompt": None})
    return extra

# Gemini 圖像模型偶爾會不出圖、只回一段純文字聊天式回覆——實測發現這跟 prompt
# 讀起來像不像「聊天訊息」高度相關：越像一段對話/討論文字（例如上游文字節點
# 生成的長篇分析），模型就越容易把它當成聊天來回覆而不畫圖。加上明確的繪圖
# 指令前綴可顯著改善成功率，仍會不穩定則再靠重試補強。
_GEMINI_IMAGE_MAX_RETRIES = 2

# ─── Gemini 圖像尺寸的運作方式（2026-08-10 對正式網關實測歸納）────────────────
# 輸出像素**不是**直接指定寬高，而是由兩個參數共同決定：
#   imageConfig.imageSize   = 總像素預算（1K ≈ 105 萬、2K ≈ 4×、4K ≈ 16×）
#   imageConfig.aspectRatio = 形狀
# 上游取「符合該比例、且總像素最接近預算」的一組寬高，且兩邊都對齊到 16 的倍數。
# 以 1K 實測（gemini-3.1-flash-lite-image）：
#   1:1  1024x1024 (1,048,576)   4:3  1200x896  (1,075,200)   3:4  896x1200
#   3:2  1264x848  (1,071,872)   2:3  848x1264  (1,071,872)
#   5:4  1152x928  (1,069,056)   4:5  928x1152  (1,069,056)
#   16:9 1376x768  (1,056,768)   9:16 768x1376  (1,056,768)   21:9 1584x672 (1,064,448)
# 注意「4K」是像素預算而不是 UHD 解析度——4K + 16:9 實測是 5504x3072（約 1,690 萬
# 像素），不是 3840x2160。想要精確的寬高就得自己換算，這個 API 沒有直接指定的方式。
#
# imageSize 各型號支援度不同（每個型號 × 每個值都實際產圖量過寬高）：
#   gemini-3-pro-image / gemini-3.1-flash-image  1K/2K/4K 都真的生效（1024/2048/4096）
#   gemini-2.5-flash-image                       接受參數但**靜默忽略**，永遠回 1024
#   gemini-3.1-flash-lite-image                  2K/4K 直接回 400
# 所以 MODELS 裡後兩個型號的 sizes 只列 1K——列出來卻做不到的選項比沒有更糟。
#
# aspectRatio 則是四個型號都支援全部 10 種（含最偏門的 21:9，四個型號都實測過；
# gemini-2.5-flash-image 的量化略有不同，21:9 給的是 1536x672 而非 1584x672）。
# 非法值（99:1、banana、8K）一律回 400，不會被靜默忽略——但錯誤訊息是通用的
# "Request contains an invalid argument."，看不出是哪個欄位有問題。
_GEMINI_IMAGE_SIZES = {"1K", "2K", "4K"}

async def _gemini_image_once(client: httpx.AsyncClient, model: str, parts: list,
                             gen_config: dict, api_key: str) -> tuple[Optional[dict], str, Optional[JSONResponse], dict]:
    """對原生端點打一次，回傳 (圖片, 這次拿到的純文字, 致命錯誤, 這次的 usage)。

    usage 要一併帶出來——Gemini 圖像模型是按 token 計費的，前端沒有這個數字就
    無法把花費算進「本次花費」。多張時每次呼叫的 usage 要累加。
    """
    resp = await client.post(
        f"{NENAI_BASE}/v1beta/models/{model}:generateContent",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"contents": [{"parts": parts}], "generationConfig": gen_config},
    )
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        return None, "", JSONResponse(status_code=resp.status_code, content={"error": msg}), {}
    rj = resp.json()
    usage = _gemini_usage(rj.get("usageMetadata") or {}) if rj.get("usageMetadata") else {}
    text = ""
    for cand in rj.get("candidates", []):
        for part in (cand.get("content") or {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                ext = mime.split("/")[-1] or "png"
                raw = base64.b64decode(inline["data"])
                saved = await _save_image_bytes(raw, ext)
                return {"url": None, "local_path": saved, "actual_prompt": None}, text, None, usage
            if part.get("text"):
                text += part["text"]
    return None, text, None, usage


async def _generate_gemini_image(model: str, prompt: str, n: int, api_key: str,
                                 image_files: Optional[list] = None,
                                 aspect_ratio: Optional[str] = None,
                                 image_size: Optional[str] = None) -> dict:
    """Gemini 圖像模型走 Gemini 原生的 /v1beta/models/{model}:generateContent。

    先前這裡是走 OpenAI 相容的 /v1/chat/completions + modalities，那條路徑上結構化的
    imageConfig 會被靜默忽略，只能用「在 prompt 文字裡拜託模型輸出某個比例」的權宜
    做法，而且**完全沒有辦法控制輸出解析度**（使用者反映「選不了生成結果大小」）。
    2026-08-10 實測確認原生端點兩個參數都真的生效：
      imageConfig.aspectRatio → 1:1 得到 1024x1024、9:16 得到 768x1376
      imageConfig.imageSize   → 1K/2K/4K 得到長邊 1024/2048/4096
    而且圖像編輯（帶參考圖）也吃 aspectRatio——舊路徑在有參考圖時是連比例都不處理的。

    兩個要注意的地方：
      - 原生端點不接受 candidateCount（送了直接 400），多張只能並發打 n 次。
      - imageSize 的支援度各型號不同，見 _GEMINI_IMAGE_SIZES 的註解。
    """
    parts: list = []
    for _fname, fbytes, ftype in (image_files or []):
        parts.append({"inlineData": {"mimeType": ftype or "image/png",
                                     "data": base64.b64encode(fbytes).decode()}})
    parts.append({"text": (f"Edit the image(s) as follows: {prompt}" if image_files
                           else f"Generate an image depicting: {prompt}")})

    gen_config: dict = {"responseModalities": ["IMAGE"]}
    image_config: dict = {}
    if aspect_ratio:
        image_config["aspectRatio"] = aspect_ratio
    # 只接受 1K/2K/4K：瀏覽器可能還快取著舊版前端、送來 "1024*1024" 這種其他家族的
    # 尺寸字串，直接轉發會讓上游回 400
    if image_size and image_size.upper() in _GEMINI_IMAGE_SIZES:
        image_config["imageSize"] = image_size.upper()
    if image_config:
        gen_config["imageConfig"] = image_config

    images: list = []
    last_text = ""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    async with httpx.AsyncClient(timeout=300.0) as client:
        for _attempt in range(_GEMINI_IMAGE_MAX_RETRIES + 1):
            missing = max(0, n - len(images))
            if not missing:
                break
            results = await asyncio.gather(*[
                _gemini_image_once(client, model, parts, gen_config, api_key)
                for _ in range(missing)
            ])
            for img, text, err, u in results:
                if err is not None:
                    return err
                # 每次呼叫都計費，即使那次沒出圖（重試那幾次照樣消耗 token）
                total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += u.get("completion_tokens", 0)
                if img:
                    images.append(img)
                elif text:
                    last_text = text
            if len(images) >= n:
                break
    if images:
        out = {"success": True, "images": images[:n], "model": model}
        if total_usage["prompt_tokens"] or total_usage["completion_tokens"]:
            out["usage"] = total_usage
        return out
    preview = last_text[:200] + ("…" if len(last_text) > 200 else "")
    return JSONResponse(status_code=500, content={
        "error": f"模型未回傳圖片，改用純文字回覆（重試 {_GEMINI_IMAGE_MAX_RETRIES} 次仍失敗）：{preview}"
    })

# 千問「生成與編輯融合模型」：最多 3 張參考圖、可輸出 1-6 張，且不支援 ref_strength 參數，
# 改以 prompt_extend 控制（3.0 系列的上游規格同樣是 n 1~6、I2I 最多 3 張，行為一致）
_QWEN_FUSION_EDIT_MODELS = {"qwen-image-2.0-pro", "qwen-image-2.0",
                            "qwen-image-3.0-pro", "qwen-image-3.0"}
# GPT Image 系列額外支援 OpenAI 標準的 quality/background/output_format 三個參數（已實測確認有效）
_GPT_IMAGE_MODELS = {"gpt-image-2", "gpt-image-1.5"}
# 支援組圖模式（enable_sequential）與更高解析度的萬相 2.7 系列
_WAN27_IMAGE_MODELS = {"wan2.7-image-pro", "wan2.7-image"}
# 圖像編輯的參考圖張數上限，以 MODELS 的 max_ref 為單一來源（前端讀同一份資料）。
# 2026-08-10 對正式網關逐一實測——做法是「前 N 張純紅 + 最後一張純藍 + 要求模型輸出
# 所有參考圖的混色」，量輸出的平均 RGB：出現藍色成分就代表最後那張真的被讀進去了。
# 只驗「送得出去」是不夠的，上游可能接受請求卻靜默忽略多出來的圖。
#   wan2.7-image / -pro       9 張都生效（實測第 9 張有效）
#   wan2.6-image              上限 4（送 5 張回 "the last message must contain 1 to 4 images"）
#   qwen-image-2.0 / -pro     上限 3（送 4 張回 "supports 0~3 image content items"）
#   MAI-Image-2.5 / -Flash / -Pro   只接受剛好 1 張（多送回 "Exactly one image file must be attached"）
#   gpt-image-2 / -1.5、seedream、Gemini   9 張都生效
_EDIT_MAX_REF = {m["id"]: m["max_ref"] for m in MODELS["image"]
                 if m.get("type") == "i2i" and m.get("max_ref")}
# 圖片端點統一的上游逾時。有些圖片模型（例如萬相 2.7 系列）在閘道端走的是非同步
# 端點、由閘道代為輪詢到 SUCCEEDED 才回應，客戶端看到的是一次很慢的同步請求，
# 原本的 120 秒不夠用
_IMAGE_TIMEOUT = 300.0

# ─── API: Image Generate (T2I) ────────────────────────────────────
class ImageGenerateRequest(BaseModel):
    model: str = "z-image-turbo"
    prompt: str = ""
    negative_prompt: str = ""
    size: str = "1024*1024"
    n: int = 1
    prompt_extend: bool = False
    watermark: bool = False
    seed: Optional[int] = None
    aspect_ratio: Optional[str] = None       # 僅 Gemini 圖片模型使用
    quality: Optional[str] = None            # 僅 GPT Image 使用：auto/low/medium/high
    background: Optional[str] = None         # 僅 GPT Image 使用：auto/opaque/transparent
    output_format: Optional[str] = None      # 僅 GPT Image 使用：png/jpeg/webp
    enable_sequential: bool = False          # 僅萬相 2.7 使用：組圖模式
    # 自訂尺寸的第二條路：直接送頂層 width/height（目前只有 MAI 這條路通）。
    # 兩者都給時**上游以 width/height 為準**——實測 size=1024x1024 配上不合法的
    # width/height=2000x2000 會回 400 而不是照 size 產圖。所以這裡二選一送出，
    # 不同時帶，避免使用者選了 size 卻被殘留的 width/height 蓋掉。
    width: Optional[int] = None
    height: Optional[int] = None

@app.post("/api/image/generate")
async def image_generate(data: ImageGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if data.model in _GEMINI_CHAT_IMAGE_MODELS:
        try:
            return await _generate_gemini_image(data.model, data.prompt, data.n, api_key,
                                                aspect_ratio=data.aspect_ratio,
                                                image_size=data.size)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    payload: dict = {"model": data.model, "prompt": data.prompt, "n": data.n, "size": data.size}
    # 自訂尺寸選了「width / height」那條路時，改送這兩個欄位、並把 size 拿掉。
    # 上游本來就以 width/height 為準（實測會蓋過 size），拿掉 size 只是讓送出的
    # 請求跟使用者選的機制一致，看日誌時不會誤以為兩個都在生效。
    if data.width and data.height:
        payload.pop("size", None)
        payload["width"] = data.width
        payload["height"] = data.height
    if data.negative_prompt:
        payload["negative_prompt"] = data.negative_prompt
    if data.prompt_extend:
        payload["prompt_extend"] = True
    if data.watermark:
        payload["watermark"] = True
    if data.seed is not None:
        payload["seed"] = data.seed
    if data.model in _GPT_IMAGE_MODELS:
        if data.quality:
            payload["quality"] = data.quality
        if data.background:
            payload["background"] = data.background
        if data.output_format:
            payload["output_format"] = data.output_format
    if data.model in _WAN27_IMAGE_MODELS and data.enable_sequential:
        payload["enable_sequential"] = True

    try:
        async with httpx.AsyncClient(timeout=_IMAGE_TIMEOUT) as client:
            resp = await client.post(
                f"{NENAI_V1}/images/generations",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            rj = resp.json()
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code,
                                    content={"error": rj.get("error", {}).get("message", resp.text)})
            images = await _extract_images_from_data(rj.get("data", []))
            # data[] 可能少於實際產出（見 _extract_images_from_metadata），從 metadata 補齊，
            # 否則使用者會被扣 n 張的錢卻只看到一張
            images += await _extract_images_from_metadata(rj, images)
            if not images:
                return JSONResponse(status_code=500, content={"error": f"No images in response: {rj}"})
            out = {"success": True, "images": images, "model": data.model}
            usage = _image_usage(rj)
            if usage: out["usage"] = usage
            return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GPT Image 系列的 /images/edits 不接受 ref_strength 參數，帶入會被上游拒絕（400 Unknown parameter）
_NO_REF_STRENGTH_EDIT_MODELS = _QWEN_FUSION_EDIT_MODELS | {"gpt-image-2", "gpt-image-1.5",
                                                     "MAI-Image-2.5", "MAI-Image-2.5-Flash", "MAI-Image-2.5-Pro"}

# ─── API: Image Edit (I2I) ────────────────────────────────────────
@app.post("/api/image/edit")
async def image_edit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model       = form.get("model", "wan2.6-image")
    is_fusion_edit = model in _QWEN_FUSION_EDIT_MODELS
    prompt      = form.get("prompt", "")
    neg_prompt  = form.get("negative_prompt", "")
    size        = form.get("size", "1024*1024")
    watermark   = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    prompt_extend = str(form.get("prompt_extend", "true")).lower() in ("true", "1", "yes")
    seed_str    = str(form.get("seed", ""))
    seed        = int(seed_str) if seed_str.strip() else None
    try:
        ref_strength = float(form.get("ref_strength", "0.5"))
    except ValueError:
        ref_strength = 0.5
    try:
        n = int(form.get("n", "1"))
    except ValueError:
        n = 1
    n = max(1, min(6, n)) if is_fusion_edit else 1
    quality       = form.get("quality", "")
    background    = form.get("background", "")
    output_format = form.get("output_format", "")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # 參考圖張數上限一律以 MODELS 的 max_ref 為單一來源（跟前端讀同一份資料），
    # 未標的模型沿用 9 張。各家族的實測值與依據見 MODELS 裡的註解。
    max_refs = _EDIT_MAX_REF.get(model, 9)
    image_files: list[tuple[str, bytes, str]] = []
    for i in range(1, max_refs + 1):
        f = form.get(f"image_{i}")
        if not f or not hasattr(f, "filename") or not f.filename:
            break
        raw = await f.read()
        img = PILImage.open(BytesIO(raw))
        w, h = img.size
        if w < 240 or h < 240:
            scale = max(240 / w, 240 / h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_files.append((f.filename or f"image_{i}.png", buf.getvalue(), "image/png"))

    if not image_files:
        return JSONResponse(status_code=400, content={"error": "至少需要一張參考圖片"})

    if model in _GEMINI_CHAT_IMAGE_MODELS:
        try:
            return await _generate_gemini_image(model, prompt, n, api_key, image_files,
                                                aspect_ratio=form.get("aspect_ratio") or None,
                                                image_size=size)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        form_data = {"model": model, "prompt": prompt, "size": size, "n": str(n)}
        if is_fusion_edit:
            form_data["prompt_extend"] = "true" if prompt_extend else "false"
        if model not in _NO_REF_STRENGTH_EDIT_MODELS:
            form_data["ref_strength"] = str(ref_strength)
        if neg_prompt:
            form_data["negative_prompt"] = neg_prompt
        if watermark:
            form_data["watermark"] = "true"
        if seed is not None:
            form_data["seed"] = str(seed)
        if model in _GPT_IMAGE_MODELS:
            if quality:
                form_data["quality"] = quality
            if background:
                form_data["background"] = background
            if output_format:
                form_data["output_format"] = output_format

        # 多張參考圖必須用「重複的 image 欄位」，不能用 image_2 / image_3 這種編號欄位名。
        # 2026-08-10 對正式網關實測（兩張純色圖 + 要求模型混色，量輸出的平均 RGB）：
        #   image + image_2   → RGB(202,61,69) 紅色，**只有第一張生效**，第二張被靜默丟棄
        #   image + image     → RGB(159,9,247) 紫色，兩張都吃 ✅
        #   image[] + image[] → RGB(165,3,247) 紫色，兩張都吃 ✅
        # 這是使用者實際回報的問題（「wan2.7 上傳兩張照片，似乎只會吃第一張」）。
        # 重複 image 欄位在 wan2.7-image / wan2.6-image / qwen-image-2.0 / gpt-image-2 /
        # dola-seedream-5.0-pro 上都實測正常；MAI 系列則是只接受一張（見 _MAI_EDIT_MAX_REF）。
        files = [("image", (fname, fbytes, ftype)) for fname, fbytes, ftype in image_files]

        async with httpx.AsyncClient(timeout=_IMAGE_TIMEOUT) as client:
            resp = await client.post(
                f"{NENAI_V1}/images/edits",
                headers={"Authorization": f"Bearer {api_key}"},
                data=form_data,
                files=files,
            )
            rj = resp.json()
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code,
                                    content={"error": rj.get("error", {}).get("message", resp.text)})
            images = await _extract_images_from_data(rj.get("data", []))
            images += await _extract_images_from_metadata(rj, images)
            if not images:
                return JSONResponse(status_code=500, content={"error": f"No images in response: {rj}"})
            out = {"success": True, "images": images, "model": model}
            usage = _image_usage(rj)
            if usage: out["usage"] = usage
            return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Resolution helper ────────────────────────────────────────────
# Gemini Omni 不走 /v1/videos 的非同步任務模式，而是同步呼叫 /v1beta/interactions 直接拿到完成的影片
_INTERACTIONS_VIDEO_MODELS = {"gemini-omni-flash-preview"}
# Veo 預設的 personGeneration 安全設定較嚴格，帶真人圖片容易被擋，明確放寬為 allow_adult
_VEO_MODELS = {"veo-3.1-generate-001", "veo-3.1-fast-generate-001", "veo-3.1-lite-generate-001"}
_OMNI_TASK_CACHE: Dict[str, dict] = {}

# 下面兩組都是從 MODELS 的旗標推導出來的，前端已經據此收掉對應的 UI，這裡再擋一次，
# 避免直接打 API 的呼叫端送出上游會靜默丟棄的內容（尾幀被丟掉只會拿到一支「看起來
# 就是沒有照做」的影片，不會有任何錯誤訊息，很難查）
_FIRST_FRAME_ONLY_I2V_MODELS = {
    m["id"] for m in MODELS["video"]
    if m.get("type") == "i2v" and m.get("i2v_modes") == ["first_frame"]
}
_REF_IMAGES_ONLY_MODELS = {m["id"] for m in MODELS["video"] if m.get("ref_images_only")}

# 參考圖張數上限一律以 MODELS 的 max_ref 為單一來源。先前前端寫死 3、後端另外寫
# 「happyhorse 給 5、其餘 3」，兩邊各自演化就會不一致——多出來的檔案會被靜默丟棄
# （使用者完全看不出來），少算的則是白白用不到模型支援的欄位。
_VIDEO_MAX_REF = {(m["id"], m.get("type")): m["max_ref"]
                  for m in MODELS["video"] if m.get("max_ref")}


def _video_max_ref(model: str, task_type: str, default: int) -> int:
    return _VIDEO_MAX_REF.get((model, task_type), default)

# 萬相 3.0 這種 all-in-one 模型的模型名沒有 i2v/r2v/videoedit 後綴，上游無法從模型名
# 判斷每個媒體的用途，改以「MIME／副檔名 ＋ 位置」推斷：影片→video、音訊→
# driving_audio、第一張圖→first_frame、其餘圖→reference_image。我們送的 data URI
# 是判得出來的（上游會從 data:<mime>; 取 MIME），但仍然改走上游提供的覆寫管道，
# 直接把我們自己已標好 type 的陣列放進 metadata.input.media，原因有二：
#   1. 位置推斷表達不了我們實際有的語意。上游對影片一律推成 `video`（video-edit 的
#      來源影片），永遠不會產出 `first_clip`（影片續寫的起始片段）——這兩者在 wan2.7
#      是不同語意，而我們的 i2v「影片延伸」模式送的正是 first_clip。
#   2. 實測階段的除錯價值：完全不依賴上游的推斷，萬一失敗就能確定問題出在 type 詞彙
#      本身，而不是推斷邏輯，範圍收斂得比較快。
# 上游確認這是預期用法。
_WAN30_ALLINONE_MODELS = {"wan3.0-video"}
# 同一批模型的 ratio 預設是 adaptive（跟隨輸入自適應），而不是其他家族慣用的 16:9；
# ratio 與 resolution 是兩個互相獨立的參數
_ADAPTIVE_RATIO_MODELS = _WAN30_ALLINONE_MODELS


def _default_ratio(model: str) -> str:
    return "adaptive" if model in _ADAPTIVE_RATIO_MODELS else "16:9"


def _apply_explicit_media(model: str, meta: dict, media_arr: list) -> None:
    """all-in-one 模型：繞過上游依副檔名猜測媒體用途的邏輯，直接指定 type。"""
    if model in _WAN30_ALLINONE_MODELS and media_arr:
        meta.setdefault("input", {})["media"] = media_arr


def _apply_res_and_duration(payload: dict, meta: dict, resolution: str,
                            duration: Optional[int] = None, ratio: str = "") -> None:
    """把解析度／時長／畫面比例同時以三家上游各自看得懂的形式塞進 payload 與 metadata。

    影片的四個端點是所有廠商共用的，但每一家 task adaptor 取值的欄位都不一樣，
    任何一家漏送都不會報錯、只會靜默用它自己的預設值（使用者選了 1080P 卻拿到
    720P 的影片，而且照 720P 計費）：

    - 阿里（萬相／HappyHorse）：讀頂層 `size`。"720P"/"1080P" 這種字串在它的三條
      分支都解析得出來（t2v 會再轉成 "1280*720"），所以頂層 size 必須保留。
    - Veo（gemini／vertex）：頂層 `size` 是用小寫 "x" 去切 WIDTHxHEIGHT 的，
      "1080P" 切不開會靜默 fallback 成 720p——但 `metadata.resolution` 優先權最高，
      用它蓋過去就正確；畫面比例的欄位叫 `aspectRatio`，不吃 `ratio`。
    - Seedance／Dreamina（doubao）：完全不讀頂層 `size`，也不讀頂層 `duration`，
      只吃 `metadata.resolution` 與頂層 `seconds`（字串）。畫面比例吃 `ratio`。

    三家的 metadata 都是整包 unmarshal 進各自的 payload struct、未知 key 直接忽略，
    所以重複多送幾個 key 是安全的。
    """
    payload["size"] = resolution
    meta["resolution"] = resolution.lower()
    if duration is not None:
        payload["duration"] = duration        # 阿里 / Veo（Veo 只接受 4、6、8）
        payload["seconds"] = str(duration)    # doubao 唯一吃得到的時長來源
    if ratio:
        meta["ratio"] = ratio                 # doubao
        meta["aspectRatio"] = ratio           # Veo


def _apply_audio_flag(payload: dict, meta: dict, audio: bool) -> None:
    """把「要不要配音」用三家各自的欄位名送出去。

    注意阿里這邊的實際情況：task adaptor 根本不讀統一請求的頂層 `audio`，整份
    adaptor 只有 wan2.6-i2v-flash 會去讀 `metadata.audio`（bool）——其餘萬相型號
    有沒有聲音完全由上游自己決定。MODELS 裡已經據此把那些型號的 audio 旗標關掉、
    UI 不再顯示無效的開關，這裡仍把欄位帶齊，讓有支援的型號（wan2.6-i2v-flash、
    Veo、doubao）拿得到。
    """
    payload["audio"] = audio        # 統一請求的頂層欄位（阿里不讀，其他家備援）
    meta["audio"] = audio           # wan2.6-i2v-flash 唯一吃得到的來源
    meta["generateAudio"] = audio   # Veo（非 bool 或缺省一律當無音訊，也影響計費）
    meta["generate_audio"] = audio  # doubao（影響計費 audio_presence）


def _public_base_url(request: Optional[Request]) -> Optional[str]:
    """本站對外可存取的網址，用來把上傳的媒體以公開連結交給模型抓取。

    優先讀環境變數 `PUBLIC_BASE_URL`（部署時明確指定最可靠）；沒設定就從請求標頭推導
    ——正式環境前面有 Load Balancer，真實的協定與主機名在 X-Forwarded-Proto / Host。
    本機開發（localhost）推導出來的網址外部抓不到，所以直接回 None、讓呼叫端走錯誤路徑。
    """
    env = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if env:
        return env
    if request is None:
        return None
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    host = host.split(",")[0].strip()
    if not host or host.startswith("127.0.0.1") or host.startswith("localhost"):
        return None
    # 沒有 X-Forwarded-Proto 時預設 https，不要用 request.url.scheme——那在 LB 後面
    # 是內部的 http，給出 http:// 的網址有機會被上游拒絕或被轉址擋掉
    proto = (request.headers.get("x-forwarded-proto") or "https").split(",")[0].strip()
    return f"{proto}://{host}"


async def _upload_video_for_url(raw: bytes, filename: str = "clip.mp4",
                                request: Optional[Request] = None) -> tuple[Optional[str], Optional[str]]:
    """把影片放到雲端物件儲存，回傳 (簽名網址, 錯誤訊息)。

    **上游不接受 base64 data URI 的影片**——送 `data:video/mp4;base64,...` 會在任務
    輪詢階段失敗，錯誤是 `InvalidVideo.FileFormat: Invalid video type. Only
    mp4/mov/avi is supported.`（提交時回 200，所以只看提交結果會以為成功）。
    2026-08-11 對正式環境用 wan2.2-animate-move 實測確認。

    這跟音訊是同一類限制（見 `_upload_audio_for_url`）：圖片可以用 data URI，
    影片與音訊都必須是一個真的能被下載的 URL。

    取得 URL 有兩條路，依序嘗試：
      1. 雲端物件儲存（OSS / S3 / GCS）——最可靠，簽名網址有 7 天效期
      2. **退回本站自己的公開路徑**——`/outputs` 是不需驗證就能存取的靜態掛載
         （實測正式站不帶 Authorization 也回 200），所以把檔案寫進 outputs/videos
         再給出 `https://<本站網域>/outputs/videos/<name>` 就能讓模型抓到。
         ⚠️ 這條路有兩個先天限制：Cloud Run 每個實例的檔案系統獨立（`maxScale` > 1
         時模型可能被路由到沒有這個檔案的實例而抓不到），且容器重啟後檔案消失。
         模型通常在幾秒內就抓走，實務上多半可行，但要根治仍是設定雲端儲存。
    兩條都不通（例如本機開發，推導出來的 localhost 網址外部抓不到）才回報錯誤。
    """
    if not raw:
        return None, None
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix not in (".mp4", ".mov", ".avi"):
        suffix = ".mp4"
    name = f"vid_in_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{suffix}"
    url = _cloud_put(raw, f"uploads/{name}")
    if url:
        return url, None
    base = _public_base_url(request)
    if base:
        try:
            OUTPUT_VID_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_VID_DIR / name).write_bytes(raw)
            return f"{base}/outputs/videos/{name}", None
        except Exception as e:
            print(f"[upload] 寫入本機 outputs 失敗：{e}")
    return None, ("影片需要一個可公開下載的網址才能交給模型處理，但這個環境既沒有設定"
                  "雲端物件儲存，也無法推導出對外網址。請設定雲端儲存（OSS / S3 / GCS）"
                  "或 PUBLIC_BASE_URL 後再試。")


async def _upload_audio_for_url(file_obj, request: Optional[Request] = None) -> tuple[Optional[str], Optional[str]]:
    """把使用者上傳的音檔放到雲端物件儲存，回傳 (簽名網址, 錯誤訊息)。

    上游只接受 `audio_url`（一個真的能被下載的 URL）——先前這裡是把音檔轉成
    `data:audio/...;base64,...` 塞進 metadata，上游根本不會去解析，等於使用者
    上傳的配樂從來沒有生效過。本機 `outputs/` 路徑上游同樣抓不到，所以沒有任何
    雲端儲存後端可用時直接回報錯誤，而不是送出一個註定無聲的請求。
    """
    if not file_obj or not hasattr(file_obj, "filename") or not file_obj.filename:
        return None, None
    raw = await file_obj.read()
    if not raw:
        return None, None
    suffix = Path(file_obj.filename).suffix.lower() or ".mp3"
    name = f"aud_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{suffix}"
    url = _cloud_put(raw, f"audio/{name}")
    if url:
        return url, None
    # 與影片同一套 fallback：退回本站的公開靜態路徑（限制見 _upload_video_for_url）
    base = _public_base_url(request)
    if base:
        try:
            OUTPUT_VID_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_VID_DIR / name).write_bytes(raw)
            return f"{base}/outputs/videos/{name}", None
        except Exception as e:
            print(f"[upload] 寫入本機 outputs 失敗：{e}")
    return None, ("音訊需要一個可公開下載的網址才能交給模型處理，但這個環境既沒有設定"
                  "雲端物件儲存，也無法推導出對外網址。請設定雲端儲存（OSS / S3 / GCS）"
                  "或 PUBLIC_BASE_URL，或改用模型自動配音。")

async def _save_video_bytes(data: bytes) -> Optional[str]:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"vid_{ts}_{uuid.uuid4().hex[:6]}.mp4"
        cloud_url = _cloud_put(data, f"videos/{name}")
        if cloud_url:
            return cloud_url
        fp = OUTPUT_VID_DIR / name
        fp.write_bytes(data)
        return f"/outputs/videos/{fp.name}"
    except Exception as e:
        print(f"Video save error: {e}")
        return None

async def _generate_omni_video(model: str, prompt: str, api_key: str,
                                image_files: Optional[list] = None) -> dict:
    content: list = [{"type": "text", "text": prompt}]
    for fbytes, ftype in (image_files or []):
        b64 = base64.b64encode(fbytes).decode()
        content.append({"type": "image", "data": b64, "mime_type": ftype})
    payload = {"model": model, "input": [{"type": "user_input", "content": content}]}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{NENAI_BASE}/v1beta/interactions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8"},
            json=payload,
        )
        rj = resp.json()
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code,
                                content={"error": rj.get("error", {}).get("message", resp.text)})
        video_bytes = None
        for step in rj.get("steps", []):
            if step.get("type") != "model_output":
                continue
            for c in step.get("content", []):
                if c.get("type") == "video" and c.get("data"):
                    video_bytes = base64.b64decode(c["data"])
                    break
            if video_bytes:
                break
        if not video_bytes:
            return JSONResponse(status_code=500, content={"error": f"No video in response: {rj}"})
        local_path = await _save_video_bytes(video_bytes)
        task_id = f"omni_{uuid.uuid4().hex}"
        _OMNI_TASK_CACHE[task_id] = {"status": "SUCCEEDED", "local_path": local_path, "video_url": local_path}
        return {"success": True, "task_id": task_id, "status": "queued", "model": model}

# ─── API: Video T2V ───────────────────────────────────────────────
@app.post("/api/video/t2v")
async def video_t2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model           = form.get("model", "wan2.6-t2v")
    prompt          = form.get("prompt", "")
    negative_prompt = form.get("negative_prompt", "")
    resolution      = form.get("resolution", "720P")
    ratio           = form.get("ratio") or _default_ratio(model)
    duration        = int(form.get("duration", 5))
    audio           = str(form.get("audio", "false")).lower() in ("true", "1", "yes")
    prompt_extend   = str(form.get("prompt_extend", "false")).lower() in ("true", "1", "yes")
    watermark       = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str        = str(form.get("seed", ""))
    seed            = int(seed_str) if seed_str.strip() else None
    audio_file      = form.get("audio_file")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if model in _INTERACTIONS_VIDEO_MODELS:
        try:
            return await _generate_omni_video(model, prompt, api_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    payload: dict = {"model": model, "prompt": prompt}
    meta: dict = {}
    _apply_res_and_duration(payload, meta, resolution, duration, ratio)
    if negative_prompt: meta["negative_prompt"] = negative_prompt
    if prompt_extend:   meta["prompt_extend"] = True
    if watermark:       meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"

    # 上游未收到欄位時會自行判斷是否配音，不會視為「不要配音」——
    # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
    _apply_audio_flag(payload, meta, audio)
    audio_url, audio_err = await _upload_audio_for_url(audio_file, request)
    if audio_err:
        return JSONResponse(status_code=400, content={"error": audio_err})
    if audio_url:
        payload["audio_url"] = audio_url

    if meta: payload["metadata"] = meta

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video I2V ───────────────────────────────────────────────
@app.post("/api/video/i2v")
async def video_i2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model         = form.get("model", "wan2.7-i2v")
    prompt        = form.get("prompt", "")
    neg_prompt    = form.get("negative_prompt", "")
    resolution    = form.get("resolution", "720P")
    ratio         = form.get("ratio") or _default_ratio(model)
    duration      = int(form.get("duration", 5))
    i2v_mode      = form.get("i2v_mode", "first_frame")
    prompt_extend = str(form.get("prompt_extend", "false")).lower() in ("true", "1", "yes")
    watermark     = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str      = str(form.get("seed", ""))
    seed          = int(seed_str) if seed_str.strip() else None
    audio_bgm      = str(form.get("audio", "false")).lower() in ("true", "1", "yes")
    audio_bgm_file = form.get("audio_file")

    async def _read_image_bytes(file_obj) -> Optional[bytes]:
        if not file_obj or not hasattr(file_obj, "filename") or not file_obj.filename:
            return None
        raw = await file_obj.read()
        img = PILImage.open(BytesIO(raw))
        w, h = img.size
        if w < 240 or h < 240:
            scale = max(240 / w, 240 / h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    first_frame_file = form.get("first_frame") or form.get("image")
    last_frame_file  = form.get("last_frame")
    audio_file       = form.get("driving_audio")
    clip_file        = form.get("first_clip")

    if model in _FIRST_FRAME_ONLY_I2V_MODELS and i2v_mode != "first_frame":
        return JSONResponse(status_code=400, content={
            "error": f"{model} 只支援首幀生成，尾幀／驅動音訊／影片延伸不適用於這個模型，請改用「首幀生成」模式。"})

    if model in _INTERACTIONS_VIDEO_MODELS:
        first_bytes = await _read_image_bytes(first_frame_file)
        if not first_bytes:
            return JSONResponse(status_code=400, content={"error": "I2V 需要上傳首幀圖片"})
        try:
            return await _generate_omni_video(model, prompt, api_key, [(first_bytes, "image/png")])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    actual_duration = duration
    payload: dict = {"model": model, "prompt": prompt}
    meta: dict = {"i2v_mode": i2v_mode}
    _apply_res_and_duration(payload, meta, resolution, actual_duration, ratio)
    if neg_prompt:    meta["negative_prompt"] = neg_prompt
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"
    # 上游未收到欄位時會自行判斷是否配音，不會視為「不要配音」——
    # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
    _apply_audio_flag(payload, meta, audio_bgm)
    bgm_url, bgm_err = await _upload_audio_for_url(audio_bgm_file, request)
    if bgm_err:
        return JSONResponse(status_code=400, content={"error": bgm_err})
    if bgm_url:
        payload["audio_url"] = bgm_url

    media_arr: list = []

    if i2v_mode in ("first_clip", "first_clip_last_frame"):
        if not clip_file or not hasattr(clip_file, "filename") or not clip_file.filename:
            return JSONResponse(status_code=400, content={"error": "first_clip 模式需要上傳影片片段"})
        clip_bytes = await clip_file.read()
        tmp_path = UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}.mp4"
        tmp_path.write_bytes(clip_bytes)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(tmp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            dur = float(result.stdout.strip())
            if dur > 9.9:
                return JSONResponse(status_code=400, content={"error": f"上傳影片長度為 {dur:.2f} 秒，請修剪至 9.9 秒以內再上傳。"})
        except Exception:
            pass
        finally:
            tmp_path.unlink(missing_ok=True)
        # 影片延伸的起始片段同樣不能用 data URI
        clip_b64, clip_err = await _upload_video_for_url(clip_bytes, clip_file.filename or "clip.mp4", request)
        if clip_err:
            return JSONResponse(status_code=400, content={"error": clip_err})
        media_arr.append({"url": clip_b64, "type": "first_clip"})
        actual_duration = max(duration, 15)
        _apply_res_and_duration(payload, meta, resolution, actual_duration, ratio)
        if last_frame_file and hasattr(last_frame_file, "filename") and last_frame_file.filename:
            lb = await _read_image_bytes(last_frame_file)
            if lb: media_arr.append({"url": f"data:image/png;base64,{base64.b64encode(lb).decode()}", "type": "last_frame"})
    else:
        first_bytes = await _read_image_bytes(first_frame_file)
        if not first_bytes:
            return JSONResponse(status_code=400, content={"error": "I2V 需要上傳首幀圖片"})
        media_arr.append({"url": f"data:image/png;base64,{base64.b64encode(first_bytes).decode()}", "type": "first_frame"})
        if last_frame_file and hasattr(last_frame_file, "filename") and last_frame_file.filename:
            lb = await _read_image_bytes(last_frame_file)
            if lb: media_arr.append({"url": f"data:image/png;base64,{base64.b64encode(lb).decode()}", "type": "last_frame"})
        # 驅動音訊同樣只能以 URL 形式傳給上游（base64 data URI 不會被解析）
        drive_url, drive_err = await _upload_audio_for_url(audio_file, request)
        if drive_err:
            return JSONResponse(status_code=400, content={"error": drive_err})
        if drive_url:
            # 刻意不放進 media_arr——media_arr 會整個變成 payload["images"]，
            # 混進音訊會讓上游把它當成參考圖
            payload["audio_url"] = drive_url

    # `media` array + `image` (first item URL) for maximum compatibility
    payload["media"] = media_arr
    if media_arr:
        payload["image"] = media_arr[0]["url"]
        # 平台 TaskSubmitReq 只認 images（陣列），media/image 會被忽略
        payload["images"] = [m["url"] for m in media_arr]
    _apply_explicit_media(model, meta, media_arr)
    payload["metadata"] = meta

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video Edit (wan2.7-videoedit) ──────────────────────────
@app.post("/api/video/vedit")
async def video_vedit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model         = form.get("model", "wan2.7-videoedit")
    prompt        = form.get("prompt", "")
    neg_prompt    = form.get("negative_prompt", "")
    resolution    = form.get("resolution", "1080P")
    ratio         = form.get("ratio", "")
    duration_str  = str(form.get("duration", "0"))
    duration      = int(duration_str) if duration_str.strip() else 0
    audio_setting = form.get("audio_setting", "auto")
    prompt_extend = str(form.get("prompt_extend", "true")).lower() in ("true", "1", "yes")
    watermark     = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str      = str(form.get("seed", ""))
    seed          = int(seed_str) if seed_str.strip() else None

    video_file = form.get("video")
    if not video_file or not hasattr(video_file, "filename") or not video_file.filename:
        return JSONResponse(status_code=400, content={"error": "影片檔案為必填"})

    video_bytes = await video_file.read()
    # 同上：來源影片必須是上游抓得到的網址，data URI 會在輪詢階段被拒
    video_b64, video_err = await _upload_video_for_url(video_bytes, video_file.filename or "source.mp4", request)
    if video_err:
        return JSONResponse(status_code=400, content={"error": video_err})

    media_arr: list = [{"url": video_b64, "type": "video"}]
    max_refs = _video_max_ref(model, "vedit", 3)
    for i in range(1, max_refs + 1):
        ref = form.get(f"reference_image_{i}")
        if ref and hasattr(ref, "filename") and ref.filename:
            rb = await ref.read()
            media_arr.append({"url": f"data:image/png;base64,{base64.b64encode(rb).decode()}", "type": "reference_image"})

    meta: dict = {"audio_setting": audio_setting}
    if neg_prompt:    meta["negative_prompt"] = neg_prompt
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed

    _apply_explicit_media(model, meta, media_arr)
    payload: dict = {
        "model": model, "prompt": prompt,
        "media": media_arr, "image": video_b64,
        "images": [m["url"] for m in media_arr],
        "metadata": meta,
    }
    # 視頻編輯不指定時長（保留來源影片長度），duration 為 0 時就不送
    _apply_res_and_duration(payload, meta, resolution,
                            duration if duration else None, ratio)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video R2V ───────────────────────────────────────────────
@app.post("/api/video/r2v")
async def video_r2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model         = form.get("model", "wan2.6-r2v")
    prompt        = form.get("prompt", "")
    resolution    = form.get("resolution", "720P")
    ratio         = form.get("ratio") or _default_ratio(model)
    duration      = int(form.get("duration", 5))
    prompt_extend = str(form.get("prompt_extend", "false")).lower() in ("true", "1", "yes")
    watermark     = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str      = str(form.get("seed", ""))
    seed          = int(seed_str) if seed_str.strip() else None
    audio_bgm      = str(form.get("audio", "false")).lower() in ("true", "1", "yes")
    audio_bgm_file = form.get("audio_file")

    ref_files = form.getlist("reference_files")
    if not ref_files or not hasattr(ref_files[0], "filename"):
        return JSONResponse(status_code=400, content={"error": "At least one reference file is required"})

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    # 萬相／HappyHorse 的 r2v 上游會把收到的每一個檔案都當成參考「圖片」
    # （wan2.6 走 reference_urls、wan2.7/happyhorse 走 media 的 reference_image），
    # 混入影片檔會被上游拒絕，在這裡先擋掉並給出明確訊息
    refs_images_only = model in _REF_IMAGES_ONLY_MODELS
    media_arr: list = []
    image_files: list = []
    for f in ref_files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if refs_images_only and ext in VIDEO_EXTS:
            return JSONResponse(status_code=400, content={
                "error": f"{model} 的參考生影片只接受圖片，不能帶影片檔（{f.filename}）。"
                         "若要以影片驅動，請改用「萬相動作動畫」或「視頻編輯」。"})
        fb = await f.read()
        mime = "video/mp4" if ext in VIDEO_EXTS else "image/png"
        media_type = "reference_video" if ext in VIDEO_EXTS else "reference_image"
        if media_type == "reference_video":
            # 影片不能用 data URI（見 _upload_video_for_url）
            ref_url, ref_err = await _upload_video_for_url(fb, f.filename, request)
            if ref_err:
                return JSONResponse(status_code=400, content={"error": ref_err})
            media_arr.append({"url": ref_url, "type": media_type})
        else:
            media_arr.append({"url": f"data:{mime};base64,{base64.b64encode(fb).decode()}", "type": media_type})
            image_files.append((fb, mime))

    if not media_arr:
        return JSONResponse(status_code=400, content={"error": "At least one reference file is required"})

    if model in _INTERACTIONS_VIDEO_MODELS:
        if not image_files:
            return JSONResponse(status_code=400, content={"error": "R2V 需要至少一張參考圖片"})
        try:
            return await _generate_omni_video(model, prompt, api_key, image_files[:3])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    meta: dict = {}
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"
    payload: dict = {"model": model, "prompt": prompt,
                     "media": media_arr,
                     "image": media_arr[0]["url"],
                     "images": [m["url"] for m in media_arr]}
    _apply_explicit_media(model, meta, media_arr)
    _apply_res_and_duration(payload, meta, resolution, duration, ratio)
    # 上游未收到欄位時會自行判斷是否配音，不會視為「不要配音」——
    # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
    _apply_audio_flag(payload, meta, audio_bgm)
    bgm_url, bgm_err = await _upload_audio_for_url(audio_bgm_file, request)
    if bgm_err:
        return JSONResponse(status_code=400, content={"error": bgm_err})
    if bgm_url:
        payload["audio_url"] = bgm_url

    if meta: payload["metadata"] = meta

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video Animate（視頻換人 wan2.2-animate-mix / 圖生動作 wan2.2-animate-move）──
@app.post("/api/video/animate")
async def video_animate(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model       = form.get("model", "wan2.2-animate-mix")
    mode        = form.get("mode", "wan-std")
    watermark   = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    check_image = str(form.get("check_image", "true")).lower() in ("true", "1", "yes")

    image_file = form.get("image")
    video_file = form.get("video")
    if not image_file or not hasattr(image_file, "filename") or not image_file.filename:
        return JSONResponse(status_code=400, content={"error": "請上傳人物圖片"})
    if not video_file or not hasattr(video_file, "filename") or not video_file.filename:
        return JSONResponse(status_code=400, content={"error": "請上傳參考影片"})

    img_bytes = await image_file.read()
    img_mime = image_file.content_type or "image/png"
    vid_bytes = await video_file.read()

    img_url = f"data:{img_mime};base64,{base64.b64encode(img_bytes).decode()}"
    # 影片不能用 data URI（上游只收 mp4/mov/avi 的實體檔案網址），必須先上雲端
    vid_url, vid_err = await _upload_video_for_url(vid_bytes, video_file.filename or "clip.mp4", request)
    if vid_err:
        return JSONResponse(status_code=400, content={"error": vid_err})

    payload = {
        "model": model,
        "media": [
            {"url": img_url, "type": "image"},
            {"url": vid_url, "type": "video"},
        ],
        # 平台 TaskSubmitReq 只認 images（陣列），media 會被忽略——順序 [0]=人物圖 [1]=參考影片
        "images": [img_url, vid_url],
        "metadata": {"mode": mode, "check_image": check_image, "watermark": watermark},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video Status ────────────────────────────────────────────
@app.get("/api/video/status/{task_id}")
async def video_status(task_id: str, api_key: str = Depends(get_api_key)):
    if task_id in _OMNI_TASK_CACHE:
        cached = _OMNI_TASK_CACHE[task_id]
        return {"task_id": task_id, **cached}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{NENAI_V1}/videos/{task_id}",
                                    headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            rj = resp.json()

        # status 可能在頂層或 task_info 裡
        raw_status = (
            rj.get("status")
            or rj.get("task_info", {}).get("status")
            or "pending"
        )
        status = raw_status.upper()
        if status in ("COMPLETED", "SUCCESS", "SUCCEED", "DONE", "FINISHED"):
            status = "SUCCEEDED"
        elif status in ("RUNNING", "PROCESSING", "SUBMITTED", "QUEUED", "IN_PROGRESS"):
            status = "PENDING"
        result: dict = {"task_id": task_id, "status": status, "_raw_status": raw_status}

        if status == "SUCCEEDED":
            # video URL 可能散落在多個地方（wan2.2-animate 系列位於 output.results.video_url）
            video_url = (
                rj.get("url")
                or rj.get("video_url")
                or rj.get("task_info", {}).get("video_url")
                or rj.get("output", {}).get("url")
                or rj.get("output", {}).get("video_url")
                or rj.get("output", {}).get("results", {}).get("video_url")
            )
            if not video_url and isinstance(rj.get("data"), list) and rj["data"]:
                video_url = (rj["data"][0] or {}).get("url")
            if not video_url:
                md = rj.get("metadata") or {}
                video_url = md.get("video_url") or md.get("url")
            if not video_url and isinstance(rj.get("videos"), list) and rj["videos"]:
                video_url = rj["videos"][0]
            if video_url:
                local = await _async_download_video(video_url)
                result["local_path"] = local if local else video_url
                result["video_url"] = video_url
            else:
                # Fallback: /content 可能轉址到實際檔案，也可能直接回傳影片二進位內容（如 Veo）
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as c2:
                    cr = await c2.get(f"{NENAI_V1}/videos/{task_id}/content",
                                      headers={"Authorization": f"Bearer {api_key}"})
                    loc = cr.headers.get("location") or cr.headers.get("Location")
                    if loc:
                        local = await _async_download_video(loc)
                        result["local_path"] = local if local else loc
                        result["video_url"] = loc
                    elif cr.status_code == 200 and "video" in cr.headers.get("content-type", ""):
                        local = await _save_video_bytes(cr.content)
                        result["local_path"] = local
                        result["video_url"] = local
            actual_prompt = (
                rj.get("actual_prompt")
                or rj.get("task_info", {}).get("actual_prompt")
                or rj.get("output", {}).get("actual_prompt")
                or rj.get("output", {}).get("results", {}).get("actual_prompt")
            )
            if actual_prompt:
                result["actual_prompt"] = actual_prompt
        elif status == "FAILED":
            # 失敗原因各家放的位置不同：萬相／Veo 放 error.message，doubao（Seedance）
            # 放頂層的 fail_reason——只讀 error 的話 Seedance 失敗會一律顯示
            # "Unknown error"，使用者看不到真正的原因（例如版權過濾那類訊息：
            # "The request failed because the output audio may be related to
            # copyright restrictions."，那是模型自動配樂觸發的，跟畫面無關，
            # 沒有訊息幾乎不可能猜到）
            err = rj.get("error") or rj.get("task_info", {}).get("error") or {}
            result["error_message"] = (
                (err.get("message") if isinstance(err, dict) else str(err))
                or rj.get("fail_reason")
                or rj.get("task_info", {}).get("fail_reason")
                or "Unknown error"
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video/debug/{task_id}")
async def video_status_debug(task_id: str, api_key: str = Depends(get_api_key)):
    """回傳平台原始 JSON，用於診斷 status 欄位位置。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{NENAI_V1}/videos/{task_id}",
                                headers={"Authorization": f"Bearer {api_key}"})
        return {"http_status": resp.status_code, "raw": resp.json()}

# ─── API: Voice (ASR / TTS) ─────────────────────────────────────────
# NenAI 網關對音訊的支援走 OpenAI 相容的 /v1/audio/transcriptions（ASR）與
# /v1/audio/speech（TTS），跟其他模型家族一樣用同一把 API key 直接轉發。
@app.post("/api/voice/asr")
async def voice_asr(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = str(form.get("model", "qwen-audio-3.0-asr-flash"))
    audio_file = form.get("audio")
    if not audio_file or not hasattr(audio_file, "read"):
        raise HTTPException(status_code=400, detail="缺少音檔")
    filename = getattr(audio_file, "filename", None) or "audio.wav"
    content_type = getattr(audio_file, "content_type", None) or mimetypes.guess_type(filename)[0] or "audio/wav"
    audio_bytes = await audio_file.read()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{NENAI_V1}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": model},
                files={"file": (filename, audio_bytes, content_type)},
            )
            rj = resp.json()
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code,
                                    content={"error": rj.get("error", {}).get("message", resp.text)})
            return {"success": True, "text": rj.get("text", ""), "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/asr/stream")
async def voice_asr_stream(request: Request, api_key: str = Depends(get_api_key)):
    """串流語音辨識——上游以 SSE 逐步回傳中間辨識結果，這裡原封不動轉發給前端。"""
    form = await request.form()
    model = str(form.get("model", "qwen-audio-3.0-asr-flash-streaming"))
    audio_file = form.get("audio")
    if not audio_file or not hasattr(audio_file, "read"):
        raise HTTPException(status_code=400, detail="缺少音檔")
    filename = getattr(audio_file, "filename", None) or "audio.wav"
    content_type = getattr(audio_file, "content_type", None) or mimetypes.guess_type(filename)[0] or "audio/wav"
    audio_bytes = await audio_file.read()

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{NENAI_V1}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"model": model, "stream": "true"},
                    files={"file": (filename, audio_bytes, content_type)},
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield f"data: {json.dumps({'type': 'error', 'error': err.decode(errors='ignore')})}\n\n"
                        return
                    # 上游對「整檔上傳」的請求目前一律回傳單一 JSON（即使帶 stream=true 也一樣，
                    # 只有真正的即時分段音訊輸入才會回真 SSE）——兩種情況都要能處理，避免真的
                    # 收到 SSE 以外的格式時前端什麼都收不到、看起來像卡住。
                    if "text/event-stream" in (resp.headers.get("content-type") or ""):
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            payload = line[len("data:"):].strip()
                            if payload == "[DONE]":
                                break
                            yield f"data: {payload}\n\n"
                    else:
                        body = await resp.aread()
                        try:
                            rj = json.loads(body)
                        except Exception:
                            rj = {"error": body.decode(errors="ignore")}
                        yield f"data: {json.dumps(rj)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

class VoiceTtsRequest(BaseModel):
    model: str = "qwen-audio-3.0-tts-flash"
    text: str = ""
    voice: str = ""
    format: str = "mp3"
    instructions: str = ""                      # 語氣/情緒風格描述，CosyVoice v3 專屬
    sample_rate: Optional[int] = None
    volume: Optional[int] = None
    language_hints: List[str] = []

@app.post("/api/voice/tts")
async def voice_tts(data: VoiceTtsRequest, api_key: str = Depends(get_api_key)):
    if not data.text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if data.model.startswith("gemini"):
                # Gemini TTS 系列實測是走 OpenAI 相容的 /v1/audio/speech（不是 Google 原生
                # 那套 /v1/text:synthesize——那個路徑在這個網關上會直接回錯），只吃
                # model/input/voice 三個欄位，instructions 帶了會被上游拒絕（400），
                # response_format 也會被忽略，永遠固定回傳 audio/wav。
                payload: dict = {"model": data.model, "input": data.text}
                if data.voice:
                    payload["voice"] = data.voice
                resp = await client.post(
                    f"{NENAI_V1}/audio/speech",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                if resp.status_code != 200:
                    try:
                        rj = resp.json()
                        err = rj.get("error", {}).get("message", resp.text)
                    except Exception:
                        err = resp.text
                    return JSONResponse(status_code=resp.status_code, content={"error": err})
                audio_bytes = resp.content
                ext = "wav"
            else:
                # qwen-audio-3.0-tts 系列實際上不是走 OpenAI 相容的 /v1/audio/speech（那個
                # endpoint 收 voice 一律回錯），而是走 DashScope 風格的
                # /v1/services/audio/tts/SpeechSynthesizer，回傳一段 JSON（output.audio.url
                # 是簽名過的 OSS 下載網址，data 通常是空字串），且 voice 要用 CosyVoice v3
                # 的音色 id（例如 longanlingxin、loongjohn），不是 Qwen-TTS 的 Cherry/Ethan
                # 那套；voice 留空則使用上游預設音色。
                payload = {"model": data.model, "input": data.text, "response_format": data.format}
                if data.voice:
                    payload["voice"] = data.voice
                if data.instructions:
                    payload["instructions"] = data.instructions
                metadata: dict = {}
                if data.sample_rate is not None:
                    metadata["sample_rate"] = data.sample_rate
                if data.volume is not None:
                    metadata["volume"] = data.volume
                if data.language_hints:
                    # 上游文件明載這個欄位雖然是陣列，目前版本卻只處理第一個元素，帶多個值沒意義
                    metadata["language_hints"] = data.language_hints[:1]
                if metadata:
                    payload["metadata"] = metadata

                resp = await client.post(
                    f"{NENAI_V1}/services/audio/tts/SpeechSynthesizer",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                rj = resp.json()
                if resp.status_code != 200 or "error" in rj:
                    err = rj.get("error", {}).get("message", resp.text) if isinstance(rj.get("error"), dict) else rj.get("error", resp.text)
                    return JSONResponse(status_code=resp.status_code if resp.status_code != 200 else 500,
                                        content={"error": err})
                audio_info = rj.get("output", {}).get("audio", {})
                audio_url_src = audio_info.get("url")
                b64_data = audio_info.get("data")
                if audio_url_src:
                    dl = await client.get(audio_url_src, timeout=60.0)
                    audio_bytes = dl.content
                elif b64_data:
                    audio_bytes = base64.b64decode(b64_data)
                else:
                    return JSONResponse(status_code=500, content={"error": f"模型未回傳音訊：{rj}"})
                ext = data.format if data.format in ("mp3", "wav", "opus", "flac") else "mp3"

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"tts_{ts}_{uuid.uuid4().hex[:6]}.{ext}"
            cloud_url = _cloud_put(audio_bytes, f"audio/{name}")
            if cloud_url:
                audio_url = cloud_url
            else:
                fp = OUTPUT_AUD_DIR / name
                fp.write_bytes(audio_bytes)
                audio_url = f"/outputs/audio/{fp.name}"
            return {"success": True, "audio_url": audio_url, "model": data.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Helpers ──────────────────────────────────────────────────────
def _handle_video_create_response(resp: httpx.Response, model: str) -> dict:
    rj = resp.json()
    if resp.status_code not in (200, 202):
        return JSONResponse(status_code=resp.status_code,
                            content={"success": False, "error": rj.get("error", {}).get("message", resp.text)})
    task_id = rj.get("id")
    if not task_id:
        return JSONResponse(status_code=500, content={"success": False, "error": f"No task id: {rj}"})
    return {"success": True, "task_id": task_id, "status": rj.get("status", "pending").upper(), "model": model}

async def _async_download_image(url: str) -> Optional[str]:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"img_{ts}_{uuid.uuid4().hex[:6]}.png"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=30)
            if r.status_code == 200:
                cloud_url = _cloud_put(r.content, f"images/{name}")
                if cloud_url:
                    return cloud_url
                fp = OUTPUT_IMG_DIR / name
                fp.write_bytes(r.content)
                return f"/outputs/images/{fp.name}"
    except Exception as e:
        print(f"Image download error: {e}")
    return None

async def _async_download_video(url: str) -> Optional[str]:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"vid_{ts}_{uuid.uuid4().hex[:6]}.mp4"
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=120) as r:
                if r.status_code == 200:
                    data = b"".join([chunk async for chunk in r.aiter_bytes(8192)])
                    cloud_url = _cloud_put(data, f"videos/{name}")
                    if cloud_url:
                        return cloud_url
                    fp = OUTPUT_VID_DIR / name
                    fp.write_bytes(data)
                    return f"/outputs/videos/{fp.name}"
    except Exception as e:
        print(f"Video download error: {e}")
    return None

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("NenAI Testing Platform (FastAPI)")
    print("   Auth: NenAI API Key")
    print("   Endpoint: http://0.0.0.0:5050")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5050))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

