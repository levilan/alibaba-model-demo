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
        # ── 代碼 ──────────────────────────────────────────────────
        {"id": "qwen3-coder-plus",   "name": "Qwen3 Coder Plus", "group": "代碼",   "desc": "代碼生成旗艦",           "thinking": True},
        {"id": "qwen3-coder-flash",  "name": "Qwen3 Coder Flash","group": "代碼",   "desc": "代碼生成極速",           "thinking": True},
        # ── 角色 ──────────────────────────────────────────────────
        {"id": "qwen-plus-character", "name": "Qwen Plus Character","group": "角色", "desc": "角色扮演，Plus 品質",   "thinking": False},
        # ── 第三方 ────────────────────────────────────────────────
        # DeepSeek/GLM 實測都支援 enable_thinking 開關（會回傳獨立的 reasoning_content
        # 思考過程），DeepSeek V4 系列預設就是思考模式開啟，enable_thinking:false 可關閉
        {"id": "deepseek-v4-pro",    "name": "DeepSeek V4 Pro",  "group": "第三方", "desc": "最新旗艦推理",           "thinking": True},
        {"id": "deepseek-v4-flash",  "name": "DeepSeek V4 Flash","group": "第三方", "desc": "最新極速推理",           "thinking": True},
        {"id": "deepseek-v3.2",      "name": "DeepSeek V3.2",    "group": "第三方", "desc": "前代深度推理",           "thinking": True},
        {"id": "glm-5.1",            "name": "GLM 5.1",          "group": "第三方", "desc": "智譜 GLM 前一版",        "thinking": True},
        {"id": "glm-5.2",            "name": "GLM 5.2",          "group": "第三方", "desc": "智譜 GLM 最新版",        "thinking": True},
        # ── Claude（實測過 enable_thinking 與 Anthropic 原生 thinking 參數在這個
        #    網關上都不會回傳任何思考過程，thinking 一律維持 False；temperature/
        #    top_p 也不能送，Bedrock 後端會直接回 400 "temperature is deprecated"）──
        {"id": "claude-opus-4-8",             "name": "Claude Opus 4.8",   "group": "Claude", "desc": "旗艦，最強推理",   "thinking": False},
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
        {"id": "gpt-5.6-terra", "name": "GPT 5.6 Terra", "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.6-sol",   "name": "GPT 5.6 Sol",   "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.6-luna",  "name": "GPT 5.6 Luna",  "group": "GPT", "desc": "最新特化模型", "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.5",       "name": "GPT 5.5",       "group": "GPT", "desc": "均衡模型",     "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.4",       "name": "GPT 5.4",       "group": "GPT", "desc": "均衡模型",     "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.4-mini",  "name": "GPT 5.4 Mini",  "group": "GPT", "desc": "輕量極速",     "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.4-nano",  "name": "GPT 5.4 Nano",  "group": "GPT", "desc": "超輕量極速",   "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5.2",       "name": "GPT 5.2",       "group": "GPT", "desc": "前代均衡模型", "thinking": False, "reasoning_effort": True},
        {"id": "gpt-5-mini",    "name": "GPT 5 Mini",    "group": "GPT", "desc": "前代輕量模型", "thinking": False, "reasoning_effort": True},
        # ── Gemini（3.x 系列實測會無條件消耗 reasoning_tokens 思考，enable_thinking/
        #    reasoning_effort 兩種參數都試過，沒有任何一個能讓它降到 0——這個網關上
        #    目前無法關閉，thinking 維持 False，沒有開關讓使用者假裝能控制）──
        {"id": "gemini-3.1-pro-preview",      "name": "Gemini 3.1 Pro Preview",      "group": "Gemini", "desc": "旗艦，最強推理",   "thinking": False},
        {"id": "gemini-3.6-flash",            "name": "Gemini 3.6 Flash",            "group": "Gemini", "desc": "新一代均衡模型",   "thinking": False},
        {"id": "gemini-3.5-flash",            "name": "Gemini 3.5 Flash",            "group": "Gemini", "desc": "前代均衡模型",     "thinking": False},
        {"id": "gemini-3-flash-preview",      "name": "Gemini 3 Flash Preview",      "group": "Gemini", "desc": "前代均衡模型",     "thinking": False},
        {"id": "gemini-2.5-pro",              "name": "Gemini 2.5 Pro",              "group": "Gemini", "desc": "前代旗艦",         "thinking": False},
        {"id": "gemini-2.5-flash",            "name": "Gemini 2.5 Flash",            "group": "Gemini", "desc": "前代均衡模型",     "thinking": False},
        {"id": "gemini-2.5-flash-lite",       "name": "Gemini 2.5 Flash Lite",       "group": "Gemini", "desc": "前代輕量極速",     "thinking": False},
    ],
    "image": [
        # ── 千問文生圖 ────────────────────────────────────────────
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
        # 實際張數由模型決定、不保證等於 n。wan2.7-image-pro 純文生圖情境下額外
        # 支援 2048*2048（2K）與 4096*4096（4K）高解析度，其餘情境上游僅支援到 2K。
        {
            "id": "wan2.7-image-pro", "name": "萬相 2.7 Image Pro", "group": "萬相文生圖",
            "desc": "旗艦文生圖，細節與畫質更佳", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960","2048*2048","4096*4096"],
            "supports_sequential": True,
        },
        {
            "id": "wan2.7-image", "name": "萬相 2.7 Image", "group": "萬相文生圖",
            "desc": "標準文生圖", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960"],
            "supports_sequential": True,
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
        {
            "id": "MAI-Image-2.5", "name": "MAI-Image-2.5", "group": "MAI Image",
            "desc": "旗艦圖像生成", "type": "t2i", "max_n": 4,
            "sizes": ["1024x1024","1536x1024","1024x1536"],
        },
        {
            "id": "MAI-Image-2.5-Flash", "name": "MAI-Image-2.5-Flash", "group": "MAI Image",
            "desc": "極速圖像生成", "type": "t2i", "max_n": 4,
            "sizes": ["1024x1024","1536x1024","1024x1536"],
        },
        # ── 萬相圖像編輯 ──────────────────────────────────────────
        {
            "id": "wan2.7-image-pro", "name": "萬相 2.7 Image Pro（編輯）", "group": "萬相圖像編輯",
            "desc": "多圖融合、風格遷移", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        {
            "id": "wan2.7-image", "name": "萬相 2.7 Image（編輯）", "group": "萬相圖像編輯",
            "desc": "標準圖像編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        {
            "id": "wan2.6-image", "name": "萬相 2.6 Image", "group": "萬相圖像編輯",
            "desc": "前代編輯模型", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
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
            "desc": "生成與編輯融合模型加速版", "type": "i2i", "max_n": 6, "max_ref": 3,
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
        # ── Gemini Image（走 /v1/chat/completions + modalities，不支援 size 參數；
        #    aspect_ratio 走「自然語言注入 prompt」而非結構化欄位——實測過
        #    imageConfig/aspect_ratio/generationConfig 這些結構化參數在這個
        #    網關上一律被靜默忽略，直接在 prompt 文字裡要求比例反而有效）──
        {
            "id": "gemini-3-pro-image", "name": "Gemini 3 Pro Image", "group": "Gemini Image",
            "desc": "Google 旗艦圖像生成，畫質最佳", "type": "t2i", "max_n": 4, "no_size": True,
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        },
        {
            "id": "gemini-3.1-flash-image", "name": "Gemini 3.1 Flash Image", "group": "Gemini Image",
            "desc": "速度與品質平衡，建議日常使用", "type": "t2i", "max_n": 4, "no_size": True,
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        },
        {
            "id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image", "group": "Gemini Image",
            "desc": "穩定版，較成熟的圖像模型", "type": "t2i", "max_n": 4, "no_size": True,
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"],
        },
        {
            "id": "gemini-3.1-flash-lite-image", "name": "Gemini 3.1 Flash Lite Image", "group": "Gemini Image",
            "desc": "輕量極速圖像生成", "type": "t2i", "max_n": 4, "no_size": True,
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"],
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
        # ── MAI Image 編輯（Azure OpenAI 管道，沿用一般 /v1/images/edits 流程，不支援 ref_strength）──
        {
            "id": "MAI-Image-2.5", "name": "MAI-Image-2.5（編輯）", "group": "MAI Image",
            "desc": "旗艦圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True,
            "sizes": ["1024x1024","1536x1024","1024x1536"],
        },
        {
            "id": "MAI-Image-2.5-Flash", "name": "MAI-Image-2.5-Flash（編輯）", "group": "MAI Image",
            "desc": "極速圖像編輯", "type": "i2i", "max_n": 1, "no_ref_strength": True,
            "sizes": ["1024x1024","1536x1024","1024x1536"],
        },
        # ── Gemini Image 編輯（走 /v1/chat/completions + modalities，帶入參考圖）──
        {
            "id": "gemini-3-pro-image", "name": "Gemini 3 Pro Image（編輯）", "group": "Gemini Image",
            "desc": "Google 旗艦圖像編輯，畫質最佳", "type": "i2i", "max_n": 1, "no_size": True, "no_ref_strength": True,
        },
        {
            "id": "gemini-3.1-flash-image", "name": "Gemini 3.1 Flash Image（編輯）", "group": "Gemini Image",
            "desc": "速度與品質平衡，建議日常使用", "type": "i2i", "max_n": 1, "no_size": True, "no_ref_strength": True,
        },
        {
            "id": "gemini-2.5-flash-image", "name": "Gemini 2.5 Flash Image（編輯）", "group": "Gemini Image",
            "desc": "穩定版，較成熟的圖像模型", "type": "i2i", "max_n": 1, "no_size": True, "no_ref_strength": True,
        },
        {
            "id": "gemini-3.1-flash-lite-image", "name": "Gemini 3.1 Flash Lite Image（編輯）", "group": "Gemini Image",
            "desc": "輕量極速圖像編輯", "type": "i2i", "max_n": 1, "no_size": True, "no_ref_strength": True,
        },
    ],
    "video": [
        # ── 文生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-t2v", "name": "萬相 2.7 T2V", "group": "文生影片",   "desc": "多鏡頭、自動配音", "type": "t2v",   "audio": True,  "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-t2v", "name": "萬相 2.6 T2V", "group": "文生影片",   "desc": "前代文生影片",     "type": "t2v",   "audio": True, "min_dur": 2, "max_dur": 15},
        # ── 圖生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-i2v", "name": "萬相 2.7 I2V", "group": "圖生影片",   "desc": "首幀/首尾幀/配音/影片延伸", "type": "i2v", "audio": True, "min_dur": 2, "max_dur": 15},
        # wan2.6-i2v / wan2.6-i2v-flash 目前 NenAI 平台端 pipeline 故障（無論送任何欄位格式都回
        # "Field required: input.img_url"，已用直連 API 排除是本專案的請求格式問題），保留在清單中等待平台方修復
        {"id": "wan2.6-i2v", "name": "萬相 2.6 I2V", "group": "圖生影片",   "desc": "前代圖生影片",       "type": "i2v", "audio": True, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-i2v-flash", "name": "萬相 2.6 I2V Flash", "group": "圖生影片", "desc": "前代圖生影片極速版", "type": "i2v", "audio": True, "min_dur": 2, "max_dur": 15},
        # ── 參考生影片 ────────────────────────────────────────────
        {"id": "wan2.7-r2v", "name": "萬相 2.7 R2V", "group": "參考生影片", "desc": "角色形象參考",       "type": "r2v", "audio": True, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-r2v", "name": "萬相 2.6 R2V", "group": "參考生影片", "desc": "前代參考生影片",     "type": "r2v", "audio": True, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-r2v-flash", "name": "萬相 2.6 R2V Flash", "group": "參考生影片", "desc": "前代參考生影片極速版", "type": "r2v", "audio": True, "min_dur": 2, "max_dur": 15},
        # ── HappyHorse ────────────────────────────────────────────
         {"id": "happyhorse-1.1-t2v",        "name": "HappyHorse 1.1 T2V",        "group": "HappyHorse", "desc": "高還原度文生影片",          "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 15},    
        {"id": "happyhorse-1.0-t2v",        "name": "HappyHorse 1.0 T2V",        "group": "HappyHorse", "desc": "前一代高還原度文生影片",          "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.1-i2v",        "name": "HappyHorse 1.1 I2V",        "group": "HappyHorse", "desc": "高還原度圖生影片（首幀）",   "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-i2v",        "name": "HappyHorse 1.0 I2V",        "group": "HappyHorse", "desc": "前一代高還原度圖生影片（首幀）",   "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.1-r2v",        "name": "HappyHorse 1.1 R2V",        "group": "HappyHorse", "desc": "多圖參考生影片（最多 9 張）", "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-r2v",        "name": "HappyHorse 1.0 R2V",        "group": "HappyHorse", "desc": "前一代多圖參考生影片（最多 9 張）", "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-video-edit", "name": "HappyHorse Video Edit 1.0", "group": "HappyHorse", "desc": "視頻編輯（最多 5 張參考圖）", "type": "vedit", "audio": False, "min_dur": 3, "max_dur": 15},      
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
        # ── Gemini Omni（走 /v1beta/interactions，模型自行決定長度/解析度，固定含原生配音）──
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview", "group": "Gemini",
         "desc": "Google 多模態影片生成（預覽版），最長約 10 秒，自動含原生配音（無需另設定）", "type": "t2v", "audio": False, "no_duration": True},
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview（圖生影片）", "group": "Gemini",
         "desc": "Google 多模態圖生影片（預覽版），最長約 10 秒，自動含原生配音（無需另設定）", "type": "i2v", "audio": False, "no_duration": True},
        {"id": "gemini-omni-flash-preview", "name": "Gemini Omni Flash Preview（參考生影片）", "group": "Gemini",
         "desc": "Google 多模態參考生影片（預覽版，最多 3 張參考圖），最長約 10 秒，自動含原生配音（無需另設定）", "type": "r2v", "audio": False, "no_duration": True},
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

@app.post("/login")
async def login(data: LoginRequest):
    """Validate NenAI API key."""
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
        return {"success": True}
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err or "invalid" in err.lower():
            return JSONResponse(status_code=401, content={"success": False, "message": "API Key 無效或權限不足"})
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

# Canvas 節點間傳遞的圖片/影片網址可能落在 OSS bucket，瀏覽器直接 fetch 會被 CORS 擋下，
# 故提供一個僅允許白名單網域的代理端點；不可放行任意網址，否則會變成 SSRF 入口
_PROXY_ALLOWED_SUFFIXES = (".aliyuncs.com",)

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


@app.websocket("/ws/omni")
async def ws_omni_proxy(websocket: WebSocket, api_key: str, model: str = "qwen3.5-omni-flash-realtime"):
    await websocket.accept()
    url = f"wss://nen.com.tw/api-ws/v1/realtime?model={model}"
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

@app.post("/api/text/generate")
async def text_generate(data: TextGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    messages = []
    if data.system_prompt:
        messages.append({"role": "system", "content": data.system_prompt})
    messages.extend(data.history)
    messages.append({"role": "user", "content": data.prompt})

    extra_body = {}
    if data.enable_thinking:
        extra_body["enable_thinking"] = True

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
                result["usage"] = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
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
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
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

# Gemini 圖像模型偶爾會不出圖、只回一段純文字聊天式回覆——實測發現這跟 prompt
# 讀起來像不像「聊天訊息」高度相關：越像一段對話/討論文字（例如上游文字節點
# 生成的長篇分析），模型就越容易把它當成聊天來回覆而不畫圖。加上明確的繪圖
# 指令前綴可顯著改善成功率，仍會不穩定則再靠重試補強。
_GEMINI_IMAGE_MAX_RETRIES = 2

async def _generate_gemini_chat_image(model: str, prompt: str, n: int, api_key: str,
                                       image_files: Optional[list] = None,
                                       aspect_ratio: Optional[str] = None) -> dict:
    if image_files:
        content: Any = [{"type": "text", "text": f"Edit the image(s) as follows: {prompt}"}]
        for fname, fbytes, ftype in image_files:
            b64 = base64.b64encode(fbytes).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{ftype};base64,{b64}"}})
    elif aspect_ratio:
        # 實測過結構化的 imageConfig/aspect_ratio/generationConfig 欄位在這個網關上
        # 一律被靜默忽略（回傳圖片永遠是預設比例），改成直接在 prompt 文字裡用自然
        # 語言要求比例才有效——這不是 Gemini 官方 API 的正規做法，是繞過網關限制
        # 的權宜之計，未來若網關支援結構化參數應優先改用那個。
        content = f"Generate an image with aspect ratio {aspect_ratio} depicting: {prompt}"
    else:
        content = f"Generate an image depicting: {prompt}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text", "image"],
        "n": n,
    }
    last_text = ""
    async with httpx.AsyncClient(timeout=180.0) as client:
        for attempt in range(_GEMINI_IMAGE_MAX_RETRIES + 1):
            resp = await client.post(
                f"{NENAI_V1}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            rj = resp.json()
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code,
                                    content={"error": rj.get("error", {}).get("message", resp.text)})
            images = []
            for choice in rj.get("choices", []):
                content = choice.get("message", {}).get("content", "") or ""
                last_text = content
                for ext, b64 in _B64_IMAGE_RE.findall(content):
                    raw = base64.b64decode(b64)
                    images.append({"url": None, "local_path": await _save_image_bytes(raw, ext), "actual_prompt": None})
            if images:
                return {"success": True, "images": images, "model": model}
        preview = last_text[:200] + ("…" if len(last_text) > 200 else "")
        return JSONResponse(status_code=500, content={
            "error": f"模型未回傳圖片，改用純文字回覆（重試 {_GEMINI_IMAGE_MAX_RETRIES} 次仍失敗）：{preview}"
        })

# qwen-image-2.0 系列為「生成與編輯融合模型」：最多 3 張參考圖、可輸出 1-6 張，且不支援 ref_strength 參數
_QWEN2_EDIT_MODELS = {"qwen-image-2.0-pro", "qwen-image-2.0"}
# GPT Image 系列額外支援 OpenAI 標準的 quality/background/output_format 三個參數（已實測確認有效）
_GPT_IMAGE_MODELS = {"gpt-image-2", "gpt-image-1.5"}
# 支援組圖模式（enable_sequential）與更高解析度的萬相 2.7 系列
_WAN27_IMAGE_MODELS = {"wan2.7-image-pro", "wan2.7-image"}

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

@app.post("/api/image/generate")
async def image_generate(data: ImageGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    if data.model in _GEMINI_CHAT_IMAGE_MODELS:
        try:
            return await _generate_gemini_chat_image(data.model, data.prompt, data.n, api_key,
                                                       aspect_ratio=data.aspect_ratio)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    payload: dict = {"model": data.model, "prompt": data.prompt, "n": data.n, "size": data.size}
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
        async with httpx.AsyncClient(timeout=120.0) as client:
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
            if not images:
                return JSONResponse(status_code=500, content={"error": f"No images in response: {rj}"})
            return {"success": True, "images": images, "model": data.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GPT Image 系列的 /images/edits 不接受 ref_strength 參數，帶入會被上游拒絕（400 Unknown parameter）
_NO_REF_STRENGTH_EDIT_MODELS = _QWEN2_EDIT_MODELS | {"gpt-image-2", "gpt-image-1.5", "MAI-Image-2.5", "MAI-Image-2.5-Flash"}

# ─── API: Image Edit (I2I) ────────────────────────────────────────
@app.post("/api/image/edit")
async def image_edit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model       = form.get("model", "wan2.6-image")
    is_qwen2_edit = model in _QWEN2_EDIT_MODELS
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
    n = max(1, min(6, n)) if is_qwen2_edit else 1
    quality       = form.get("quality", "")
    background    = form.get("background", "")
    output_format = form.get("output_format", "")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Read and optionally resize reference images in memory
    # qwen-image-2.0 系列（生成與編輯融合模型）最多 3 張參考圖，其餘模型最多 9 張
    max_refs = 3 if is_qwen2_edit else 9
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
            return await _generate_gemini_chat_image(model, prompt, n, api_key, image_files)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        form_data = {"model": model, "prompt": prompt, "size": size, "n": str(n)}
        if is_qwen2_edit:
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

        files = [(("image" if i == 0 else f"image_{i+1}"), (fname, fbytes, ftype))
                 for i, (fname, fbytes, ftype) in enumerate(image_files)]

        async with httpx.AsyncClient(timeout=120.0) as client:
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
            if not images:
                return JSONResponse(status_code=500, content={"error": f"No images in response: {rj}"})
            return {"success": True, "images": images, "model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Resolution helper ────────────────────────────────────────────
_RESOLUTION_WH = {"480P": (854, 480), "720P": (1280, 720), "1080P": (1920, 1080)}

def _res_to_wh(resolution: str) -> tuple[int, int]:
    if resolution.upper() in _RESOLUTION_WH:
        return _RESOLUTION_WH[resolution.upper()]
    for sep in ("*", "x", "X"):
        if sep in resolution:
            w, h = resolution.split(sep, 1)
            return int(w), int(h)
    return 1280, 720

# Gemini Omni 不走 /v1/videos 的非同步任務模式，而是同步呼叫 /v1beta/interactions 直接拿到完成的影片
_INTERACTIONS_VIDEO_MODELS = {"gemini-omni-flash-preview"}
# Veo 預設的 personGeneration 安全設定較嚴格，帶真人圖片容易被擋，明確放寬為 allow_adult
_VEO_MODELS = {"veo-3.1-generate-001", "veo-3.1-fast-generate-001", "veo-3.1-lite-generate-001"}
_OMNI_TASK_CACHE: Dict[str, dict] = {}

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
    ratio           = form.get("ratio", "16:9")
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

    w, h = _res_to_wh(resolution)
    payload: dict = {"model": model, "prompt": prompt,
                     "duration": duration, "width": w, "height": h}
    meta: dict = {}
    if negative_prompt: meta["negative_prompt"] = negative_prompt
    if prompt_extend:   meta["prompt_extend"] = True
    if watermark:       meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if ratio:           meta["ratio"] = ratio
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"

    if audio_file and hasattr(audio_file, "filename") and audio_file.filename:
        ab = await audio_file.read()
        audio_mime = audio_file.content_type or "audio/mpeg"
        meta["audio"] = f"data:{audio_mime};base64,{base64.b64encode(ab).decode()}"
    else:
        # 上游未收到 audio 欄位時會自行判斷是否配音，不會視為「不要配音」——
        # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
        meta["audio"] = audio

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
    ratio         = form.get("ratio", "16:9")
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

    if model in _INTERACTIONS_VIDEO_MODELS:
        first_bytes = await _read_image_bytes(first_frame_file)
        if not first_bytes:
            return JSONResponse(status_code=400, content={"error": "I2V 需要上傳首幀圖片"})
        try:
            return await _generate_omni_video(model, prompt, api_key, [(first_bytes, "image/png")])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    w, h = _res_to_wh(resolution)
    actual_duration = duration
    payload: dict = {"model": model, "prompt": prompt,
                     "duration": actual_duration, "width": w, "height": h}
    meta: dict = {"i2v_mode": i2v_mode}
    if neg_prompt:    meta["negative_prompt"] = neg_prompt
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if ratio:         meta["ratio"] = ratio
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"
    if audio_bgm_file and hasattr(audio_bgm_file, "filename") and audio_bgm_file.filename:
        ab = await audio_bgm_file.read()
        audio_mime = audio_bgm_file.content_type or "audio/mpeg"
        meta["audio"] = f"data:{audio_mime};base64,{base64.b64encode(ab).decode()}"
    else:
        # 上游未收到 audio 欄位時會自行判斷是否配音，不會視為「不要配音」——
        # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
        meta["audio"] = audio_bgm

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
        clip_b64 = f"data:video/mp4;base64,{base64.b64encode(clip_bytes).decode()}"
        media_arr.append({"url": clip_b64, "type": "first_clip"})
        actual_duration = max(duration, 15)
        payload["duration"] = actual_duration
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
        if audio_file and hasattr(audio_file, "filename") and audio_file.filename:
            ab = await audio_file.read()
            if ab: media_arr.append({"url": f"data:audio/mpeg;base64,{base64.b64encode(ab).decode()}", "type": "driving_audio"})

    # `media` array + `image` (first item URL) for maximum compatibility
    payload["media"] = media_arr
    if media_arr:
        payload["image"] = media_arr[0]["url"]
        # 平台 TaskSubmitReq 只認 images（陣列），media/image 會被忽略
        payload["images"] = [m["url"] for m in media_arr]
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
    video_b64 = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}"
    w, h = _res_to_wh(resolution)

    media_arr: list = [{"url": video_b64, "type": "video"}]
    max_refs = 5 if "happyhorse" in model else 3
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
    if ratio:         meta["ratio"] = ratio

    payload: dict = {
        "model": model, "prompt": prompt,
        "duration": duration, "width": w, "height": h,
        "media": media_arr, "image": video_b64,
        "images": [m["url"] for m in media_arr],
        "metadata": meta,
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

# ─── API: Video R2V ───────────────────────────────────────────────
@app.post("/api/video/r2v")
async def video_r2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model         = form.get("model", "wan2.6-r2v")
    prompt        = form.get("prompt", "")
    resolution    = form.get("resolution", "720P")
    ratio         = form.get("ratio", "16:9")
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
    media_arr: list = []
    image_files: list = []
    for f in ref_files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        fb = await f.read()
        mime = "video/mp4" if ext in VIDEO_EXTS else "image/png"
        media_type = "reference_video" if ext in VIDEO_EXTS else "reference_image"
        media_arr.append({"url": f"data:{mime};base64,{base64.b64encode(fb).decode()}", "type": media_type})
        if media_type == "reference_image":
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

    w, h = _res_to_wh(resolution)
    meta: dict = {}
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if ratio:         meta["ratio"] = ratio
    if model in _VEO_MODELS: meta["person_generation"] = "allow_adult"
    if audio_bgm_file and hasattr(audio_bgm_file, "filename") and audio_bgm_file.filename:
        ab = await audio_bgm_file.read()
        audio_mime = audio_bgm_file.content_type or "audio/mpeg"
        meta["audio"] = f"data:{audio_mime};base64,{base64.b64encode(ab).decode()}"
    else:
        # 上游未收到 audio 欄位時會自行判斷是否配音，不會視為「不要配音」——
        # 使用者關閉開關時務必明確帶 False 覆蓋掉上游的預設行為
        meta["audio"] = audio_bgm

    payload: dict = {"model": model, "prompt": prompt,
                     "duration": duration, "width": w, "height": h,
                     "media": media_arr,
                     "image": media_arr[0]["url"],
                     "images": [m["url"] for m in media_arr]}
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

    payload = {
        "model": model,
        "media": [
            {"url": f"data:{img_mime};base64,{base64.b64encode(img_bytes).decode()}", "type": "image"},
            {"url": f"data:video/mp4;base64,{base64.b64encode(vid_bytes).decode()}", "type": "video"},
        ],
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
            err = rj.get("error") or rj.get("task_info", {}).get("error") or {}
            result["error_message"] = (err.get("message") if isinstance(err, dict) else str(err)) or "Unknown error"

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
                    return JSONResponse(status_code=500, content={"error": f"上游未回傳音訊：{rj}"})
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

