"""
Alibaba Cloud AI Model Testing Platform
FastAPI Backend - API Key per-user authentication
"""
import os, sys, json, time, uuid, mimetypes, shutil
from PIL import Image as PILImage
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message
from dashscope import VideoSynthesis
from dashscope.audio.tts import SpeechSynthesizer as TTSv1
from dashscope.audio.asr import Recognition

_INTL_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
import requests as http_requests

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Alibaba Cloud AI Model Testing Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL_COMPATIBLE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_HTTP_URL  = "https://dashscope-intl.aliyuncs.com/api/v1"
CUSTOM_HEADERS = {"X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'}

# 設定 dashscope 全域 base URL（只需設定一次，不可當 call() 的 kwarg 傳入）
dashscope.base_http_api_url = DASHSCOPE_HTTP_URL
dashscope.base_websocket_api_url = _INTL_WS_URL

UPLOAD_DIR      = Path(__file__).parent / "static" / "uploads"
OUTPUT_IMG_DIR  = Path(__file__).parent / "outputs" / "images"
OUTPUT_VID_DIR  = Path(__file__).parent / "outputs" / "videos"
OUTPUT_AUDIO_DIR = Path(__file__).parent / "outputs" / "audio"
for d in (UPLOAD_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR, OUTPUT_AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 靜態檔案掛載
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=Path(__file__).parent / "outputs"), name="outputs")

# ─── Model Registry ───────────────────────────────────────────
# sizes: 支援的尺寸清單；max_n: 最大生成張數；audio: 支援配音；min/max_dur: 影片時長範圍
MODELS = {
    "text": [
        # ── 旗艦 ──────────────────────────────────────────────────
        {"id": "qwen3.7-max",        "name": "Qwen3.7 Max",      "group": "旗艦",   "desc": "最新旗艦，最強推理",     "thinking": True},
        {"id": "qwen3.6-max-preview","name": "Qwen3.6 Max",     "group": "旗艦",   "desc": "新一代旗艦，強推理",     "thinking": True},
        {"id": "qwen3-max",          "name": "Qwen3 Max",        "group": "旗艦",   "desc": "最強推理，262K context", "thinking": True},
        # ── 均衡 ──────────────────────────────────────────────────
        {"id": "qwen3.6-plus",       "name": "Qwen3.6 Plus",     "group": "均衡",   "desc": "1M context，性價比最佳", "thinking": True},
        {"id": "qwen3.5-plus",       "name": "Qwen3.5 Plus",     "group": "均衡",   "desc": "前代均衡模型",           "thinking": True},
        {"id": "qwen-plus",          "name": "Qwen Plus",        "group": "均衡",   "desc": "穩定均衡，廣泛任務",     "thinking": False},
        # ── 極速 ──────────────────────────────────────────────────
        {"id": "qwen3.6-flash",      "name": "Qwen3.6 Flash",    "group": "極速",   "desc": "新一代極速模型",         "thinking": True},
        {"id": "qwen3.5-flash",      "name": "Qwen3.5 Flash",    "group": "極速",   "desc": "速度快、成本低",         "thinking": True},
        {"id": "qwen-flash",         "name": "Qwen Flash",       "group": "極速",   "desc": "前代極速模型",           "thinking": False},
        # ── 代碼 ──────────────────────────────────────────────────
        {"id": "qwen3-coder-plus",   "name": "Qwen3 Coder Plus", "group": "代碼",   "desc": "代碼生成旗艦",           "thinking": True},
        {"id": "qwen3-coder-flash",  "name": "Qwen3 Coder Flash","group": "代碼",   "desc": "代碼生成極速",           "thinking": True},
        # ── 翻譯 ──────────────────────────────────────────────────
        {"id": "qwen-mt-plus",       "name": "Qwen MT Plus",     "group": "翻譯",   "desc": "機器翻譯，高品質",       "thinking": False},
        {"id": "qwen-mt-flash",      "name": "Qwen MT Flash",    "group": "翻譯",   "desc": "機器翻譯，極速",         "thinking": False},
        {"id": "qwen-mt-lite",       "name": "Qwen MT Lite",     "group": "翻譯",   "desc": "機器翻譯，輕量",         "thinking": False},
        # ── 角色 ──────────────────────────────────────────────────
        {"id": "qwen-flash-character","name": "Qwen Flash Character","group": "角色", "desc": "角色扮演專用",          "thinking": False},
        # ── 第三方 ────────────────────────────────────────────────
        {"id": "deepseek-v3.2",      "name": "DeepSeek V3.2",    "group": "第三方", "desc": "深度推理（國際版）",     "thinking": False},
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
        {
            "id": "qwen-image-max", "name": "千問圖像 Max", "group": "千問文生圖",
            "desc": "旗艦畫質，細節豐富", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        {
            "id": "qwen-image-plus", "name": "千問圖像 Plus", "group": "千問文生圖",
            "desc": "均衡品質與速度", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        # ── 萬相文生圖 ────────────────────────────────────────────
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
        # ── 萬相圖像編輯 ──────────────────────────────────────────
        {
            "id": "wan2.7-image-pro", "name": "萬相 2.7 Image Pro", "group": "萬相圖像編輯",
            "desc": "多圖融合、風格遷移", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        {
            "id": "wan2.7-image", "name": "萬相 2.7 Image", "group": "萬相圖像編輯",
            "desc": "標準圖像編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        {
            "id": "wan2.6-image", "name": "萬相 2.6 Image", "group": "萬相圖像編輯",
            "desc": "前代編輯模型", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960"],
        },
        # ── 千問圖像編輯 ──────────────────────────────────────────
        {
            "id": "qwen-image-edit-max", "name": "千問圖像編輯 Max", "group": "千問圖像編輯",
            "desc": "複雜圖文編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
        {
            "id": "qwen-image-edit-plus", "name": "千問圖像編輯 Plus", "group": "千問圖像編輯",
            "desc": "輕量圖文編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
    ],
    "video": [
        # ── 文生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-t2v", "name": "萬相 2.7 T2V", "group": "文生影片",   "desc": "多鏡頭、自動配音", "type": "t2v",   "audio": True,  "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-t2v", "name": "萬相 2.6 T2V", "group": "文生影片",   "desc": "前代文生影片",     "type": "t2v",   "audio": False, "min_dur": 2, "max_dur": 15},
        # ── 圖生影片 ──────────────────────────────────────────────
        {"id": "wan2.7-i2v", "name": "萬相 2.7 I2V", "group": "圖生影片",   "desc": "首幀/首尾幀/配音/影片延伸", "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-i2v", "name": "萬相 2.6 I2V", "group": "圖生影片",   "desc": "前代圖生影片",       "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-i2v-flash", "name": "萬相 2.6 I2V Flash", "group": "圖生影片", "desc": "前代圖生影片極速版", "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15},
        # ── 參考生影片 ────────────────────────────────────────────
        {"id": "wan2.7-r2v", "name": "萬相 2.7 R2V", "group": "參考生影片", "desc": "角色形象參考",       "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-r2v", "name": "萬相 2.6 R2V", "group": "參考生影片", "desc": "前代參考生影片",     "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-r2v-flash", "name": "萬相 2.6 R2V Flash", "group": "參考生影片", "desc": "前代參考生影片極速版", "type": "r2v", "audio": False, "min_dur": 2, "max_dur": 15},
        # ── HappyHorse ────────────────────────────────────────────
        {"id": "happyhorse-1.0-t2v",        "name": "HappyHorse T2V",        "group": "HappyHorse", "desc": "高還原度文生影片",          "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-i2v",        "name": "HappyHorse I2V",        "group": "HappyHorse", "desc": "高還原度圖生影片（首幀）",   "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-r2v",        "name": "HappyHorse R2V",        "group": "HappyHorse", "desc": "多圖參考生影片（最多 9 張）", "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 15},
        {"id": "happyhorse-1.0-video-edit", "name": "HappyHorse Video Edit", "group": "HappyHorse", "desc": "視頻編輯（最多 5 張參考圖）", "type": "vedit", "audio": False, "min_dur": 3, "max_dur": 15},
        # ── 視頻編輯 ──────────────────────────────────────────────
        {"id": "wan2.7-videoedit", "name": "萬相 2.7 視頻編輯", "group": "萬相視頻編輯",
         "desc": "文字/參考圖驅動編輯", "type": "vedit", "audio": False, "min_dur": 2, "max_dur": 15},
    ],
    "voice": {
        "asr": [
            {"id": "qwen3-asr-flash", "name": "Qwen3 ASR Flash",  "group": "Qwen3",   "desc": "新一代極速識別，多語言"},
            {"id": "paraformer-v2",   "name": "Fun-ASR 語音識別", "group": "Fun-ASR", "desc": "高精度普通話識別"},
            {"id": "sensevoice-v1",   "name": "Fun-ASR 多語言",   "group": "Fun-ASR", "desc": "中/英/日/韓/粵多語言"},
        ],
                "tts": [
            {"id": "qwen-tts", "name": "Qwen TTS", "group": "Qwen", "desc": "HTTP 同步合成，穩定可靠"},
        ],
    },
    "muleai": [
        {"id": "wan2.7-i2v-spicy", "name": "Wan 2.7 I2V Spicy", "group": "MuleAI", "desc": "Spicy 模型 (支援文字/圖片)"},
    ],
}

# TTS 預設音色清單（qwen3-tts-flash / qwen-tts 共用）
TTS_VOICES = [
    {"id": "Cherry",   "name": "芊悅",   "gender": "女", "style": "親切"},
    {"id": "Ethan",    "name": "逸軒",   "gender": "男", "style": "穩重"},
    {"id": "Serena",   "name": "晨煦",   "gender": "女", "style": "清爽"},
    {"id": "Wayne",    "name": "韋恩",   "gender": "男", "style": "磁性"},
    {"id": "Summer",   "name": "甜茶",   "gender": "女", "style": "活潑"},
    {"id": "Belle",    "name": "不吃魚", "gender": "女", "style": "元氣"},
    {"id": "Cove",     "name": "詹妮弗", "gender": "女", "style": "知性"},
    {"id": "Aria",     "name": "卡捷琳娜","gender": "女", "style": "優雅"},
    {"id": "Kai",      "name": "嘉熙",   "gender": "男", "style": "輕快"},
    {"id": "Luna",     "name": "月桐",   "gender": "女", "style": "溫柔"},
]


# ─── Auth: API Key per user ────────────────────────────────────────
def get_api_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        api_key = auth[7:].strip()
    else:
        api_key = request.query_params.get("api_key", "").strip()
    
    if not (api_key and api_key.startswith("sk-") and len(api_key) > 20):
        raise HTTPException(status_code=401, detail="Unauthorized - invalid API key")
    return api_key

def get_muleai_api_key(request: Request) -> Optional[str]:
    return request.headers.get("X-MuleAI-API-Key", "").strip() or None

# ─── Pages ────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "templates" / "index.html")

class LoginRequest(BaseModel):
    api_key: str

@app.post("/login")
async def login(data: LoginRequest):
    """Validate DashScope API key by calling the models list endpoint."""
    api_key = data.api_key.strip()
    if not (api_key and api_key.startswith("sk-") and len(api_key) > 20):
        return JSONResponse(status_code=400, content={"success": False, "message": "API Key 格式有誤，須以 sk- 開頭"})

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
    return {**MODELS, "tts_voices": TTS_VOICES}

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

@app.post("/api/text/generate")
async def text_generate(data: TextGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    messages = []
    if data.system_prompt:
        messages.append({"role": "system", "content": data.system_prompt})
    messages.append({"role": "user", "content": data.prompt})

    extra_body = {}
    if data.enable_thinking:
        extra_body["enable_thinking"] = True

    create_kwargs = dict(
        model=data.model,
        messages=messages,
        temperature=data.temperature,
        top_p=data.top_p,
        max_tokens=data.max_tokens,
        presence_penalty=data.presence_penalty,
        frequency_penalty=data.frequency_penalty,
        stream=data.stream,
        extra_headers=CUSTOM_HEADERS,
        extra_body=extra_body or None,
    )
    if data.top_k is not None and data.top_k > 0:
        create_kwargs["extra_body"] = {**(extra_body or {}), "top_k": data.top_k}
    if data.seed is not None:
        create_kwargs["seed"] = data.seed
    if data.stop:
        create_kwargs["stop"] = data.stop[:4]

    if not data.stream:
        try:
            user_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            resp = user_client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            return {"content": content, "done": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate() -> AsyncGenerator[str, None]:
        try:
            user_client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            stream = await user_client.chat.completions.create(**create_kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )





# ─── API: MuleAI Video Generation (I2V) ───────────────────────────────────────
@app.post("/api/muleai/video")
async def muleai_video_generate(
    request: Request,
    model: str = Form("wan2.7-i2v-spicy"),
    prompt: str = Form(""),
    negative_prompt: Optional[str] = Form(None),
    resolution: str = Form("1080p"),
    duration: int = Form(5),
    prompt_extend: bool = Form(True),
    seed: Optional[int] = Form(None),
    image: UploadFile = File(...),
    muleai_key: Optional[str] = Depends(get_muleai_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    if not muleai_key:
        raise HTTPException(status_code=401, detail="MuleAI API Key is missing.")

    # We need to read the image file and either convert it to base64 or upload it somewhere.
    # The MuleRouter documentation says "URL or Base64-encoded string".
    image_bytes = await image.read()
    import base64
    b64_img = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = image.content_type or 'image/jpeg'
    data_uri = f"data:{mime_type};base64,{b64_img}"

    MULEAI_VIDEO_URL = f"https://api.mulerouter.ai/vendors/carrothub/v1/{model}/generation"
    headers = {
        "Authorization": f"Bearer {muleai_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "image": data_uri,
        "resolution": resolution,
        "duration": duration,
        "prompt_extend": prompt_extend
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed

    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(MULEAI_VIDEO_URL, headers=headers, json=payload)
            
            if resp.status_code == 202 or resp.status_code == 200:
                data = resp.json()
                task_id = data.get("task_info", {}).get("id") or data.get("id")
                return {"success": True, "task_id": task_id, "status": "pending"}
            else:
                return JSONResponse(status_code=resp.status_code, content={"success": False, "error": resp.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/muleai/video/status/{task_id}")
async def muleai_video_status(task_id: str, muleai_key: Optional[str] = Depends(get_muleai_api_key)):
    if not muleai_key:
        raise HTTPException(status_code=401, detail="MuleAI API Key is missing.")

    MULEAI_STATUS_URL = f"https://api.mulerouter.ai/vendors/carrothub/v1/wan2.7-i2v-spicy/generation/{task_id}"
    headers = {
        "Authorization": f"Bearer {muleai_key}"
    }

    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(MULEAI_STATUS_URL, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("task_info", {}).get("status", "pending")
                videos = data.get("videos", [])
                err = data.get("task_info", {}).get("error")
                return {"success": True, "status": status, "videos": videos, "error_message": err}
            else:
                return JSONResponse(status_code=resp.status_code, content={"success": False, "error": resp.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

@app.post("/api/image/generate")
async def image_generate(data: ImageGenerateRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        call_kwargs = dict(
            model=data.model,
            api_key=api_key,
            messages=[Message(role="user", content=[{"text": data.prompt}])],
            negative_prompt=data.negative_prompt or None,
            prompt_extend=data.prompt_extend,
            watermark=data.watermark,
            n=data.n,
            size=data.size,
            headers=CUSTOM_HEADERS,
        )
        if data.seed is not None:
            call_kwargs["seed"] = data.seed
        rsp = ImageGeneration.call(**call_kwargs)
        return _handle_image_response(rsp, data.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Image Edit (I2I) ────────────────────────────────────────
@app.post("/api/image/edit")
async def image_edit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = form.get("model", "wan2.6-image")
    prompt = form.get("prompt", "")
    negative_prompt = form.get("negative_prompt", "")
    size = form.get("size", "1024*1024")
    watermark = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    try:
        ref_strength = float(form.get("ref_strength", "0.5"))
    except ValueError:
        ref_strength = 0.5
    seed_str = str(form.get("seed", ""))
    seed = int(seed_str) if seed_str.strip() else None

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    image_urls = []
    for i in range(1, 10):
        f = form.get(f"image_{i}")
        if not f or not hasattr(f, "filename") or not f.filename:
            break
        ext = Path(f.filename).suffix or ".png"
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        with open(fp, "wb") as out_f:
            shutil.copyfileobj(f.file, out_f)
        image_urls.append(f"file://{fp.resolve()}")

    if not image_urls:
        return JSONResponse(status_code=400, content={"error": "至少需要一張參考圖片"})

    content = [{"text": prompt}] + [{"image": u} for u in image_urls]

    try:
        call_kwargs = dict(
            model=model,
            api_key=api_key,
            messages=[Message(role="user", content=content)],
            negative_prompt=negative_prompt or None,
            watermark=watermark,
            n=1,
            size=size,
            headers=CUSTOM_HEADERS,
        )
        if seed is not None:
            call_kwargs["seed"] = seed
        call_kwargs["ref_strength"] = ref_strength
        rsp = ImageGeneration.call(**call_kwargs)
        return _handle_image_response(rsp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_HAPPYHORSE_MODELS = {"happyhorse-1.0-t2v", "happyhorse-1.0-i2v",
                      "happyhorse-1.0-r2v", "happyhorse-1.0-video-edit"}
_SIZE_MAP = {"480P": "854*480", "720P": "1280*720", "1080P": "1920*1080"}

def _apply_resolution(kwargs: dict, model: str, resolution: str, ratio: str = "") -> None:
    if model in _HAPPYHORSE_MODELS:
        kwargs["resolution"] = resolution
        if ratio:
            kwargs["ratio"] = ratio
    else:
        kwargs["size"] = _SIZE_MAP.get(resolution, "1280*720")

# ─── API: Video T2V ───────────────────────────────────────────────
class VideoT2VRequest(BaseModel):
    model: str = "wan2.6-t2v"
    prompt: str = ""
    negative_prompt: str = ""
    resolution: str = "720P"
    ratio: str = "16:9"
    duration: int = 5
    audio: bool = False
    prompt_extend: bool = False
    watermark: bool = False
    seed: Optional[int] = None

@app.post("/api/video/t2v")
async def video_t2v(data: VideoT2VRequest, api_key: str = Depends(get_api_key)):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    kwargs = dict(
        model=data.model,
        prompt=data.prompt,
        duration=data.duration,
        prompt_extend=data.prompt_extend,
        watermark=data.watermark,
        api_key=api_key,
        headers=CUSTOM_HEADERS,
    )
    _apply_resolution(kwargs, data.model, data.resolution, data.ratio)
    if data.negative_prompt:
        kwargs["negative_prompt"] = data.negative_prompt
    if data.audio:
        kwargs["audio"] = True
    if data.seed is not None:
        kwargs["seed"] = data.seed

    try:
        rsp = VideoSynthesis.async_call(**kwargs)
        return _handle_video_async_response(rsp, data.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video I2V ───────────────────────────────────────────────
@app.post("/api/video/i2v")
async def video_i2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = form.get("model", "wan2.7-i2v")
    prompt = form.get("prompt", "")
    negative_prompt = form.get("negative_prompt", "")
    resolution = form.get("resolution", "720P")
    ratio = form.get("ratio", "16:9")
    duration = int(form.get("duration", 5))
    i2v_mode = form.get("i2v_mode", "first_frame")
    prompt_extend = str(form.get("prompt_extend", "false")).lower() in ("true", "1", "yes")
    watermark = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str = str(form.get("seed", ""))
    seed = int(seed_str) if seed_str.strip() else None

    import subprocess

    def _get_video_duration(filepath: Path) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Failed to get video duration: {e}")
            return 0.0

    def _save_file(file_obj, default_ext=".png", is_image=False):
        if not file_obj or not hasattr(file_obj, "filename") or not file_obj.filename:
            return None, None
        ext = Path(file_obj.filename).suffix or default_ext
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        if is_image:
            img = PILImage.open(file_obj.file)
            w, h = img.size
            if w < 240 or h < 240:
                scale = max(240 / w, 240 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
            if ext.lower() in (".jpg", ".jpeg") and img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(fp)
        else:
            with open(fp, "wb") as out_f:
                shutil.copyfileobj(file_obj.file, out_f)
        return f"file://{fp.resolve()}", fp

    media = []
    first_frame_file = form.get("first_frame") or form.get("image")
    last_frame_file  = form.get("last_frame")
    audio_file       = form.get("driving_audio")
    clip_file        = form.get("first_clip")

    if i2v_mode in ("first_clip", "first_clip_last_frame"):
        url, fp = _save_file(clip_file, ".mp4")
        if not url:
            return JSONResponse(status_code=400, content={"error": "first_clip 模式需要上傳影片片段"})
        
        # 檢查影片長度是否超過 10 秒限制 (嚴格限制為 9.9 秒以避免阿里雲後端解碼時的些微誤差)
        if fp:
            dur = _get_video_duration(fp)
            if dur > 9.9:
                return JSONResponse(status_code=400, content={"error": f"上傳影片長度為 {dur:.2f} 秒。阿里雲 API 嚴格限制參考影片不得超過 10 秒，為避免解碼誤差，請將影片修剪至 9.9 秒以內再上傳。"})

        media.append({"url": url, "type": "first_clip"})
        l_url, _ = _save_file(last_frame_file, ".png", is_image=True)
        if l_url:
            media.append({"url": l_url, "type": "last_frame"})
    else:
        url, _ = _save_file(first_frame_file, ".png", is_image=True)
        if not url:
            return JSONResponse(status_code=400, content={"error": "I2V 需要上傳首幀圖片"})
        media.append({"url": url, "type": "first_frame"})
        l_url, _ = _save_file(last_frame_file, ".png", is_image=True)
        if l_url:
            media.append({"url": l_url, "type": "last_frame"})
        a_url, _ = _save_file(audio_file, ".mp3")
        if a_url:
            media.append({"url": a_url, "type": "driving_audio"})

    try:
        kwargs = dict(
            model=model,
            prompt=prompt,
            duration=duration if i2v_mode not in ("first_clip", "first_clip_last_frame") else max(duration, 15), # 影片延伸模式時自動將 duration 設為最大值(15)，以涵蓋可能大於 5 秒的上傳片段
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
        # 只有舊版 wan 系列 (wan2.6, wan2.5, wan2.2) 仍嚴格要求使用 img_url，其餘新模型 (wan2.7, happyhorse) 皆使用 media
        if model.startswith("wan2.6") or model.startswith("wan2.5") or model.startswith("wan2.2"):
            for m in media:
                if m["type"] in ("first_frame", "first_clip"):
                    kwargs["img_url"] = m["url"]
                    break
            if "img_url" not in kwargs and media:
                kwargs["img_url"] = media[0]["url"]
        else:
            kwargs["media"] = media
        _apply_resolution(kwargs, model, resolution, ratio)
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            kwargs["seed"] = seed
        rsp = VideoSynthesis.async_call(**kwargs)
        return _handle_video_async_response(rsp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video Edit (wan2.7-videoedit) ──────────────────────────
@app.post("/api/video/vedit")
async def video_vedit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = form.get("model", "wan2.7-videoedit")
    prompt = form.get("prompt", "")
    negative_prompt = form.get("negative_prompt", "")
    resolution = form.get("resolution", "1080P")
    ratio = form.get("ratio", "")
    duration_str = str(form.get("duration", "0"))
    duration = int(duration_str) if duration_str.strip() else 0
    audio_setting = form.get("audio_setting", "auto")
    prompt_extend = str(form.get("prompt_extend", "true")).lower() in ("true", "1", "yes")
    watermark = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str = str(form.get("seed", ""))
    seed = int(seed_str) if seed_str.strip() else None

    video_file = form.get("video")
    if not video_file or not hasattr(video_file, "filename") or not video_file.filename:
        return JSONResponse(status_code=400, content={"error": "影片檔案為必填"})

    ext = Path(video_file.filename).suffix or ".mp4"
    vp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(vp, "wb") as out_f:
        shutil.copyfileobj(video_file.file, out_f)

    max_refs = 5 if model in _HAPPYHORSE_MODELS else 3
    media = [{"url": f"file://{vp.resolve()}", "type": "video"}]
    for i in range(1, max_refs + 1):
        ref = form.get(f"reference_image_{i}")
        if ref and hasattr(ref, "filename") and ref.filename:
            rext = Path(ref.filename).suffix or ".png"
            rp = UPLOAD_DIR / f"{uuid.uuid4().hex}{rext}"
            with open(rp, "wb") as out_f:
                shutil.copyfileobj(ref.file, out_f)
            media.append({"url": f"file://{rp.resolve()}", "type": "reference_image"})

    try:
        kwargs = dict(
            model=model,
            media=media,
            prompt=prompt,
            duration=duration,
            audio_setting=audio_setting,
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
        _apply_resolution(kwargs, model, resolution, ratio)
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            kwargs["seed"] = seed
        rsp = VideoSynthesis.async_call(**kwargs)
        return _handle_video_async_response(rsp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video R2V ───────────────────────────────────────────────
@app.post("/api/video/r2v")
async def video_r2v(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = form.get("model", "wan2.6-r2v")
    prompt = form.get("prompt", "")
    resolution = form.get("resolution", "720P")
    ratio = form.get("ratio", "16:9")
    duration = int(form.get("duration", 5))
    prompt_extend = str(form.get("prompt_extend", "false")).lower() in ("true", "1", "yes")
    watermark = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str = str(form.get("seed", ""))
    seed = int(seed_str) if seed_str.strip() else None

    files = form.getlist("reference_files")
    if not files or (len(files) > 0 and not hasattr(files[0], "filename")):
        return JSONResponse(status_code=400, content={"error": "At least one reference file is required"})

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    media = []
    for f in files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        ext = Path(f.filename).suffix or ".png"
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        with open(fp, "wb") as out_f:
            shutil.copyfileobj(f.file, out_f)
        media_type = (VideoSynthesis.MediaType.REFERENCE_VIDEO
                      if ext.lower() in VIDEO_EXTS
                      else VideoSynthesis.MediaType.REFERENCE_IMAGE)
        media.append({"url": f"file://{fp.resolve()}", "type": media_type})

    try:
        r2v_kwargs = dict(
            model=model,
            prompt=prompt,
            media=media,
            duration=duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
        _apply_resolution(r2v_kwargs, model, resolution, ratio)
        if seed is not None:
            r2v_kwargs["seed"] = seed
        rsp = VideoSynthesis.async_call(**r2v_kwargs)
        return _handle_video_async_response(rsp, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Video Status ────────────────────────────────────────────
@app.get("/api/video/status/{task_id}")
async def video_status(task_id: str, api_key: str = Depends(get_api_key)):
    try:
        dashscope.api_key = api_key
        rsp = VideoSynthesis.fetch(task_id)
        status = getattr(rsp.output, "task_status", "UNKNOWN")
        result = {"task_id": task_id, "status": status}
        if status == "SUCCEEDED":
            video_url = getattr(rsp.output, "video_url", "")
            if video_url:
                # Do not download locally anymore, just pass the remote URL
                result["local_path"] = video_url
                result["video_url"] = video_url
        elif status == "FAILED":
            result["error_message"] = getattr(rsp.output, "message", "Unknown")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Helpers ──────────────────────────────────────────────────────
def _handle_image_response(rsp, model):
    if rsp.status_code == 200:
        images = []
        output = rsp.output
        if hasattr(output, "choices") and output.choices:
            for choice in output.choices:
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    for item in choice.message.content:
                        url = None
                        if isinstance(item, dict):
                            url = item.get("image") or item.get("url")
                        elif hasattr(item, "image"):
                            url = item.image
                        elif hasattr(item, "url"):
                            url = item.url
                        if url:
                            images.append({"url": url, "local_path": _download_image(url)})
        elif hasattr(output, "results") and output.results:
            for r in output.results:
                url = None
                if isinstance(r, dict):
                    url = r.get("url")
                elif hasattr(r, "url"):
                    url = r.url
                if url:
                    images.append({"url": url, "local_path": _download_image(url)})
        if not images:
            return JSONResponse(status_code=500, content={"error": f"No images in response. output={output}"})
        return {"success": True, "images": images, "model": model}
    return JSONResponse(status_code=500, content={"error": f"Generation failed ({rsp.status_code}): {rsp.message}", "code": rsp.code})

def _handle_video_async_response(rsp, model):
    from http import HTTPStatus
    if rsp.status_code == HTTPStatus.OK:
        task_id = getattr(rsp.output, "task_id", "")
        task_status = getattr(rsp.output, "task_status", "PENDING")
        if task_id:
            return {"success": True, "task_id": task_id, "status": task_status, "model": model}
        if task_status == "SUCCEEDED":
            video_url = getattr(rsp.output, "video_url", "")
            return {"success": True, "status": "SUCCEEDED", "video_url": video_url,
                    "local_path": _download_video(video_url), "model": model}
        return JSONResponse(status_code=500, content={"error": f"No task_id in response. status={task_status}"})
    return JSONResponse(status_code=500, content={"error": f"API error ({rsp.status_code}): {rsp.message}", "code": rsp.code})

def _download_image(url):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = OUTPUT_IMG_DIR / f"img_{ts}_{uuid.uuid4().hex[:6]}.png"
        r = http_requests.get(url, timeout=30)
        if r.status_code == 200:
            fp.write_bytes(r.content)
            return f"/outputs/images/{fp.name}"
    except Exception as e:
        print(f"Image download error: {e}")
    return None

def _download_video(url):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = OUTPUT_VID_DIR / f"vid_{ts}_{uuid.uuid4().hex[:6]}.mp4"
        r = http_requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            with open(fp, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return f"/outputs/videos/{fp.name}"
    except Exception as e:
        print(f"Video download error: {e}")
    return None

# ─── API: Voice ASR ───────────────────────────────────────────────
_ASR_FMT = {".wav": "wav", ".mp3": "mp3", ".m4a": "m4a",
            ".flac": "flac", ".ogg": "ogg", ".opus": "opus"}

@app.post("/api/voice/asr")
async def voice_asr(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model = form.get("model", "paraformer-v2")
    audio_file = form.get("audio")
    
    if not audio_file or not hasattr(audio_file, "filename") or not audio_file.filename:
        return JSONResponse(status_code=400, content={"error": "請上傳音訊檔案"})

    ext = Path(audio_file.filename).suffix.lower() or ".wav"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(tmp_path, "wb") as out_f:
        shutil.copyfileobj(audio_file.file, out_f)

    try:
        audio_fmt = _ASR_FMT.get(ext, "wav")
        recognizer = Recognition(
            model=model,
            callback=None,
            format=audio_fmt,
            sample_rate=16000,
            api_key=api_key,
        )
        rsp = recognizer.call(f"file://{tmp_path.resolve()}")
        if rsp and rsp.status_code == 200:
            sentences = getattr(rsp, "get_sentence", None)
            if sentences:
                text = " ".join(s.get("text", "") for s in rsp.get_sentence())
            else:
                out = rsp.output if hasattr(rsp, "output") else {}
                text = (out.get("text", "") if isinstance(out, dict)
                        else getattr(out, "text", str(rsp)))
            return {"success": True, "text": text, "model": model}
        return JSONResponse(status_code=500, content={"error": f"ASR 失敗: {getattr(rsp, 'message', str(rsp))}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

# ─── API: Voice TTS ───────────────────────────────────────────────
class VoiceTTSRequest(BaseModel):
    model: str = "qwen-tts"
    voice: str = "Cherry"
    text: str = ""
    format: str = "mp3"

@app.post("/api/voice/tts")
async def voice_tts(data: VoiceTTSRequest, api_key: str = Depends(get_api_key)):
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="請輸入合成文字")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="文字長度不可超過 4000 字")

    try:
        rsp = TTSv1.call(
            model=data.model,
            text=text,
            voice=data.voice,
            format=data.format,
            sample_rate=22050,
            api_key=api_key,
        )
        audio_data = rsp.get_audio_data()
        if audio_data:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{ts}_{uuid.uuid4().hex[:6]}.{data.format}"
            fp = OUTPUT_AUDIO_DIR / filename
            fp.write_bytes(audio_data)
            return {"success": True, "audio_url": f"/outputs/audio/{filename}",
                    "model": data.model, "voice": data.voice}
        response = rsp.get_response()
        msg = getattr(response, "message", None) or getattr(response, "text", None) or str(response)
        return JSONResponse(status_code=500, content={"error": f"TTS 失敗: {msg}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Alibaba Cloud AI Model Testing Platform (FastAPI)")
    print("   Auth: DashScope API Key (sk-...)")
    print("   Endpoint: http://0.0.0.0:5050")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5050))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

