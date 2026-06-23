"""
NenAI Testing Platform
FastAPI Backend - NenAI API Key per-user authentication
"""
import os, sys, json, time, uuid, mimetypes, shutil, base64, subprocess
from io import BytesIO
from PIL import Image as PILImage
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Depends, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
import websockets
import asyncio
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="NenAI Testing Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NENAI_BASE          = "https://nen.com.tw"
BASE_URL_COMPATIBLE = f"{NENAI_BASE}/v1"
NENAI_V1            = f"{NENAI_BASE}/v1"

UPLOAD_DIR      = Path(__file__).parent / "static" / "uploads"
OUTPUT_IMG_DIR  = Path(__file__).parent / "outputs" / "images"
OUTPUT_VID_DIR  = Path(__file__).parent / "outputs" / "videos"
for d in (UPLOAD_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR):
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
        {"id": "deepseek-v4-pro",    "name": "DeepSeek V4 Pro",  "group": "第三方", "desc": "最新旗艦推理",           "thinking": False},
        {"id": "deepseek-v4-flash",  "name": "DeepSeek V4 Flash","group": "第三方", "desc": "最新極速推理",           "thinking": False},
        {"id": "deepseek-v3.2",      "name": "DeepSeek V3.2",    "group": "第三方", "desc": "前代深度推理",           "thinking": False},
        {"id": "glm-5.1",            "name": "GLM 5.1",          "group": "第三方", "desc": "智譜 GLM 最新版",        "thinking": False},
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
    ],
    "muleai": [
        {"id": "wan2.7-i2v-spicy", "name": "Wan 2.7 I2V Spicy", "group": "影片生成", "desc": "Spicy 模型 (支援文字/圖片)"},
        {"id": "z-image-spicy", "name": "Z-Image Spicy", "group": "圖片生成", "desc": "Spicy 圖片生成模型"},
    ],
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
@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "templates" / "index.html")

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
@app.post("/api/muleai/generate")
async def muleai_generate(
    request: Request,
    model: str = Form("wan2.7-i2v-spicy"),
    prompt: str = Form(""),
    negative_prompt: Optional[str] = Form(None),
    resolution: str = Form("1080p"),
    duration: Optional[int] = Form(5),
    img_resolution: Optional[str] = Form("1024*1536"),
    prompt_extend: bool = Form(True),
    seed: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
    api_key: str = Depends(get_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    is_image_model = "z-image" in model
    if is_image_model:
        MULEAI_URL = "https://nen.com.tw/v1/image/generations"
    else:
        MULEAI_URL = "https://nen.com.tw/v1/video/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
    }

    if not is_image_model:
        payload["size"] = resolution
        payload["duration"] = duration
        if image:
            image_bytes = await image.read()
            b64_img = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = image.content_type or 'image/jpeg'
            data_uri = f"data:{mime_type};base64,{b64_img}"
            payload["image"] = data_uri
        else:
            raise HTTPException(status_code=400, detail="Image is required for video generation")
    else:
        payload["prompt_extend"] = prompt_extend
        if img_resolution:
            parts = img_resolution.split("*")
            if len(parts) == 2:
                payload["width"] = int(parts[0])
                payload["height"] = int(parts[1])

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

    if "z-image" in model:
        MULEAI_STATUS_URL = f"https://nen.com.tw/v1/image/generations/{task_id}"
    else:
        MULEAI_STATUS_URL = f"https://nen.com.tw/v1/video/generations/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(MULEAI_STATUS_URL, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("task_info", {}).get("status", "pending")
                videos = data.get("videos", [])
                images = data.get("images", [])
                if not images and data.get("image_url"):
                    images = [data.get("image_url")]
                err = data.get("task_info", {}).get("error")

                if status.upper() in ("COMPLETED", "SUCCEEDED"):
                    if videos and len(videos) > 0:
                        local_path = await _async_download_video(videos[0])
                        if local_path:
                            videos = [local_path]
                    elif images and len(images) > 0:
                        local_path = await _async_download_image(images[0].get("url") if isinstance(images[0], dict) else images[0])
                        if local_path:
                            images = [local_path]
                return {"success": True, "status": status, "videos": videos, "images": images, "error_message": err}

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

    payload: dict = {"model": data.model, "prompt": data.prompt, "n": data.n, "size": data.size}
    if data.negative_prompt:
        payload["negative_prompt"] = data.negative_prompt
    if data.prompt_extend:
        payload["prompt_extend"] = True
    if data.watermark:
        payload["watermark"] = True
    if data.seed is not None:
        payload["seed"] = data.seed

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
            images = []
            for item in rj.get("data", []):
                url = item.get("url")
                if url:
                    images.append({"url": url, "local_path": await _async_download_image(url)})
            if not images:
                return JSONResponse(status_code=500, content={"error": f"No images in response: {rj}"})
            return {"success": True, "images": images, "model": data.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── API: Image Edit (I2I) ────────────────────────────────────────
@app.post("/api/image/edit")
async def image_edit(request: Request, api_key: str = Depends(get_api_key)):
    form = await request.form()
    model       = form.get("model", "wan2.6-image")
    prompt      = form.get("prompt", "")
    neg_prompt  = form.get("negative_prompt", "")
    size        = form.get("size", "1024*1024")
    watermark   = str(form.get("watermark", "false")).lower() in ("true", "1", "yes")
    seed_str    = str(form.get("seed", ""))
    seed        = int(seed_str) if seed_str.strip() else None
    try:
        ref_strength = float(form.get("ref_strength", "0.5"))
    except ValueError:
        ref_strength = 0.5

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Read and optionally resize reference images in memory
    image_files: list[tuple[str, bytes, str]] = []
    for i in range(1, 10):
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

    try:
        form_data = {"model": model, "prompt": prompt, "size": size, "n": "1",
                     "ref_strength": str(ref_strength)}
        if neg_prompt:
            form_data["negative_prompt"] = neg_prompt
        if watermark:
            form_data["watermark"] = "true"
        if seed is not None:
            form_data["seed"] = str(seed)

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
            images = []
            for item in rj.get("data", []):
                url = item.get("url")
                if url:
                    images.append({"url": url, "local_path": await _async_download_image(url)})
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

    w, h = _res_to_wh(data.resolution)
    payload: dict = {"model": data.model, "prompt": data.prompt,
                     "duration": data.duration, "width": w, "height": h}
    meta: dict = {}
    if data.negative_prompt: meta["negative_prompt"] = data.negative_prompt
    if data.audio:           meta["audio"] = True
    if data.prompt_extend:   meta["prompt_extend"] = True
    if data.watermark:       meta["watermark"] = True
    if data.seed is not None: meta["seed"] = data.seed
    if data.ratio:           meta["ratio"] = data.ratio
    if meta: payload["metadata"] = meta

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{NENAI_V1}/videos",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"},
                                     json=payload)
            return _handle_video_create_response(resp, data.model)
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

    ref_files = form.getlist("reference_files")
    if not ref_files or not hasattr(ref_files[0], "filename"):
        return JSONResponse(status_code=400, content={"error": "At least one reference file is required"})

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    media_arr: list = []
    for f in ref_files:
        if not hasattr(f, "filename") or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        fb = await f.read()
        mime = "video/mp4" if ext in VIDEO_EXTS else "image/png"
        media_type = "reference_video" if ext in VIDEO_EXTS else "reference_image"
        media_arr.append({"url": f"data:{mime};base64,{base64.b64encode(fb).decode()}", "type": media_type})

    if not media_arr:
        return JSONResponse(status_code=400, content={"error": "At least one reference file is required"})

    w, h = _res_to_wh(resolution)
    meta: dict = {}
    if prompt_extend: meta["prompt_extend"] = True
    if watermark:     meta["watermark"] = True
    if seed is not None: meta["seed"] = seed
    if ratio:         meta["ratio"] = ratio

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

# ─── API: Video Status ────────────────────────────────────────────
@app.get("/api/video/status/{task_id}")
async def video_status(task_id: str, api_key: str = Depends(get_api_key)):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{NENAI_V1}/videos/{task_id}",
                                    headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            rj = resp.json()

        status = rj.get("status", "pending").upper()
        # 平台可能回傳 completed/success，統一成前端認得的 SUCCEEDED
        if status in ("COMPLETED", "SUCCESS"):
            status = "SUCCEEDED"
        result: dict = {"task_id": task_id, "status": status}

        if status == "SUCCEEDED":
            # New-api may put the video URL in several places
            video_url = rj.get("url")
            if not video_url and isinstance(rj.get("data"), list) and rj["data"]:
                video_url = (rj["data"][0] or {}).get("url")
            if not video_url:
                md = rj.get("metadata") or {}
                video_url = md.get("video_url") or md.get("url")
            if video_url:
                local = await _async_download_video(video_url)
                result["local_path"] = local if local else video_url
                result["video_url"] = video_url
            else:
                # Fallback: try /content which may redirect to the actual file
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as c2:
                    cr = await c2.get(f"{NENAI_V1}/videos/{task_id}/content",
                                      headers={"Authorization": f"Bearer {api_key}"})
                    loc = cr.headers.get("location") or cr.headers.get("Location")
                    if loc:
                        local = await _async_download_video(loc)
                        result["local_path"] = local if local else loc
                        result["video_url"] = loc
        elif status == "FAILED":
            result["error_message"] = rj.get("error", {}).get("message", "Unknown error")

        return result
    except HTTPException:
        raise
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
        fp = OUTPUT_IMG_DIR / f"img_{ts}_{uuid.uuid4().hex[:6]}.png"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=30)
            if r.status_code == 200:
                fp.write_bytes(r.content)
                return f"/outputs/images/{fp.name}"
    except Exception as e:
        print(f"Image download error: {e}")
    return None

async def _async_download_video(url: str) -> Optional[str]:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = OUTPUT_VID_DIR / f"vid_{ts}_{uuid.uuid4().hex[:6]}.mp4"
        async with httpx.AsyncClient() as client:
            async with client.stream('GET', url, timeout=120) as r:
                if r.status_code == 200:
                    with open(fp, "wb") as f:
                        async for chunk in r.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
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

