"""
Alibaba Cloud AI Model Testing Platform
Flask Backend - API Key per-user authentication
"""
import os, sys, json, time, uuid, functools, mimetypes
from PIL import Image as PILImage
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template, send_from_directory, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message
from dashscope import VideoSynthesis
from dashscope.audio.tts import SpeechSynthesizer as TTSv1
from dashscope.audio.asr import Recognition

_INTL_WS_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"
import requests as http_requests

# ─── App Setup ────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB for video uploads
CORS(app)

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "上傳檔案過大，上限 200MB"}), 413

BASE_URL_COMPATIBLE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_HTTP_URL  = "https://dashscope-intl.aliyuncs.com/api/v1"
CUSTOM_HEADERS = {"X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'}

# 設定 dashscope 全域 base URL（只需設定一次，不可當 call() 的 kwarg 傳入）
dashscope.base_http_api_url = DASHSCOPE_HTTP_URL

UPLOAD_DIR      = Path(__file__).parent / "static" / "uploads"
OUTPUT_IMG_DIR  = Path(__file__).parent / "outputs" / "images"
OUTPUT_VID_DIR  = Path(__file__).parent / "outputs" / "videos"
for d in (UPLOAD_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ─── Model Registry ───────────────────────────────────────────
# sizes: 支援的尺寸清單；max_n: 最大生成張數；audio: 支援配音；min/max_dur: 影片時長範圍
MODELS = {
    "text": [
        # ── 旗艦 ──────────────────────────────────────────────────
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
def _extract_api_key():
    """Extract DashScope API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.args.get("api_key", "").strip()

def _validate_key(api_key: str) -> bool:
    """Quick validation: DashScope keys start with sk- and are long enough."""
    return bool(api_key and api_key.startswith("sk-") and len(api_key) > 20)


def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = _extract_api_key()
        if not _validate_key(api_key):
            return jsonify({"error": "Unauthorized - invalid API key"}), 401
        return f(*args, api_key=api_key, **kwargs)
    return decorated


# ─── Pages ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["POST"])
def login():
    """Validate DashScope API key by calling the models list endpoint."""
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()

    if not _validate_key(api_key):
        return jsonify({"success": False, "message": "API Key 格式有誤，須以 sk- 開頭"}), 400

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
        return jsonify({"success": True})
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err or "invalid" in err.lower():
            return jsonify({"success": False, "message": "API Key 無效或權限不足"}), 401
        # Other errors (rate limit etc.) - key is likely valid
        return jsonify({"success": True})


# ─── API: Models ──────────────────────────────────────────────────
@app.route("/api/models")
@require_auth
def get_models(api_key):
    return jsonify({**MODELS, "tts_voices": TTS_VOICES})


# ─── API: Text Generation (SSE Streaming) ─────────────────────────
@app.route("/api/text/generate", methods=["POST"])
@require_auth
def text_generate(api_key):
    data = request.get_json()
    model              = data.get("model", "qwen3.5-flash")
    prompt             = data.get("prompt", "")
    system_prompt      = data.get("system_prompt", "You are a helpful assistant.")
    temperature        = data.get("temperature", 0.7)
    top_p              = data.get("top_p", 0.8)
    top_k              = data.get("top_k", None)       # None = not sent
    max_tokens         = data.get("max_tokens", 4096)
    presence_penalty   = data.get("presence_penalty", 0)
    frequency_penalty  = data.get("frequency_penalty", 0)
    seed               = data.get("seed", None)         # None = random
    stop               = data.get("stop", []) or []
    use_stream         = data.get("stream", True)
    enable_thinking    = data.get("enable_thinking", False)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    extra_body = {}
    if enable_thinking:
        extra_body["enable_thinking"] = True

    create_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stream=use_stream,
        extra_headers=CUSTOM_HEADERS,
        extra_body=extra_body or None,
    )
    if top_k is not None and int(top_k) > 0:
        create_kwargs["extra_body"] = {**(extra_body or {}), "top_k": int(top_k)}
    if seed is not None:
        create_kwargs["seed"] = int(seed)
    if stop:
        create_kwargs["stop"] = stop[:4]

    if not use_stream:
        # 非串流：直接回傳 JSON
        try:
            user_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            resp = user_client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            return jsonify({"content": content, "done": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def generate():
        try:
            user_client = OpenAI(api_key=api_key, base_url=BASE_URL_COMPATIBLE)
            stream = user_client.chat.completions.create(**create_kwargs)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── API: Image Generate (T2I) ────────────────────────────────────
@app.route("/api/image/generate", methods=["POST"])
@require_auth
def image_generate(api_key):
    data = request.get_json()
    model          = data.get("model", "z-image-turbo")
    prompt         = data.get("prompt", "")
    negative_prompt= data.get("negative_prompt", "")
    size           = data.get("size", "1024*1024")
    n              = int(data.get("n", 1))
    prompt_extend  = data.get("prompt_extend", False)
    watermark      = data.get("watermark", False)
    seed           = data.get("seed", None)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        call_kwargs = dict(
            model=model,
            api_key=api_key,
            messages=[Message(role="user", content=[{"text": prompt}])],
            negative_prompt=negative_prompt or None,
            prompt_extend=prompt_extend,
            watermark=watermark,
            n=n,
            size=size,
            headers=CUSTOM_HEADERS,
        )
        if seed is not None:
            call_kwargs["seed"] = int(seed)
        rsp = ImageGeneration.call(**call_kwargs)
        return _handle_image_response(rsp, model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Image Edit (I2I) ────────────────────────────────────────
@app.route("/api/image/edit", methods=["POST"])
@require_auth
def image_edit(api_key):
    model          = request.form.get("model", "wan2.6-image")
    prompt         = request.form.get("prompt", "")
    negative_prompt= request.form.get("negative_prompt", "")
    size           = request.form.get("size", "1024*1024")
    watermark_str  = request.form.get("watermark", "false")
    watermark      = watermark_str.lower() in ("true", "1", "yes")
    ref_strength_str = request.form.get("ref_strength", "0.5")
    try:
        ref_strength = float(ref_strength_str)
    except ValueError:
        ref_strength = 0.5
    seed_str = request.form.get("seed", "")
    seed = int(seed_str) if seed_str.strip() else None

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    image_urls = []
    for i in range(1, 10):
        f = request.files.get(f"image_{i}")
        if not f:
            break
        ext = Path(f.filename).suffix or ".png"
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        f.save(fp)
        image_urls.append(f"file://{fp.resolve()}")

    if not image_urls:
        return jsonify({"error": "至少需要一張參考圖片"}), 400

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
        # ref_strength 僅部分模型支援，傳入不會影響不支援的模型
        call_kwargs["ref_strength"] = ref_strength
        rsp = ImageGeneration.call(**call_kwargs)
        return _handle_image_response(rsp, model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_HAPPYHORSE_MODELS = {"happyhorse-1.0-t2v", "happyhorse-1.0-i2v",
                      "happyhorse-1.0-r2v", "happyhorse-1.0-video-edit"}

_SIZE_MAP = {"480P": "854*480", "720P": "1280*720", "1080P": "1920*1080"}


def _apply_resolution(kwargs: dict, model: str, resolution: str, ratio: str = "") -> None:
    """HappyHorse uses resolution+ratio; Wan models use size."""
    if model in _HAPPYHORSE_MODELS:
        kwargs["resolution"] = resolution
        if ratio:
            kwargs["ratio"] = ratio
    else:
        kwargs["size"] = _SIZE_MAP.get(resolution, "1280*720")


# ─── API: Video T2V ───────────────────────────────────────────────
@app.route("/api/video/t2v", methods=["POST"])
@require_auth
def video_t2v(api_key):
    data = request.get_json()
    model          = data.get("model", "wan2.6-t2v")
    prompt         = data.get("prompt", "")
    negative_prompt= data.get("negative_prompt", "")
    resolution     = data.get("resolution", "720P")
    ratio          = data.get("ratio", "16:9")
    duration       = data.get("duration", 5)
    enable_audio   = data.get("audio", False)
    prompt_extend  = data.get("prompt_extend", False)
    watermark      = data.get("watermark", False)
    seed           = data.get("seed", None)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    kwargs = dict(
        model=model,
        prompt=prompt,
        duration=duration,
        prompt_extend=prompt_extend,
        watermark=watermark,
        api_key=api_key,
        headers=CUSTOM_HEADERS,
    )
    _apply_resolution(kwargs, model, resolution, ratio)
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if enable_audio:
        kwargs["audio"] = True
    if seed is not None:
        kwargs["seed"] = int(seed)

    try:
        rsp = VideoSynthesis.async_call(**kwargs)
        return _handle_video_async_response(rsp, model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Video I2V ───────────────────────────────────────────────
@app.route("/api/video/i2v", methods=["POST"])
@require_auth
def video_i2v(api_key):
    model          = request.form.get("model", "wan2.7-i2v")
    prompt         = request.form.get("prompt", "")
    negative_prompt= request.form.get("negative_prompt", "")
    resolution     = request.form.get("resolution", "720P")
    ratio          = request.form.get("ratio", "16:9")
    duration       = int(request.form.get("duration", 5))
    i2v_mode       = request.form.get("i2v_mode", "first_frame")
    prompt_extend_str = request.form.get("prompt_extend", "false")
    prompt_extend  = prompt_extend_str.lower() in ("true", "1", "yes")
    watermark_str  = request.form.get("watermark", "false")
    watermark      = watermark_str.lower() in ("true", "1", "yes")
    seed_str       = request.form.get("seed", "")
    seed           = int(seed_str) if seed_str.strip() else None

    def _save_file(file_obj, default_ext=".png", is_image=False):
        ext = Path(file_obj.filename).suffix or default_ext
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        if is_image:
            img = PILImage.open(file_obj.stream)
            w, h = img.size
            # wan2.7-i2v 要求圖片最小 240x240，不足則等比例放大
            if w < 240 or h < 240:
                scale = max(240 / w, 240 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
            # JPEG 不支援 Alpha channel，強制轉 RGB
            if ext.lower() in (".jpg", ".jpeg") and img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(fp)
        else:
            file_obj.save(fp)
        return f"file://{fp.resolve()}"

    # Build media array based on mode
    media = []
    first_frame_file = request.files.get("first_frame") or request.files.get("image")
    last_frame_file  = request.files.get("last_frame")
    audio_file       = request.files.get("driving_audio")
    clip_file        = request.files.get("first_clip")

    if i2v_mode == "first_clip":
        if not clip_file:
            return jsonify({"error": "first_clip 模式需要上傳影片片段"}), 400
        media.append({"url": _save_file(clip_file, ".mp4"), "type": "first_clip"})
        if last_frame_file:
            media.append({"url": _save_file(last_frame_file, ".png", is_image=True), "type": "last_frame"})
    else:
        if not first_frame_file:
            return jsonify({"error": "I2V 需要上傳首幀圖片"}), 400
        media.append({"url": _save_file(first_frame_file, ".png", is_image=True), "type": "first_frame"})
        if last_frame_file:
            media.append({"url": _save_file(last_frame_file, ".png", is_image=True), "type": "last_frame"})
        if audio_file:
            media.append({"url": _save_file(audio_file, ".mp3"), "type": "driving_audio"})

    try:
        kwargs = dict(
            model=model,
            media=media,
            prompt=prompt,
            duration=duration,
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
        return jsonify({"error": str(e)}), 500


# ─── API: Video Edit (wan2.7-videoedit) ──────────────────────────
@app.route("/api/video/vedit", methods=["POST"])
@require_auth
def video_vedit(api_key):
    model          = request.form.get("model", "wan2.7-videoedit")
    prompt         = request.form.get("prompt", "")
    negative_prompt= request.form.get("negative_prompt", "")
    resolution     = request.form.get("resolution", "1080P")
    ratio          = request.form.get("ratio", "")           # 留空 = 跟隨輸入影片
    duration_str   = request.form.get("duration", "0")
    duration       = int(duration_str) if duration_str.strip() else 0
    audio_setting  = request.form.get("audio_setting", "auto")
    prompt_extend_str = request.form.get("prompt_extend", "true")
    prompt_extend  = prompt_extend_str.lower() in ("true", "1", "yes")
    watermark_str  = request.form.get("watermark", "false")
    watermark      = watermark_str.lower() in ("true", "1", "yes")
    seed_str       = request.form.get("seed", "")
    seed           = int(seed_str) if seed_str.strip() else None

    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "影片檔案為必填"}), 400

    ext = Path(video_file.filename).suffix or ".mp4"
    vp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    video_file.save(vp)

    # HappyHorse Video Edit 支援最多 5 張參考圖，Wan 最多 3 張
    max_refs = 5 if model in _HAPPYHORSE_MODELS else 3
    media = [{"url": f"file://{vp.resolve()}", "type": "video"}]
    for i in range(1, max_refs + 1):
        ref = request.files.get(f"reference_image_{i}")
        if ref:
            rext = Path(ref.filename).suffix or ".png"
            rp = UPLOAD_DIR / f"{uuid.uuid4().hex}{rext}"
            ref.save(rp)
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
        return jsonify({"error": str(e)}), 500


# ─── API: Video R2V ───────────────────────────────────────────────
@app.route("/api/video/r2v", methods=["POST"])
@require_auth
def video_r2v(api_key):
    model     = request.form.get("model", "wan2.6-r2v")
    prompt    = request.form.get("prompt", "")
    resolution= request.form.get("resolution", "720P")
    ratio     = request.form.get("ratio", "16:9")
    duration  = int(request.form.get("duration", 5))
    prompt_extend_str = request.form.get("prompt_extend", "false")
    prompt_extend = prompt_extend_str.lower() in ("true", "1", "yes")
    watermark_str = request.form.get("watermark", "false")
    watermark = watermark_str.lower() in ("true", "1", "yes")
    seed_str  = request.form.get("seed", "")
    seed      = int(seed_str) if seed_str.strip() else None

    files = request.files.getlist("reference_files")
    if not files:
        return jsonify({"error": "At least one reference file is required"}), 400

    # wan2.x-r2v 需要 input.media，使用 SDK 的 media 參數（SDK 會自動處理 file:// 上傳）
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    media = []
    for f in files:
        ext = Path(f.filename).suffix or ".png"
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        f.save(fp)
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
        # 修正：reference_urls → media（wan2.x-r2v API 需要 input.media，非 input.reference_urls）
        rsp = VideoSynthesis.async_call(**r2v_kwargs)
        return _handle_video_async_response(rsp, model)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── API: Video Status ────────────────────────────────────────────
@app.route("/api/video/status/<task_id>")
@require_auth
def video_status(task_id, api_key):
    try:
        dashscope.api_key = api_key
        rsp = VideoSynthesis.fetch(task_id)
        status = getattr(rsp.output, "task_status", "UNKNOWN")
        result = {"task_id": task_id, "status": status}
        if status == "SUCCEEDED":
            video_url = getattr(rsp.output, "video_url", "")
            if video_url:
                result["local_path"] = _download_video(video_url)
                result["video_url"] = video_url
        elif status == "FAILED":
            result["error_message"] = getattr(rsp.output, "message", "Unknown")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Static Outputs ───────────────────────────────────────────────
@app.route("/outputs/images/<filename>")
def serve_image(filename):
    return send_from_directory(OUTPUT_IMG_DIR, filename)

@app.route("/outputs/videos/<filename>")
def serve_video(filename):
    return send_from_directory(OUTPUT_VID_DIR, filename)


# ─── Helpers ──────────────────────────────────────────────────────
def _handle_image_response(rsp, model):
    if rsp.status_code == 200:
        images = []
        output = rsp.output
        # 格式1: choices[].message.content[].image
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
        # 格式2: results[].url
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
            return jsonify({"error": f"No images in response. output={output}"}), 500
        return jsonify({"success": True, "images": images, "model": model})
    return jsonify({"error": f"Generation failed ({rsp.status_code}): {rsp.message}", "code": rsp.code}), 500


def _handle_video_async_response(rsp, model):
    """處理 VideoSynthesis.async_call() 的回應（立即返回 task_id）"""
    from http import HTTPStatus
    if rsp.status_code == HTTPStatus.OK:
        task_id = getattr(rsp.output, "task_id", "")
        task_status = getattr(rsp.output, "task_status", "PENDING")
        if task_id:
            return jsonify({"success": True, "task_id": task_id, "status": task_status, "model": model})
        # 若已同步完成（極少見）
        if task_status == "SUCCEEDED":
            video_url = getattr(rsp.output, "video_url", "")
            return jsonify({"success": True, "status": "SUCCEEDED", "video_url": video_url,
                            "local_path": _download_video(video_url), "model": model})
        return jsonify({"error": f"No task_id in response. status={task_status}"}), 500
    return jsonify({"error": f"API error ({rsp.status_code}): {rsp.message}", "code": rsp.code}), 500


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
OUTPUT_AUDIO_DIR = Path(__file__).parent / "outputs" / "audio"
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

_ASR_FMT = {".wav": "wav", ".mp3": "mp3", ".m4a": "m4a",
            ".flac": "flac", ".ogg": "ogg", ".opus": "opus"}

@app.route("/api/voice/asr", methods=["POST"])
@require_auth
def voice_asr(api_key):
    model = request.form.get("model", "paraformer-v2")
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "請上傳音訊檔案"}), 400

    ext = Path(audio_file.filename).suffix.lower() or ".wav"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    audio_file.save(tmp_path)

    try:
        audio_fmt = _ASR_FMT.get(ext, "wav")
        dashscope.api_key = api_key
        recognizer = Recognition(
            model=model,
            callback=None,
            format=audio_fmt,
            sample_rate=16000,
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
            return jsonify({"success": True, "text": text, "model": model})
        return jsonify({"error": f"ASR 失敗: {getattr(rsp, 'message', str(rsp))}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── API: Voice TTS ───────────────────────────────────────────────
@app.route("/api/voice/tts", methods=["POST"])
@require_auth
def voice_tts(api_key):
    data  = request.get_json()
    model = data.get("model", "qwen-tts")
    voice = data.get("voice", "Cherry")
    text  = data.get("text", "").strip()
    fmt   = data.get("format", "mp3")

    if not text:
        return jsonify({"error": "請輸入合成文字"}), 400
    if len(text) > 4000:
        return jsonify({"error": "文字長度不可超過 4000 字"}), 400

    try:
        dashscope.api_key = api_key
        rsp = TTSv1.call(
            model=model,
            text=text,
            voice=voice,
            format=fmt,
            sample_rate=22050,
        )
        audio_data = rsp.get_audio_data()
        if audio_data:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{ts}_{uuid.uuid4().hex[:6]}.{fmt}"
            fp = OUTPUT_AUDIO_DIR / filename
            fp.write_bytes(audio_data)
            return jsonify({"success": True, "audio_url": f"/outputs/audio/{filename}",
                            "model": model, "voice": voice})
        response = rsp.get_response()
        msg = getattr(response, "message", None) or getattr(response, "text", None) or str(response)
        return jsonify({"error": f"TTS 失敗: {msg}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(OUTPUT_AUDIO_DIR, filename)


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Alibaba Cloud AI Model Testing Platform")
    print("   Auth: DashScope API Key (sk-...)")
    print("   Endpoint: http://localhost:5050")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
