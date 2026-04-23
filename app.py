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
        {"id": "qwen3-max",    "name": "Qwen3 Max",       "group": "旗艦",   "desc": "最強推理，262K context", "thinking": True},
        {"id": "qwen3.6-plus", "name": "Qwen3.6 Plus",    "group": "均衡",   "desc": "1M context，性價比最佳", "thinking": True},
        {"id": "qwen3.5-plus", "name": "Qwen3.5 Plus",    "group": "均衡",   "desc": "前代均衡模型",           "thinking": True},
        {"id": "qwen3.5-flash","name": "Qwen3.5 Flash",   "group": "極速",   "desc": "速度快、成本低",         "thinking": True},
        {"id": "qwen-flash",   "name": "Qwen Flash",      "group": "極速",   "desc": "前代極速模型",           "thinking": True},
        {"id": "deepseek-v3.2","name": "DeepSeek V3.2",   "group": "第三方", "desc": "深度推理（國際版可用）",  "thinking": False},
    ],
    "image": [
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
            "id": "wan2.6-t2i", "name": "萬相 2.6 T2I", "group": "萬相文生圖",
            "desc": "自由選尺寸", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","960*1280","1280*960","960*1696","1696*960"],
        },
        {
            "id": "z-image-turbo", "name": "Z-Image Turbo", "group": "Z-Image",
            "desc": "輕量級快速生成", "type": "t2i", "max_n": 4,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
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
        {
            "id": "qwen-image-edit-max", "name": "千問圖像編輯 Max", "group": "千問圖像編輯",
            "desc": "複雜圖文編輯", "type": "i2i", "max_n": 1,
            "sizes": ["1024*1024","1280*720","720*1280","1024*768","768*1024"],
        },
    ],
    "video": [
        {"id": "wan2.7-t2v", "name": "萬相 2.7 T2V", "group": "文生影片",   "desc": "多鏡頭、自動配音", "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 10},
        {"id": "wan2.6-t2v", "name": "萬相 2.6 T2V", "group": "文生影片",   "desc": "前代文生影片",     "type": "t2v",   "audio": False, "min_dur": 3, "max_dur": 10},
        {"id": "wan2.7-i2v", "name": "萬相 2.7 I2V", "group": "圖生影片",   "desc": "首幀/首尾幀/配音/影片延伸", "type": "i2v", "audio": False, "min_dur": 2, "max_dur": 15},
        {"id": "wan2.6-i2v", "name": "萬相 2.6 I2V", "group": "圖生影片",   "desc": "前代圖生影片",     "type": "i2v",   "audio": False, "min_dur": 3, "max_dur": 10},
        {"id": "wan2.7-r2v", "name": "萬相 2.7 R2V", "group": "參考生影片", "desc": "角色形象參考",     "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 10},
        {"id": "wan2.6-r2v", "name": "萬相 2.6 R2V", "group": "參考生影片", "desc": "前代參考生影片",   "type": "r2v",   "audio": False, "min_dur": 3, "max_dur": 10},
        {"id": "wan2.7-videoedit", "name": "萬相 2.7 視頻編輯", "group": "萬相視頻編輯",
         "desc": "文字/參考圖驅動編輯", "type": "vedit", "audio": False, "min_dur": 0, "max_dur": 10},
    ]
}


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
    return jsonify(MODELS)


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
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Image file is required"}), 400

    ext = Path(image_file.filename).suffix or ".png"
    fp = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    image_file.save(fp)
    image_url = f"file://{fp.resolve()}"

    try:
        call_kwargs = dict(
            model=model,
            api_key=api_key,
            messages=[Message(role="user", content=[{"text": prompt}, {"image": image_url}])],
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


# ─── API: Video T2V ───────────────────────────────────────────────
@app.route("/api/video/t2v", methods=["POST"])
@require_auth
def video_t2v(api_key):
    data = request.get_json()
    model          = data.get("model", "wan2.6-t2v")
    prompt         = data.get("prompt", "")
    negative_prompt= data.get("negative_prompt", "")
    resolution     = data.get("resolution", "720P")
    duration       = data.get("duration", 5)
    enable_audio   = data.get("audio", False)
    prompt_extend  = data.get("prompt_extend", False)
    watermark      = data.get("watermark", False)
    seed           = data.get("seed", None)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    size_map = {"480P": "854*480", "720P": "1280*720", "1080P": "1920*1080"}
    kwargs = dict(
        model=model,
        prompt=prompt,
        size=size_map.get(resolution, "1280*720"),
        duration=duration,
        prompt_extend=prompt_extend,
        watermark=watermark,
        api_key=api_key,
        headers=CUSTOM_HEADERS,
    )
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if enable_audio:
        kwargs["audio"] = True
    if seed is not None:
        kwargs["seed"] = int(seed)

    try:
        # 修正：使用 async_call() 立即返回 task_id，call() 會內部等待完成導致卡住
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
            size={"480P": "854*480", "720P": "1280*720", "1080P": "1920*1080"}.get(resolution, "1280*720"),
            duration=duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
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

    media = [{"url": f"file://{vp.resolve()}", "type": "video"}]
    for i in range(1, 4):
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
            size={"720P": "1280*720", "1080P": "1920*1080"}.get(resolution, "1920*1080"),
            duration=duration,
            audio_setting=audio_setting,
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if ratio:
            kwargs["ratio"] = ratio
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
            size={"480P": "854*480", "720P": "1280*720", "1080P": "1920*1080"}.get(resolution, "1280*720"),
            duration=duration,
            prompt_extend=prompt_extend,
            watermark=watermark,
            api_key=api_key,
            headers=CUSTOM_HEADERS,
        )
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


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Alibaba Cloud AI Model Testing Platform")
    print("   Auth: DashScope API Key (sk-...)")
    print("   Endpoint: http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=True)
