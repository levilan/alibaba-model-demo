"""MCP endpoint（/mcp）的單元測試——全部不打上游、不產生費用。

涵蓋：JSON-RPC 協定面（initialize / tools/list / notification / batch 拒絕）、
無 key 的 tools/call、MODELS 驅動的參數預檢（錯誤必附 valid_values）、
以及設計文件要求的「約束欄位白名單」守門測試。

會打上游的路徑（實際生成、pricing）不在單元測試範圍——那是上線冒煙的事。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module
from app import app, MODELS, _MCP_CONSTRAINT_FIELDS, _MCP_TOOLS


def _rpc(body: dict, headers: dict | None = None) -> httpx.Response:
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post("/mcp", json=body, headers=headers or {})
    return asyncio.run(go())


def _prime_key(key: str, valid: bool = True) -> None:
    """把 key 有效性寫進快取，讓測試不出網路。"""
    import hashlib
    app_module._MCP_KEY_VALID_CACHE[hashlib.sha256(key.encode()).hexdigest()] = (valid, time.time())


def _call_tool(name: str, arguments: dict, key: str | None = "sk-test") -> dict:
    if key:
        _prime_key(key, True)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}}, headers)
    assert r.status_code == 200
    result = r.json()["result"]
    payload = json.loads(result["content"][0]["text"])
    return {"isError": result.get("isError", False), **payload}


# ── 協定面 ──────────────────────────────────────────────────────────

def test_initialize_handshake():
    r = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18",
                         "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
    res = r.json()["result"]
    assert res["protocolVersion"] == "2025-06-18"
    assert res["serverInfo"]["name"] == "nenai-playground"
    assert "tools" in res["capabilities"]


def test_tools_list_has_phase1_tools():
    r = _rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert names == {"nenai_list_models", "nenai_generate_image",
                     "nenai_generate_video", "nenai_task_status",
                     "nenai_edit_image", "nenai_tts", "nenai_asr"}
    # 每個工具都要有 inputSchema（客戶端靠這個做參數 UI 與驗證）
    for t in r.json()["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_notification_returns_202():
    r = _rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202


def test_batch_rejected():
    r = _rpc([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])  # type: ignore[arg-type]
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600


def test_get_mcp_is_405():
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get("/mcp")
    assert asyncio.run(go()).status_code == 405


def test_unknown_method():
    r = _rpc({"jsonrpc": "2.0", "id": 9, "method": "resources/list"})
    assert r.json()["error"]["code"] == -32601


# ── tools/call 驗證路徑（不打上游）────────────────────────────────

def test_call_without_key_is_guided_error():
    out = _call_tool("nenai_generate_image", {"model": "z-image-turbo", "prompt": "x"}, key=None)
    assert out["isError"] and out["error"] == "missing_api_key"
    assert "Authorization" in out["message"]


def test_call_with_invalid_key_is_rejected():
    """假 key 要被擋（2026-08-25 文檔站回報：假 key 可拿模型目錄＋單價，Levi 裁示收緊）。"""
    _prime_key("sk-definitely-invalid", valid=False)
    r = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "nenai_list_models", "arguments": {}}},
             {"Authorization": "Bearer sk-definitely-invalid"})
    result = r.json()["result"]
    payload = json.loads(result["content"][0]["text"])
    assert result["isError"] and payload["error"] == "invalid_api_key"


def test_generate_image_unknown_model_lists_valid():
    out = _call_tool("nenai_generate_image", {"model": "no-such-model", "prompt": "x"})
    assert out["isError"] and out["field"] == "model"
    assert "z-image-turbo" in out["valid_values"]


def test_generate_image_bad_size_lists_valid():
    m = next(m for m in MODELS["image"]
             if m.get("type") == "t2i" and m.get("sizes") and not m.get("custom_size"))
    out = _call_tool("nenai_generate_image",
                     {"model": m["id"], "prompt": "x", "size": "1x1"})
    assert out["isError"] and out["field"] == "size"
    assert out["valid_values"] == m["sizes"]


def test_generate_image_n_over_limit():
    m = next(m for m in MODELS["image"] if m.get("type") == "t2i")
    out = _call_tool("nenai_generate_image",
                     {"model": m["id"], "prompt": "x", "n": 99})
    assert out["isError"] and out["field"] == "n"


def test_generate_video_bad_duration_lists_valid_steps():
    veo = next(m for m in MODELS["video"]
               if m.get("type") == "t2v" and m.get("dur_step", 1) > 1)
    out = _call_tool("nenai_generate_video",
                     {"model": veo["id"], "prompt": "x", "duration": veo["min_dur"] + 1})
    assert out["isError"] and out["field"] == "duration"
    assert veo["min_dur"] in out["valid_values"]
    assert veo["min_dur"] + 1 not in out["valid_values"]


def test_generate_video_i2v_model_split():
    # 只有 t2v 沒有 i2v 的模型，帶 image_url 要被擋下並列出 i2v 合法清單
    t2v_only = {m["id"] for m in MODELS["video"] if m.get("type") == "t2v"} - \
               {m["id"] for m in MODELS["video"] if m.get("type") == "i2v"}
    if not t2v_only:
        pytest.skip("目前所有 t2v 模型都有 i2v 對應")
    out = _call_tool("nenai_generate_video",
                     {"model": sorted(t2v_only)[0], "prompt": "x",
                      "image_url": "data:image/png;base64,aGk="})
    assert out["isError"] and out["field"] == "model"


def test_generate_video_bad_image_url_scheme():
    m = next(m for m in MODELS["video"] if m.get("type") == "i2v")
    out = _call_tool("nenai_generate_video",
                     {"model": m["id"], "prompt": "x", "image_url": "ftp://x/y.png"})
    assert out["isError"] and out["field"] == "image_url"


def test_edit_image_t2i_only_model_rejected():
    # 只有 t2i 沒有 i2i 的模型不能拿來編輯，錯誤要列出 i2i 合法清單
    i2i = {m["id"] for m in MODELS["image"] if m.get("type") == "i2i"}
    t2i_only = {m["id"] for m in MODELS["image"] if m.get("type") == "t2i"} - i2i
    if not t2i_only:
        pytest.skip("所有 t2i 模型都有 i2i 對應")
    out = _call_tool("nenai_edit_image",
                     {"model": sorted(t2i_only)[0], "prompt": "x",
                      "images": ["data:image/png;base64,aGk="]})
    assert out["isError"] and out["field"] == "model"
    assert set(out["valid_values"]) == i2i


def test_edit_image_too_many_refs():
    m = next(m for m in MODELS["image"] if m.get("type") == "i2i" and m.get("max_ref"))
    imgs = ["data:image/png;base64,aGk="] * (m["max_ref"] + 1)
    out = _call_tool("nenai_edit_image", {"model": m["id"], "prompt": "x", "images": imgs})
    assert out["isError"] and out["field"] == "images"
    assert str(m["max_ref"]) in out["message"]


def test_tts_bad_voice_lists_valid_ids():
    m = next(m for m in MODELS["voice"]["tts"] if m.get("voices"))
    out = _call_tool("nenai_tts", {"model": m["id"], "text": "hi", "voice": "no-such-voice"})
    assert out["isError"] and out["field"] == "voice"
    assert out["valid_values"] == [v["id"] for v in m["voices"]]


def test_asr_unknown_model():
    out = _call_tool("nenai_asr", {"model": "not-asr", "audio_url": "data:audio/wav;base64,aGk="})
    assert out["isError"] and out["field"] == "model"
    assert all(m["id"] in out["valid_values"] for m in MODELS["voice"]["asr"])


def test_unknown_tool():
    r = _rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "nope", "arguments": {}}},
             {"Authorization": "Bearer sk-test"})
    assert r.json()["error"]["code"] == -32602


# ── list_models（monkeypatch 掉兩個上游快取，不出網路）──────────────

def test_list_models_offline(monkeypatch):
    monkeypatch.setitem(app_module._UPSTREAM_IDS_CACHE, "ids", set())
    monkeypatch.setitem(app_module._UPSTREAM_IDS_CACHE, "ts", time.time())
    monkeypatch.setitem(app_module._PRICING_CACHE, "data", {})
    monkeypatch.setitem(app_module._PRICING_CACHE, "ts", time.time())
    out = _call_tool("nenai_list_models", {"category": "image"})
    assert not out["isError"]
    ids = {m["id"] for m in out["image"]}
    assert "z-image-turbo" in ids
    # 約束欄位要出現在輸出（agent 靠這個知道合法值）
    sized = next(m for m in out["image"] if m.get("sizes"))
    assert isinstance(sized["sizes"], list)


# ── 設計文件要求的守門：MODELS 出現 MCP 不認識的欄位就 fail ─────────

def test_models_fields_are_known_to_mcp():
    """新模型加了新約束欄位時，這條會 fail——強迫當下決定要不要進 MCP 白名單
    （_MCP_CONSTRAINT_FIELDS），而不是等 agent 端出錯才發現。詳見
    docs/mcp-tool-design.md 第七節。"""
    non_constraint = {
        # 識別與文案
        "id", "name", "desc", "group", "vendor",
        # 計價／UI 專用，MCP 由 pricing 與 description 另行涵蓋
        "input_rate", "output_rate", "duration_hint", "default_voice", "voices",
        # 行為開關（轉譯層內部使用，agent 不需要知道）
        "fusion_edit", "no_negative_prompt", "no_prompt_extend", "no_ref_strength",
        "no_size", "supports_gpt_params", "supports_sequential",
        "sequential_max_size", "image_input", "audio_only", "turn_modes",
        "reasoning_effort",
    }
    known = set(_MCP_CONSTRAINT_FIELDS) | non_constraint
    unknown: dict[str, set] = {}
    for cat in ("image", "video"):
        for m in MODELS[cat]:
            extra = set(m) - known
            if extra:
                unknown.setdefault(m["id"], set()).update(extra)
    assert not unknown, (
        f"MODELS 出現 MCP 映射表不認識的欄位：{unknown}——"
        "請決定它要進 _MCP_CONSTRAINT_FIELDS（agent 可見約束）"
        "還是 test_mcp.py 的 non_constraint（內部欄位）")
