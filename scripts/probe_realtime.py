#!/usr/bin/env python3
"""realtime（/v1/realtime WebSocket）模型探測——新增 realtime 模型時用這支，
不要每次重寫一次性腳本。HTTP 模型的探測用 probe_model.py，這支只管 WebSocket。

用法（--gateway / --key-file 與 probe_model.py 相同）：
    venv/bin/python scripts/probe_realtime.py <model-id> --test basic
    venv/bin/python scripts/probe_realtime.py <model-id> --test audio --wav q.wav
    venv/bin/python scripts/probe_realtime.py <model-id> --test image --image a.jpg --prompt 這是什麼顏色
    venv/bin/python scripts/probe_realtime.py <model-id> --test voices --voices Tina,Serena
    venv/bin/python scripts/probe_realtime.py <model-id> --test turn

設計原則（延續 probe_model.py，外加 realtime 專屬的兩條）：

1. **session.update 的 ack 不能當驗證。** 它對任何字串都回 session.updated——連
   `NoSuchVoiceZZZ` 都照收。要驗音色／參數必須真的 response.create 一次，看是收到
   音訊還是 error 事件。所以 voices 測試每個音色都會真的生成一小段音訊（有計費，
   但問題固定是「說「好」。」，量極小）。
2. **「沒收到 error」是弱證據。** turn 測試只能回報「送了沒被拒」，不等於該模式
   真的在運作——VAD 的行為要在瀏覽器上用真實語音驗。輸出會把這點標清楚。
3. 送出的事件形狀**刻意跟前端 static/js/app.js 完全一致**（input_audio_format
   'pcm16'、output 'pcm'、conversation.item.create + response.create……），這樣
   探測通過就等於前端那條路徑通。改前端的形狀時這裡要跟著改。
"""
from __future__ import annotations

import argparse
import array
import asyncio
import base64
import json
import os
import sys
import time
import wave
from pathlib import Path

import websockets

GATEWAYS = {
    "prod": "https://nen.com.tw",
    "test": "http://192.168.0.245",
}

IN_RATE = 16000   # 上行 PCM16
OUT_RATE = 24000  # 下行 PCM16


def _key(args) -> str:
    if args.key_file:
        return Path(args.key_file).read_text().strip()
    env = os.environ.get("NENAI_API_KEY", "").strip()
    if env:
        return env
    sys.exit("需要金鑰：用 --key-file PATH 或設定環境變數 NENAI_API_KEY")


def _ws_url(gateway: str, model: str) -> str:
    base = GATEWAYS[gateway].replace("https://", "wss://").replace("http://", "ws://")
    return f"{base}/v1/realtime?model={model}"


def _session_update(voice: str | None, modalities: list[str], turn_detection,
                    instructions: str = "你是友善的中文語音助理，回答簡潔自然。") -> dict:
    session = {
        "modalities": modalities,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm",
        "instructions": instructions,
        "turn_detection": turn_detection,
    }
    if voice:
        session["voice"] = voice
    return {"type": "session.update", "session": session}


def _text_turn(text: str) -> list[dict]:
    return [
        {"type": "conversation.item.create",
         "item": {"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": text}]}},
        {"type": "response.create"},
    ]


async def _collect(ws, deadline_s: float = 60.0) -> dict:
    """收事件直到 response.done / error / 逾時。回傳彙整結果。"""
    out = {"transcript": "", "text": "", "audio": bytearray(), "usage": None,
           "errors": [], "events": [], "first_audio_ms": None, "done": False}
    t0 = time.monotonic()
    while True:
        remain = deadline_s - (time.monotonic() - t0)
        if remain <= 0:
            out["errors"].append(f"逾時（{deadline_s}s 內沒收到 response.done）")
            return out
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remain)
        except asyncio.TimeoutError:
            out["errors"].append(f"逾時（{deadline_s}s 內沒收到 response.done）")
            return out
        except websockets.ConnectionClosed as e:
            out["errors"].append(f"連線被關閉：{e}")
            return out
        ev = json.loads(raw)
        t = ev.get("type", "?")
        if t not in out["events"]:
            out["events"].append(t)
        if t == "response.audio.delta":
            if out["first_audio_ms"] is None:
                out["first_audio_ms"] = round((time.monotonic() - t0) * 1000)
            out["audio"].extend(base64.b64decode(ev.get("delta", "")))
        elif t == "response.audio_transcript.delta":
            out["transcript"] += ev.get("delta", "")
        elif t == "response.text.delta":
            out["text"] += ev.get("delta", "")
        elif t == "response.done":
            out["usage"] = (ev.get("response") or {}).get("usage")
            out["done"] = True
            return out
        elif t == "error":
            out["errors"].append(json.dumps(ev.get("error", ev), ensure_ascii=False))
            return out


def _rms16(buf: bytes) -> int:
    # audioop 在 Python 3.13 被移除，這裡自己算 s16le 的 RMS
    a = array.array("h")
    a.frombytes(buf[: len(buf) // 2 * 2])
    return int((sum(x * x for x in a) / len(a)) ** 0.5) if a else 0


def _audio_stats(buf: bytes) -> str:
    if not buf:
        return "無音訊"
    rms = _rms16(buf)
    dur = len(buf) / 2 / OUT_RATE
    silent = "（⚠️ 疑似靜音）" if rms < 100 else ""
    return f"{len(buf)} bytes ≈ {dur:.1f}s@{OUT_RATE}Hz RMS={rms}{silent}"


def _report(label: str, r: dict, save_audio: str | None = None):
    print(f"── {label}")
    print(f"   事件：{', '.join(r['events']) or '（無）'}")
    if r["transcript"]:
        print(f"   逐字稿：{r['transcript']}")
    if r["text"]:
        print(f"   文字：{r['text']}")
    print(f"   音訊：{_audio_stats(bytes(r['audio']))}"
          + (f"　首包 {r['first_audio_ms']}ms" if r["first_audio_ms"] is not None else ""))
    if r["usage"] is not None:
        print(f"   usage：{json.dumps(r['usage'], ensure_ascii=False)}")
    for e in r["errors"]:
        print(f"   ❌ {e}")
    if save_audio and r["audio"]:
        with wave.open(save_audio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(OUT_RATE)
            w.writeframes(bytes(r["audio"]))
        print(f"   音訊已存：{save_audio}")


async def _connect(args, key):
    # 握手偶爾會逾時（實測掃音色掃到一半發生過），重試一次再放棄
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            ws = await websockets.connect(_ws_url(args.gateway, args.model),
                                          additional_headers={"Authorization": f"Bearer {key}"},
                                          max_size=16 * 1024 * 1024)
            break
        except TimeoutError:
            if attempt == 2:
                raise
            print("   （握手逾時，重試一次）")
            await asyncio.sleep(2)
    print(f"握手 {round((time.monotonic() - t0) * 1000)}ms → {_ws_url(args.gateway, args.model)}")
    return ws


async def test_basic(args, key):
    async with await _connect(args, key) as ws:
        await ws.send(json.dumps(_session_update(args.voice, ["text", "audio"], None)))
        for ev in _text_turn(args.prompt or "一加一等於幾？請只用一句話回答。"):
            await ws.send(json.dumps(ev))
        _report("文字輸入 → 語音+文字輸出", await _collect(ws), args.save_audio)


async def test_audio(args, key):
    if not args.wav:
        sys.exit("--test audio 需要 --wav（PCM16 16kHz mono 的 wav 檔）")
    with wave.open(args.wav, "rb") as w:
        assert w.getframerate() == IN_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            f"wav 必須是 {IN_RATE}Hz mono 16-bit，實際 {w.getframerate()}Hz {w.getnchannels()}ch {w.getsampwidth() * 8}bit"
        pcm = w.readframes(w.getnframes())
    async with await _connect(args, key) as ws:
        await ws.send(json.dumps(_session_update(args.voice, ["text", "audio"], None)))
        step = IN_RATE // 10 * 2  # 100ms 一包，貼近瀏覽器的行為
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + step]).decode()}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))
        _report(f"語音輸入（{len(pcm) / 2 / IN_RATE:.1f}s）→ 輸出", await _collect(ws), args.save_audio)


async def test_image(args, key):
    if not args.image:
        sys.exit("--test image 需要至少一個 --image（jpeg）")
    async with await _connect(args, key) as ws:
        await ws.send(json.dumps(_session_update(args.voice, ["text", "audio"], None)))
        # image buffer 的前置條件：先送過音訊（0.6 秒靜音就夠），且 turn_detection 必須關
        silence = b"\x00" * int(IN_RATE * 0.6) * 2
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(silence).decode()}))
        for p in args.image:
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            await ws.send(json.dumps({"type": "input_image_buffer.append", "image": b64}))
        for ev in _text_turn(args.prompt or "描述你看到的畫面。"):
            await ws.send(json.dumps(ev))
        _report(f"圖片輸入（{len(args.image)} 張）→ 輸出", await _collect(ws), args.save_audio)


async def test_voices(args, key):
    if not args.voices:
        sys.exit("--test voices 需要 --voices A,B,C（逗號分隔）")
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    # 先用一個必定不存在的音色確認 error 的形狀——收不到 error 就代表這個模型
    # 對非法音色不報錯，整輪「有音訊＝有效」的判準不成立，直接停下來。
    order = ["NoSuchVoiceZZZ"] + voices
    valid, invalid = [], []
    for v in order:
        async with await _connect(args, key) as ws:
            await ws.send(json.dumps(_session_update(v, ["text", "audio"], None)))
            for ev in _text_turn("說「好」。"):
                await ws.send(json.dumps(ev))
            r = await _collect(ws, deadline_s=45)
        ok = len(r["audio"]) > 0 and not r["errors"]
        mark = "✅" if ok else "❌"
        detail = _audio_stats(bytes(r["audio"])) if ok else "; ".join(r["errors"]) or "無音訊也無錯誤（？）"
        print(f"{mark} {v}: {detail}")
        if v == "NoSuchVoiceZZZ":
            if ok or not r["errors"]:
                sys.exit("⚠️ 哨兵音色沒有被拒——這個模型的音色驗證方式不同，這套判準不適用")
            continue
        (valid if ok else invalid).append(v)
    print(f"\n有效 {len(valid)}：{','.join(valid)}")
    print(f"無效 {len(invalid)}：{','.join(invalid)}")


async def test_turn(args, key):
    # ⚠️ 只能驗「送了會不會被拒」。沒被拒 ≠ 模式真的在運作，VAD 行為要用真實語音驗。
    for td in [{"type": "semantic_vad"}, {"type": "server_vad"}, {"type": "smart_turn"}, None]:
        async with await _connect(args, key) as ws:
            await ws.send(json.dumps(_session_update(args.voice, ["text", "audio"], td)))
            for ev in _text_turn("說「好」。"):
                await ws.send(json.dumps(ev))
            r = await _collect(ws, deadline_s=45)
        label = td["type"] if td else "null（手動）"
        ok = r["done"] and not r["errors"]
        print(f"{'✅' if ok else '❌'} turn_detection={label}: "
              + (_audio_stats(bytes(r["audio"])) if ok else "; ".join(r["errors"])))


TESTS = {"basic": test_basic, "audio": test_audio, "image": test_image,
         "voices": test_voices, "turn": test_turn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--gateway", choices=list(GATEWAYS), default="prod")
    ap.add_argument("--key-file")
    ap.add_argument("--test", choices=list(TESTS), default="basic")
    ap.add_argument("--voice", help="basic/audio/image/turn 用的音色（不帶就用模型預設）")
    ap.add_argument("--voices", help="voices 測試的候選清單，逗號分隔")
    ap.add_argument("--prompt")
    ap.add_argument("--wav", help="audio 測試的輸入（PCM16 16kHz mono wav）")
    ap.add_argument("--image", action="append", help="image 測試的輸入，可重複")
    ap.add_argument("--save-audio", help="把回覆音訊存成 wav（24kHz）")
    args = ap.parse_args()
    asyncio.run(TESTS[args.test](args, _key(args)))


if __name__ == "__main__":
    main()
