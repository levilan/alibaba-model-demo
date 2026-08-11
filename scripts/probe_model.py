#!/usr/bin/env python3
"""模型能力探測——新增模型時用這支，不要每次重寫一次性腳本。

用法：
    venv/bin/python scripts/probe_model.py <model-id> [--gateway prod|test] [--key-file PATH]
    venv/bin/python scripts/probe_model.py --drift          # 只做清單漂移比對

設計原則（都是踩過坑換來的，改動前先讀）：

1. **能不花錢就不花錢。** 不合法的參數會在驗證階段被拒、不會產生內容，所以「送一個
   必定違規的值」是免費的探測手段。合法的值才會真的生成並計費。

2. **「送得出去」不等於「能用」。** 閘道的驗證比原廠寬鬆，會放行原廠實際不支援的
   組合（例如萬相 2.7 送 4096*4096、組圖模式送 4K 都能過閘道，但官方文件明說不支援）。
   本腳本只回報「閘道接不接受」，**不等於該值可用**——最終要以原廠文件與實際產出為準。

3. **探測手法不能跨家族沿用。** 「用超範圍的 n 當哨兵」在萬相有效（size 先驗），但
   千問 3.0 是 n 先驗，導致連 10*10 都回「n 的錯誤」、看起來像尺寸通過了；而 MAI 與
   Seedream 根本不拒絕 n=13，會**真的產圖**（曾因此一次意外生成上百張）。所以哨兵
   模式預設關閉，要用必須先用一個已知非法的值確認該模型真的會拒絕哨兵。

4. **權限問題與參數問題要分開。** 送一個必定違規的參數：回 403 AccessDenied 表示在
   權限層就被擋（配額未開通），回 InvalidParameter 才表示權限通了、是參數的問題。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

GATEWAYS = {
    "prod": "https://nen.com.tw",
    "test": "http://192.168.0.245",
}

# 明顯非法、用來觸發驗證錯誤的值。合法的值才會計費，這些不會。
BAD_VALUES = {
    "size_star": "10*10",
    "size_x": "10x10",
    "resolution": "9999P",
}


def _key(args) -> str:
    if args.key_file:
        return Path(args.key_file).read_text().strip()
    env = os.environ.get("NENAI_API_KEY", "").strip()
    if env:
        return env
    sys.exit("需要金鑰：用 --key-file PATH 或設定環境變數 NENAI_API_KEY")


def _post(base: str, path: str, key: str, **kw) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"}
    if "json" in kw:
        headers["Content-Type"] = "application/json"
    return httpx.post(f"{base}{path}", headers=headers, timeout=180, **kw)


def _err(resp: httpx.Response) -> str:
    try:
        j = resp.json()
        return (j.get("error", {}) or {}).get("message") or j.get("message") or resp.text[:200]
    except Exception:
        return resp.text[:200]


def check_listed(base: str, key: str, model: str) -> bool:
    r = httpx.get(f"{base}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    ids = {m["id"] for m in r.json().get("data", [])}
    listed = model in ids
    print(f"  在 /v1/models 清單裡        : {'✅' if listed else '❌'}")
    if not listed:
        print("     ⚠️ 清單是依 key 的群組回傳的，看不到不代表全平台沒有——"
              "可能只是這個群組沒開，別急著從 MODELS 移除")
    return listed


def check_pricing(base: str, key: str, model: str) -> None:
    r = httpx.get(f"{base}/api/pricing", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    row = next((m for m in r.json().get("data", []) if m.get("model_name") == model), None)
    if not row:
        print("  計價                        : ❌ 計費表裡沒有這個模型")
        return
    if row.get("quota_type") == 1:
        print(f"  計價                        : 按次 ${row.get('model_price')}")
    else:
        ratio = row.get("model_ratio") or 0
        print(f"  計價                        : 按 token（ratio {ratio} → ${ratio * 2}/1M）")
        print("     ⚠️ 按 token 的模型，前端要能拿到 usage 才計得進「本次花費」")


def check_access(base: str, key: str, model: str, kind: str) -> None:
    """送必定違規的參數，用回應區分「配額未開」與「參數問題」。不會計費。"""
    if kind == "video":
        r = _post(base, "/v1/videos", key,
                  json={"model": model, "prompt": "probe",
                        "metadata": {"resolution": BAD_VALUES["resolution"]}})
    else:
        r = _post(base, "/v1/images/generations", key,
                  json={"model": model, "prompt": "probe", "n": 1,
                        "size": BAD_VALUES["size_star"]})
    msg = _err(r)
    if r.status_code == 200:
        print("  權限                        : ✅ 已開通（⚠️ 但這次意外建立了任務，會計費）")
    elif "AccessDenied" in msg or r.status_code == 403:
        print("  權限                        : ❌ AccessDenied — 配額／模型未開通，先別測參數")
    else:
        print(f"  權限                        : ✅ 已開通（收到參數錯誤：{msg[:80]}）")


def probe_sizes(base: str, key: str, model: str, sizes: list[str]) -> None:
    """逐一測尺寸。**合法的值會真的產圖並計費**，所以要明確傳入想測的清單。"""
    print(f"  尺寸（{len(sizes)} 個，合法值會產圖計費）:")
    for sz in sizes:
        r = _post(base, "/v1/images/generations", key,
                  json={"model": model, "prompt": "a grey dot", "n": 1, "size": sz})
        print(f"     {sz:14s} {'✅ 可用' if r.status_code == 200 else '❌ ' + _err(r)[:90]}")


def drift_check(base: str, key: str) -> int:
    """比對 MODELS ↔ /v1/models ↔ /api/pricing，回傳有問題的項數。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import app  # noqa: E402

    hdr = {"Authorization": f"Bearer {key}"}
    avail = {m["id"] for m in httpx.get(f"{base}/v1/models", headers=hdr, timeout=30).json()["data"]}
    price = {m["model_name"] for m in httpx.get(f"{base}/api/pricing", headers=hdr, timeout=60).json()["data"]}

    ours: set[str] = set()
    for k in ("text", "image", "video", "muleai"):
        ours |= {m["id"] for m in app.MODELS[k]}
    for k in ("asr", "tts"):
        ours |= {m["id"] for m in app.MODELS["voice"][k]}

    problems = 0
    missing = sorted(ours - avail)
    if missing:
        problems += len(missing)
        print(f"\n❌ MODELS 有、閘道清單沒有（{len(missing)}）——可能是群組權限，先確認再決定移除：")
        for m in missing:
            print(f"     {m}")

    nopay = sorted(ours & avail - price)
    if nopay:
        problems += len(nopay)
        print(f"\n⚠️ 有模型但計費表查不到（{len(nopay)}）——UI 會顯示不出單價：")
        for m in nopay:
            print(f"     {m}")

    extra = sorted(avail - ours)
    if extra:
        print(f"\nℹ️ 閘道有、我們沒收錄（{len(extra)}）——不是錯誤，但值得定期檢視是否要上架：")
        for m in extra:
            print(f"     {m}")

    if not problems:
        print("\n✅ 沒有發現漂移問題")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="要探測的模型 id")
    ap.add_argument("--gateway", choices=list(GATEWAYS), default="prod")
    ap.add_argument("--key-file")
    ap.add_argument("--kind", choices=["image", "video"], default="image")
    ap.add_argument("--sizes", help="逗號分隔的尺寸清單。⚠️ 合法值會產圖並計費")
    ap.add_argument("--drift", action="store_true", help="只做清單漂移比對（不花錢）")
    args = ap.parse_args()

    base, key = GATEWAYS[args.gateway], _key(args)
    print(f"網關：{base}（{args.gateway}）")

    if args.drift:
        return 1 if drift_check(base, key) else 0

    if not args.model:
        ap.error("請指定模型 id，或用 --drift 做清單比對")

    print(f"模型：{args.model}\n")
    if not check_listed(base, key, args.model):
        return 1
    check_pricing(base, key, args.model)
    check_access(base, key, args.model, args.kind)
    if args.sizes:
        probe_sizes(base, key, args.model, [x.strip() for x in args.sizes.split(",") if x.strip()])
    print("\n提醒：本腳本只回報「閘道接不接受」。閘道驗證比原廠寬鬆，"
          "最終仍要對照原廠文件並確認實際產出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
